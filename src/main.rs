use std::{
    convert::Infallible,
    env,
    net::SocketAddr,
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result};
use axum::{
    Json, Router,
    extract::{Request, State, rejection::JsonRejection},
    http::{StatusCode, header::AUTHORIZATION},
    middleware::{self, Next},
    response::{
        IntoResponse, Response,
        sse::{Event, KeepAlive, Sse},
    },
    routing::{get, post},
};
use github_copilot_sdk::{
    Client, ClientOptions, MessageOptions, SessionConfig, SystemMessageConfig, session::Session,
    subscription::EventSubscription,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use subtle::ConstantTimeEq;
use tokio::{net::TcpListener, sync::mpsc};
use tokio_stream::wrappers::ReceiverStream;
use tracing::{error, info, warn};
use uuid::Uuid;

const SERVICE_NAME: &str = "copilot-openai-api";

#[derive(Clone)]
struct AppState {
    client: Arc<Client>,
    config: Arc<Config>,
}

#[derive(Debug)]
struct Config {
    listen_addr: SocketAddr,
    default_model: String,
    request_timeout: Duration,
    api_key: Option<String>,
}

impl Config {
    fn from_env() -> Result<Self> {
        let host = env::var("HOST").unwrap_or_else(|_| "127.0.0.1".to_string());
        let port = env::var("PORT")
            .unwrap_or_else(|_| "3000".to_string())
            .parse::<u16>()
            .context("PORT must be a valid TCP port")?;
        let request_timeout_seconds = env::var("REQUEST_TIMEOUT_SECONDS")
            .unwrap_or_else(|_| "120".to_string())
            .parse::<u64>()
            .context("REQUEST_TIMEOUT_SECONDS must be a positive integer")?;

        if request_timeout_seconds == 0 {
            anyhow::bail!("REQUEST_TIMEOUT_SECONDS must be greater than zero");
        }

        let listen_addr = format!("{host}:{port}")
            .parse()
            .context("HOST and PORT must form a valid socket address")?;
        let default_model = non_empty_env("DEFAULT_MODEL").unwrap_or_else(|| "auto".to_string());

        Ok(Self {
            listen_addr,
            default_model,
            request_timeout: Duration::from_secs(request_timeout_seconds),
            api_key: non_empty_env("API_KEY"),
        })
    }
}

fn non_empty_env(name: &str) -> Option<String> {
    env::var(name).ok().filter(|value| !value.trim().is_empty())
}

#[derive(Debug, Deserialize)]
struct ChatCompletionRequest {
    #[serde(default)]
    model: Option<String>,
    messages: Vec<ChatMessage>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    n: Option<u32>,
    #[serde(default)]
    tools: Option<Value>,
}

#[derive(Debug, Deserialize)]
struct ChatMessage {
    role: String,
    content: Option<Value>,
}

#[derive(Serialize)]
struct HealthResponse {
    status: &'static str,
}

#[derive(Serialize)]
struct ModelListResponse {
    object: &'static str,
    data: Vec<ModelResponse>,
}

#[derive(Serialize)]
struct ModelResponse {
    id: String,
    object: &'static str,
    created: u64,
    owned_by: &'static str,
}

#[derive(Serialize)]
struct ChatCompletionResponse {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<ChatCompletionChoice>,
}

#[derive(Serialize)]
struct ChatCompletionChoice {
    index: u32,
    message: AssistantMessage,
    finish_reason: &'static str,
}

#[derive(Serialize)]
struct AssistantMessage {
    role: &'static str,
    content: String,
}

#[derive(Serialize)]
struct ChatCompletionChunk {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<ChatCompletionChunkChoice>,
}

#[derive(Serialize)]
struct ChatCompletionChunkChoice {
    index: u32,
    delta: AssistantDelta,
    #[serde(skip_serializing_if = "Option::is_none")]
    finish_reason: Option<&'static str>,
}

#[derive(Default, Serialize)]
struct AssistantDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
}

struct PromptParts {
    system_message: Option<String>,
    prompt: String,
}

#[derive(Debug)]
struct ApiError {
    status: StatusCode,
    message: String,
    error_type: &'static str,
    code: Option<&'static str>,
}

impl ApiError {
    fn invalid_request(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: message.into(),
            error_type: "invalid_request_error",
            code: None,
        }
    }

    fn unauthorized() -> Self {
        Self {
            status: StatusCode::UNAUTHORIZED,
            message: "Invalid API key".to_string(),
            error_type: "authentication_error",
            code: Some("invalid_api_key"),
        }
    }

    fn unsupported(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: message.into(),
            error_type: "invalid_request_error",
            code: Some("unsupported_parameter"),
        }
    }

    fn upstream(error: impl std::fmt::Display) -> Self {
        error!(%error, "Copilot SDK request failed");
        Self {
            status: StatusCode::BAD_GATEWAY,
            message: "The Copilot backend could not complete the request".to_string(),
            error_type: "api_error",
            code: Some("copilot_backend_error"),
        }
    }

    fn timeout() -> Self {
        Self {
            status: StatusCode::GATEWAY_TIMEOUT,
            message: "The Copilot backend did not finish before the request timeout".to_string(),
            error_type: "timeout_error",
            code: Some("copilot_timeout"),
        }
    }

    fn as_json(&self) -> Value {
        json!({
            "error": {
                "message": self.message,
                "type": self.error_type,
                "code": self.code,
            }
        })
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.status, Json(self.as_json())).into_response()
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    let config = Arc::new(Config::from_env()?);
    if config.api_key.is_none() && !config.listen_addr.ip().is_loopback() {
        warn!(address = %config.listen_addr, "API_KEY is unset on a non-loopback address");
    }

    let client = Arc::new(
        Client::start(ClientOptions::default())
            .await
            .context("failed to start the GitHub Copilot SDK client")?,
    );
    let state = AppState {
        client: Arc::clone(&client),
        config: Arc::clone(&config),
    };

    let protected_api = Router::new()
        .route("/chat/completions", post(chat_completions))
        .route("/models", get(list_models))
        .route_layer(middleware::from_fn_with_state(
            state.clone(),
            require_api_key,
        ));
    let app = Router::new()
        .route("/health", get(health))
        .nest("/v1", protected_api)
        .with_state(state);

    let listener = TcpListener::bind(config.listen_addr)
        .await
        .with_context(|| format!("failed to bind {}", config.listen_addr))?;
    info!(address = %config.listen_addr, "OpenAI-compatible API is listening");

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .context("HTTP server failed")?;

    client
        .stop()
        .await
        .context("failed to stop the Copilot SDK client")?;
    Ok(())
}

async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C signal handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install SIGTERM signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        () = ctrl_c => {},
        () = terminate => {},
    }
}

async fn require_api_key(State(state): State<AppState>, request: Request, next: Next) -> Response {
    let Some(expected_key) = state.config.api_key.as_deref() else {
        return next.run(request).await;
    };

    let supplied_key = request
        .headers()
        .get(AUTHORIZATION)
        .and_then(|header| header.to_str().ok())
        .and_then(|header| header.strip_prefix("Bearer "));
    let authenticated = supplied_key
        .map(|key| expected_key.as_bytes().ct_eq(key.as_bytes()).into())
        .unwrap_or(false);

    if authenticated {
        next.run(request).await
    } else {
        ApiError::unauthorized().into_response()
    }
}

async fn health() -> Json<HealthResponse> {
    Json(HealthResponse { status: "ok" })
}

async fn list_models(State(state): State<AppState>) -> Json<ModelListResponse> {
    Json(ModelListResponse {
        object: "list",
        data: vec![ModelResponse {
            id: state.config.default_model.clone(),
            object: "model",
            created: 0,
            owned_by: "github-copilot",
        }],
    })
}

async fn chat_completions(
    State(state): State<AppState>,
    request: Result<Json<ChatCompletionRequest>, JsonRejection>,
) -> Result<Response, ApiError> {
    let Json(request) = request.map_err(|error| ApiError::invalid_request(error.to_string()))?;
    validate_request(&request)?;

    let model = request
        .model
        .as_deref()
        .filter(|model| !model.trim().is_empty())
        .unwrap_or(&state.config.default_model)
        .to_string();
    let prompt_parts = compile_prompt(&request.messages).map_err(ApiError::invalid_request)?;
    let completion_id = format!("chatcmpl-{}", Uuid::new_v4().simple());
    let created = unix_timestamp();

    if request.stream {
        stream_chat_completion(state, model, prompt_parts, completion_id, created).await
    } else {
        complete_chat_completion(state, model, prompt_parts, completion_id, created).await
    }
}

fn validate_request(request: &ChatCompletionRequest) -> Result<(), ApiError> {
    if request.messages.is_empty() {
        return Err(ApiError::invalid_request(
            "messages must contain at least one message",
        ));
    }
    if request.n.unwrap_or(1) != 1 {
        return Err(ApiError::unsupported("Only n=1 is supported"));
    }
    if request.tools.is_some() {
        return Err(ApiError::unsupported(
            "Tool calling is not supported by this OpenAI-compatible endpoint",
        ));
    }
    Ok(())
}

async fn complete_chat_completion(
    state: AppState,
    model: String,
    prompt_parts: PromptParts,
    completion_id: String,
    created: u64,
) -> Result<Response, ApiError> {
    let session = create_session(&state, &model, &prompt_parts, false).await?;
    let result = session
        .send_and_wait(
            MessageOptions::new(prompt_parts.prompt)
                .with_wait_timeout(state.config.request_timeout),
        )
        .await;
    let _ = session.disconnect().await;

    let event = match result {
        Ok(Some(event)) => event,
        Ok(None) => {
            return Err(ApiError::upstream(
                "Copilot finished without an assistant response",
            ));
        }
        Err(error) if error.to_string().contains("Timeout") => return Err(ApiError::timeout()),
        Err(error) => return Err(ApiError::upstream(error)),
    };
    let content = assistant_content(&event.data).ok_or_else(|| {
        ApiError::upstream("Copilot returned an assistant message without textual content")
    })?;

    Ok(Json(ChatCompletionResponse {
        id: completion_id,
        object: "chat.completion",
        created,
        model,
        choices: vec![ChatCompletionChoice {
            index: 0,
            message: AssistantMessage {
                role: "assistant",
                content,
            },
            finish_reason: "stop",
        }],
    })
    .into_response())
}

async fn stream_chat_completion(
    state: AppState,
    model: String,
    prompt_parts: PromptParts,
    completion_id: String,
    created: u64,
) -> Result<Response, ApiError> {
    let config = session_config(&model, prompt_parts.system_message.clone(), true);
    let prepared = state
        .client
        .prepare_session(config)
        .map_err(ApiError::upstream)?;
    let events = prepared.subscribe();
    let session = prepared.start().await.map_err(ApiError::upstream)?;
    let (sender, receiver) = mpsc::channel(32);
    let timeout = state.config.request_timeout;

    tokio::spawn(async move {
        stream_session(
            session,
            events,
            sender,
            model,
            prompt_parts.prompt,
            completion_id,
            created,
            timeout,
        )
        .await;
    });

    Ok(Sse::new(ReceiverStream::new(receiver))
        .keep_alive(KeepAlive::default())
        .into_response())
}

async fn stream_session(
    session: Session,
    mut events: EventSubscription,
    sender: mpsc::Sender<Result<Event, Infallible>>,
    model: String,
    prompt: String,
    completion_id: String,
    created: u64,
    timeout: Duration,
) {
    let role_chunk = ChatCompletionChunk {
        id: completion_id.clone(),
        object: "chat.completion.chunk",
        created,
        model: model.clone(),
        choices: vec![ChatCompletionChunkChoice {
            index: 0,
            delta: AssistantDelta {
                role: Some("assistant"),
                content: None,
            },
            finish_reason: None,
        }],
    };
    if send_sse_json(&sender, &role_chunk).await.is_err() {
        let _ = session.disconnect().await;
        return;
    }

    if let Err(error) = session.send(prompt).await {
        error!(%error, "failed to send prompt to Copilot");
        let _ = send_sse_error(&sender, "The Copilot backend could not start the request").await;
        let _ = session.disconnect().await;
        return;
    }

    let deadline = tokio::time::Instant::now() + timeout;
    let mut received_delta = false;
    loop {
        let event = match tokio::time::timeout_at(deadline, events.recv()).await {
            Ok(Ok(event)) => event,
            Ok(Err(error)) => {
                error!(%error, "Copilot event subscription ended unexpectedly");
                let _ =
                    send_sse_error(&sender, "The Copilot event stream ended unexpectedly").await;
                break;
            }
            Err(_) => {
                let _ = send_sse_error(
                    &sender,
                    "The Copilot backend did not finish before the request timeout",
                )
                .await;
                break;
            }
        };

        match event.event_type.as_str() {
            "assistant.message_delta" => {
                if let Some(delta) = event.data.get("delta").and_then(Value::as_str) {
                    received_delta = true;
                    if send_sse_chunk(
                        &sender,
                        &completion_id,
                        created,
                        &model,
                        Some(delta.to_string()),
                        None,
                    )
                    .await
                    .is_err()
                    {
                        break;
                    }
                }
            }
            "assistant.message" if !received_delta => {
                if let Some(content) = assistant_content(&event.data) {
                    if send_sse_chunk(
                        &sender,
                        &completion_id,
                        created,
                        &model,
                        Some(content),
                        None,
                    )
                    .await
                    .is_err()
                    {
                        break;
                    }
                }
            }
            "session.error" => {
                let message = event
                    .data
                    .get("message")
                    .and_then(Value::as_str)
                    .unwrap_or("The Copilot backend returned an error");
                let _ = send_sse_error(&sender, message).await;
                break;
            }
            "session.idle" => {
                let _ =
                    send_sse_chunk(&sender, &completion_id, created, &model, None, Some("stop"))
                        .await;
                let _ = sender.send(Ok(Event::default().data("[DONE]"))).await;
                break;
            }
            _ => {}
        }
    }

    let _ = session.disconnect().await;
}

async fn create_session(
    state: &AppState,
    model: &str,
    prompt_parts: &PromptParts,
    streaming: bool,
) -> Result<Session, ApiError> {
    state
        .client
        .create_session(session_config(
            model,
            prompt_parts.system_message.clone(),
            streaming,
        ))
        .await
        .map_err(ApiError::upstream)
}

fn session_config(model: &str, system_message: Option<String>, streaming: bool) -> SessionConfig {
    let mut config = SessionConfig::default()
        .with_model(model)
        .with_client_name(SERVICE_NAME)
        .with_streaming(streaming)
        .deny_all_permissions();

    if let Some(content) = system_message {
        config = config.with_system_message(SystemMessageConfig::new().with_content(content));
    }

    config
}

async fn send_sse_json<T: Serialize>(
    sender: &mpsc::Sender<Result<Event, Infallible>>,
    value: &T,
) -> Result<(), ()> {
    let data = serde_json::to_string(value).map_err(|error| {
        error!(%error, "failed to serialize SSE response chunk");
    })?;
    sender
        .send(Ok(Event::default().data(data)))
        .await
        .map_err(|_| ())
}

async fn send_sse_chunk(
    sender: &mpsc::Sender<Result<Event, Infallible>>,
    completion_id: &str,
    created: u64,
    model: &str,
    content: Option<String>,
    finish_reason: Option<&'static str>,
) -> Result<(), ()> {
    send_sse_json(
        sender,
        &ChatCompletionChunk {
            id: completion_id.to_string(),
            object: "chat.completion.chunk",
            created,
            model: model.to_string(),
            choices: vec![ChatCompletionChunkChoice {
                index: 0,
                delta: AssistantDelta {
                    role: None,
                    content,
                },
                finish_reason,
            }],
        },
    )
    .await
}

async fn send_sse_error(
    sender: &mpsc::Sender<Result<Event, Infallible>>,
    message: &str,
) -> Result<(), ()> {
    sender
        .send(Ok(Event::default().data(
            json!({
                "error": {
                    "message": message,
                    "type": "api_error",
                    "code": "copilot_backend_error",
                }
            })
            .to_string(),
        )))
        .await
        .map_err(|_| ())
}

fn compile_prompt(messages: &[ChatMessage]) -> Result<PromptParts, String> {
    let mut system_messages = Vec::new();
    let mut conversation = String::from(
        "Continue the following chat conversation. Answer the latest user request directly.\n\n",
    );

    for message in messages {
        let role = message.role.trim().to_ascii_lowercase();
        let content = message_content(message.content.as_ref())?;

        match role.as_str() {
            "system" | "developer" => system_messages.push(content),
            "user" | "assistant" | "tool" => {
                conversation.push_str(&format!("[{}]\n{}\n\n", role.to_ascii_uppercase(), content));
            }
            _ => return Err(format!("Unsupported message role: {}", message.role)),
        }
    }

    if !conversation.contains("[USER]") {
        return Err("messages must contain at least one user message".to_string());
    }

    conversation.push_str("[ASSISTANT]\n");
    Ok(PromptParts {
        system_message: (!system_messages.is_empty()).then(|| system_messages.join("\n\n")),
        prompt: conversation,
    })
}

fn message_content(content: Option<&Value>) -> Result<String, String> {
    let Some(content) = content else {
        return Ok(String::new());
    };

    match content {
        Value::String(text) => Ok(text.clone()),
        Value::Array(parts) => parts
            .iter()
            .filter_map(|part| {
                part.get("text")
                    .and_then(Value::as_str)
                    .or_else(|| part.get("content").and_then(Value::as_str))
            })
            .map(ToOwned::to_owned)
            .collect::<Vec<_>>()
            .join("")
            .pipe(Ok),
        Value::Null => Ok(String::new()),
        _ => Err("message content must be a string or an array of text parts".to_string()),
    }
}

fn assistant_content(data: &Value) -> Option<String> {
    data.get("content")
        .and_then(Value::as_str)
        .or_else(|| data.get("message").and_then(Value::as_str))
        .map(ToOwned::to_owned)
}

fn unix_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

trait Pipe: Sized {
    fn pipe<T>(self, function: impl FnOnce(Self) -> T) -> T {
        function(self)
    }
}

impl<T> Pipe for T {}

#[cfg(test)]
mod tests {
    use super::*;

    fn message(role: &str, content: Value) -> ChatMessage {
        ChatMessage {
            role: role.to_string(),
            content: Some(content),
        }
    }

    #[test]
    fn compiles_system_and_conversation_messages() {
        let prompt = compile_prompt(&[
            message("system", json!("Be concise.")),
            message("user", json!("Hello")),
            message("assistant", json!("Hi")),
            message("user", json!("Explain Axum")),
        ])
        .expect("prompt should compile");

        assert_eq!(prompt.system_message.as_deref(), Some("Be concise."));
        assert!(prompt.prompt.contains("[USER]\nHello"));
        assert!(prompt.prompt.contains("[ASSISTANT]\nHi"));
        assert!(prompt.prompt.ends_with("[ASSISTANT]\n"));
    }

    #[test]
    fn joins_text_content_parts() {
        let content = message_content(Some(&json!([
            { "type": "text", "text": "hello " },
            { "type": "input_text", "text": "world" }
        ])))
        .expect("content should be supported");

        assert_eq!(content, "hello world");
    }

    #[test]
    fn rejects_unknown_message_roles() {
        let result = compile_prompt(&[message("function", json!("ignored"))]);
        assert!(result.is_err());
    }
}

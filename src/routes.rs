use std::{
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use axum::{
    Json, Router,
    extract::{Request, State, rejection::JsonRejection},
    http::header::AUTHORIZATION,
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use github_copilot_sdk::Client;
use subtle::ConstantTimeEq;
use uuid::Uuid;

use crate::{
    config::Config,
    copilot::{complete_chat_completion, stream_chat_completion},
    error::ApiError,
    openai::{
        ChatCompletionRequest, HealthResponse, ModelListResponse, ModelResponse, compile_prompt,
        normalize_tools,
    },
};

#[derive(Clone)]
pub(crate) struct AppState {
    pub(crate) client: Arc<Client>,
    pub(crate) config: Arc<Config>,
}

pub(crate) fn app(state: AppState) -> Router {
    let protected_api = Router::new()
        .route("/chat/completions", post(chat_completions))
        .route("/models", get(list_models))
        .route_layer(middleware::from_fn_with_state(
            state.clone(),
            require_api_key,
        ));

    Router::new()
        .route("/health", get(health))
        .nest("/v1", protected_api)
        .with_state(state)
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
    let tools = normalize_tools(request.tools.as_deref()).map_err(ApiError::invalid_request)?;

    let model = request
        .model
        .as_deref()
        .filter(|model| !model.trim().is_empty())
        .unwrap_or(&state.config.default_model)
        .to_string();
    let prompt_parts =
        compile_prompt(&request.messages, &tools).map_err(ApiError::invalid_request)?;
    let completion_id = format!("chatcmpl-{}", Uuid::new_v4().simple());
    let created = unix_timestamp();

    if request.stream {
        stream_chat_completion(state, model, prompt_parts, completion_id, created, &tools).await
    } else {
        complete_chat_completion(state, model, prompt_parts, completion_id, created, &tools).await
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
    Ok(())
}

fn unix_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

use std::{convert::Infallible, time::Duration};

use axum::{
    Json,
    response::{
        IntoResponse, Response,
        sse::{Event, KeepAlive, Sse},
    },
};
use github_copilot_sdk::{
    MessageOptions, SessionConfig, SystemMessageConfig, session::Session,
    subscription::EventSubscription,
};
use serde::Serialize;
use serde_json::{Value, json};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tracing::error;

use crate::{
    error::ApiError,
    openai::{
        AssistantDelta, AssistantMessage, AssistantToolCall, ChatCompletionChoice,
        ChatCompletionChunk, ChatCompletionChunkChoice, ChatCompletionResponse, PromptParts,
        ToolDefinition, assistant_content, parse_tool_calls, tool_call_deltas,
    },
    routes::AppState,
};

const SERVICE_NAME: &str = "copilot-openai-api";

pub(crate) async fn complete_chat_completion(
    state: AppState,
    model: String,
    prompt_parts: PromptParts,
    completion_id: String,
    created: u64,
    tools: &[ToolDefinition],
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
    let tool_calls = parse_tool_calls(&content, tools);
    let finish_reason = if tool_calls.is_some() {
        "tool_calls"
    } else {
        "stop"
    };

    Ok(Json(ChatCompletionResponse {
        id: completion_id,
        object: "chat.completion",
        created,
        model,
        choices: vec![ChatCompletionChoice {
            index: 0,
            message: AssistantMessage {
                role: "assistant",
                content: tool_calls.is_none().then_some(content),
                tool_calls,
            },
            finish_reason,
        }],
    })
    .into_response())
}

pub(crate) async fn stream_chat_completion(
    state: AppState,
    model: String,
    prompt_parts: PromptParts,
    completion_id: String,
    created: u64,
    tools: &[ToolDefinition],
) -> Result<Response, ApiError> {
    if !tools.is_empty() {
        return stream_tool_chat_completion(
            state,
            model,
            prompt_parts,
            completion_id,
            created,
            tools,
        )
        .await;
    }

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

async fn stream_tool_chat_completion(
    state: AppState,
    model: String,
    prompt_parts: PromptParts,
    completion_id: String,
    created: u64,
    tools: &[ToolDefinition],
) -> Result<Response, ApiError> {
    let session = create_session(&state, &model, &prompt_parts, false).await?;
    let (sender, receiver) = mpsc::channel(32);
    let timeout = state.config.request_timeout;
    let prompt = prompt_parts.prompt;
    let tools = tools.to_vec();

    tokio::spawn(async move {
        stream_tool_session(
            session,
            sender,
            model,
            prompt,
            completion_id,
            created,
            timeout,
            tools,
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
                tool_calls: None,
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

async fn stream_tool_session(
    session: Session,
    sender: mpsc::Sender<Result<Event, Infallible>>,
    model: String,
    prompt: String,
    completion_id: String,
    created: u64,
    timeout: Duration,
    tools: Vec<ToolDefinition>,
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
                tool_calls: None,
            },
            finish_reason: None,
        }],
    };
    if send_sse_json(&sender, &role_chunk).await.is_err() {
        let _ = session.disconnect().await;
        return;
    }

    let result = session
        .send_and_wait(MessageOptions::new(prompt).with_wait_timeout(timeout))
        .await;
    let _ = session.disconnect().await;

    let content = match result {
        Ok(Some(event)) => match assistant_content(&event.data) {
            Some(content) => content,
            None => {
                let _ = send_sse_error(
                    &sender,
                    "Copilot returned an assistant message without textual content",
                )
                .await;
                return;
            }
        },
        Ok(None) => {
            let _ = send_sse_error(&sender, "Copilot finished without an assistant response").await;
            return;
        }
        Err(error) if error.to_string().contains("Timeout") => {
            let _ = send_sse_error(
                &sender,
                "The Copilot backend did not finish before the request timeout",
            )
            .await;
            return;
        }
        Err(error) => {
            error!(%error, "Copilot SDK request failed");
            let _ = send_sse_error(
                &sender,
                "The Copilot backend could not complete the request",
            )
            .await;
            return;
        }
    };

    if let Some(tool_calls) = parse_tool_calls(&content, &tools) {
        if send_sse_tool_calls(&sender, &completion_id, created, &model, &tool_calls)
            .await
            .is_err()
        {
            return;
        }
        let _ = send_sse_chunk(
            &sender,
            &completion_id,
            created,
            &model,
            None,
            Some("tool_calls"),
        )
        .await;
    } else {
        let _ = send_sse_chunk(
            &sender,
            &completion_id,
            created,
            &model,
            Some(content),
            Some("stop"),
        )
        .await;
    }

    let _ = sender.send(Ok(Event::default().data("[DONE]"))).await;
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
                    tool_calls: None,
                },
                finish_reason,
            }],
        },
    )
    .await
}

async fn send_sse_tool_calls(
    sender: &mpsc::Sender<Result<Event, Infallible>>,
    completion_id: &str,
    created: u64,
    model: &str,
    tool_calls: &[AssistantToolCall],
) -> Result<(), ()> {
    let tool_calls = tool_call_deltas(tool_calls);

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
                    content: None,
                    tool_calls: Some(tool_calls),
                },
                finish_reason: None,
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

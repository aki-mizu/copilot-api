use axum::{
    Json,
    response::{
        IntoResponse, Response,
        sse::{KeepAlive, Sse},
    },
};
use github_copilot_sdk::MessageOptions;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use crate::{
    error::ApiError,
    openai::{
        AssistantMessage, ChatCompletionChoice, ChatCompletionResponse, PromptParts,
        ToolDefinition, assistant_content, parse_tool_calls,
    },
    routes::AppState,
};

use super::{
    session::{create_session, session_config},
    stream::{stream_session, stream_tool_session},
};

pub(crate) async fn complete_chat_completion(
    state: AppState,
    model: String,
    prompt_parts: PromptParts,
    completion_id: String,
    created: u64,
    reasoning_effort: Option<String>,
    tools: &[ToolDefinition],
) -> Result<Response, ApiError> {
    let session = create_session(
        &state,
        &model,
        &prompt_parts,
        false,
        reasoning_effort.as_deref(),
    )
    .await?;
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
    reasoning_effort: Option<String>,
    tools: &[ToolDefinition],
) -> Result<Response, ApiError> {
    if !tools.is_empty() {
        return stream_tool_chat_completion(
            state,
            model,
            prompt_parts,
            completion_id,
            created,
            reasoning_effort,
            tools,
        )
        .await;
    }

    let config = session_config(
        &model,
        prompt_parts.system_message.clone(),
        true,
        reasoning_effort.as_deref(),
    );
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
    reasoning_effort: Option<String>,
    tools: &[ToolDefinition],
) -> Result<Response, ApiError> {
    let session = create_session(
        &state,
        &model,
        &prompt_parts,
        false,
        reasoning_effort.as_deref(),
    )
    .await?;
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

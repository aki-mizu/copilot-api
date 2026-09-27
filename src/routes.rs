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
use tracing::warn;
use uuid::Uuid;

use crate::{
    config::Config,
    copilot::{complete_chat_completion, stream_chat_completion},
    error::ApiError,
    openai::{
        ChatCompletionRequest, HealthResponse, ModelListResponse, ModelResponse, compile_prompt,
        normalize_reasoning_effort, normalize_tools,
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
    let available_models = match state.client.list_models().await {
        Ok(models) => models.into_iter().map(|model| model.id).collect(),
        Err(error) => {
            warn!(%error, "could not list available Copilot models; returning the configured default");
            Vec::new()
        }
    };

    Json(model_list_response(
        &state.config.default_model,
        available_models,
    ))
}

fn model_list_response(
    default_model: &str,
    available_models: impl IntoIterator<Item = String>,
) -> ModelListResponse {
    let mut model_ids = vec![default_model.to_string()];
    for model_id in available_models {
        if !model_id.trim().is_empty() && !model_ids.iter().any(|id| id == &model_id) {
            model_ids.push(model_id);
        }
    }

    ModelListResponse {
        object: "list",
        data: model_ids
            .into_iter()
            .map(|id| ModelResponse {
                id,
                object: "model",
                created: 0,
                owned_by: "github-copilot",
            })
            .collect(),
    }
}

async fn chat_completions(
    State(state): State<AppState>,
    request: Result<Json<ChatCompletionRequest>, JsonRejection>,
) -> Result<Response, ApiError> {
    let Json(request) = request.map_err(|error| ApiError::invalid_request(error.to_string()))?;
    validate_request(&request)?;
    let tools = normalize_tools(request.tools.as_deref()).map_err(ApiError::invalid_request)?;
    let reasoning_effort = normalize_reasoning_effort(request.reasoning_effort.as_deref())
        .map_err(ApiError::invalid_request)?;

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
        stream_chat_completion(
            state,
            model,
            prompt_parts,
            completion_id,
            created,
            reasoning_effort,
            &tools,
        )
        .await
    } else {
        complete_chat_completion(
            state,
            model,
            prompt_parts,
            completion_id,
            created,
            reasoning_effort,
            &tools,
        )
        .await
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lists_default_then_unique_discovered_models() {
        let response = model_list_response(
            "auto",
            [
                "gpt-5".to_string(),
                "auto".to_string(),
                "claude-sonnet-4".to_string(),
                "gpt-5".to_string(),
                "".to_string(),
            ],
        );
        let data = serde_json::to_value(response).expect("model list should serialize");

        assert_eq!(data["data"][0]["id"], "auto");
        assert_eq!(data["data"][1]["id"], "gpt-5");
        assert_eq!(data["data"][2]["id"], "claude-sonnet-4");
        assert_eq!(data["data"].as_array().map(Vec::len), Some(3));
    }
}

use serde::Serialize;
use serde_json::Value;

use super::tools::{AssistantToolCall, AssistantToolCallDelta};

#[derive(Serialize)]
pub(crate) struct HealthResponse {
    pub(crate) status: &'static str,
}

#[derive(Serialize)]
pub(crate) struct ModelListResponse {
    pub(crate) object: &'static str,
    pub(crate) data: Vec<ModelResponse>,
}

#[derive(Serialize)]
pub(crate) struct ModelResponse {
    pub(crate) id: String,
    pub(crate) object: &'static str,
    pub(crate) created: u64,
    pub(crate) owned_by: &'static str,
}

#[derive(Serialize)]
pub(crate) struct ChatCompletionResponse {
    pub(crate) id: String,
    pub(crate) object: &'static str,
    pub(crate) created: u64,
    pub(crate) model: String,
    pub(crate) choices: Vec<ChatCompletionChoice>,
}

#[derive(Serialize)]
pub(crate) struct ChatCompletionChoice {
    pub(crate) index: u32,
    pub(crate) message: AssistantMessage,
    pub(crate) finish_reason: &'static str,
}

#[derive(Serialize)]
pub(crate) struct AssistantMessage {
    pub(crate) role: &'static str,
    pub(crate) content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) tool_calls: Option<Vec<AssistantToolCall>>,
}

#[derive(Serialize)]
pub(crate) struct ChatCompletionChunk {
    pub(crate) id: String,
    pub(crate) object: &'static str,
    pub(crate) created: u64,
    pub(crate) model: String,
    pub(crate) choices: Vec<ChatCompletionChunkChoice>,
}

#[derive(Serialize)]
pub(crate) struct ChatCompletionChunkChoice {
    pub(crate) index: u32,
    pub(crate) delta: AssistantDelta,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) finish_reason: Option<&'static str>,
}

#[derive(Default, Serialize)]
pub(crate) struct AssistantDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) role: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) tool_calls: Option<Vec<AssistantToolCallDelta>>,
}

pub(crate) fn assistant_content(data: &Value) -> Option<String> {
    data.get("content")
        .and_then(Value::as_str)
        .or_else(|| data.get("message").and_then(Value::as_str))
        .map(ToOwned::to_owned)
}

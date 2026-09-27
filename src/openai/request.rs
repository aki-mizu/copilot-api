use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::tools::OpenAiTool;

#[derive(Debug, Deserialize)]
pub(crate) struct ChatCompletionRequest {
    #[serde(default)]
    pub(crate) model: Option<String>,
    pub(crate) messages: Vec<ChatMessage>,
    #[serde(default)]
    pub(crate) stream: bool,
    #[serde(default)]
    pub(crate) n: Option<u32>,
    #[serde(default)]
    pub(crate) reasoning_effort: Option<String>,
    #[serde(default)]
    pub(crate) tools: Option<Vec<OpenAiTool>>,
}

pub(crate) fn normalize_reasoning_effort(
    reasoning_effort: Option<&str>,
) -> Result<Option<String>, String> {
    let Some(reasoning_effort) = reasoning_effort else {
        return Ok(None);
    };

    let reasoning_effort = reasoning_effort.trim().to_ascii_lowercase();
    match reasoning_effort.as_str() {
        "none" | "low" | "medium" | "high" | "xhigh" | "max" => Ok(Some(reasoning_effort)),
        _ => {
            Err("reasoning_effort must be one of: none, low, medium, high, xhigh, max".to_string())
        }
    }
}

#[derive(Debug, Deserialize)]
pub(crate) struct ChatMessage {
    pub(crate) role: String,
    pub(crate) content: Option<Value>,
    #[serde(default)]
    pub(crate) tool_call_id: Option<String>,
    #[serde(default)]
    pub(crate) tool_calls: Option<Vec<IncomingToolCall>>,
}

#[derive(Debug, Deserialize, Serialize)]
pub(crate) struct IncomingToolCall {
    pub(crate) id: String,
    #[serde(rename = "type")]
    pub(crate) kind: String,
    pub(crate) function: IncomingToolFunction,
}

#[derive(Debug, Deserialize, Serialize)]
pub(crate) struct IncomingToolFunction {
    pub(crate) name: String,
    pub(crate) arguments: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_supported_reasoning_effort() {
        assert_eq!(
            normalize_reasoning_effort(Some(" HIGH ")),
            Ok(Some("high".to_string()))
        );
        assert_eq!(normalize_reasoning_effort(None), Ok(None));
        assert!(normalize_reasoning_effort(Some("extra-high")).is_err());
    }
}

use std::collections::HashSet;

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use uuid::Uuid;

#[derive(Debug, Deserialize)]
pub(crate) struct OpenAiTool {
    #[serde(rename = "type")]
    kind: String,
    function: OpenAiFunction,
}

#[derive(Debug, Deserialize)]
struct OpenAiFunction {
    name: String,
    #[serde(default)]
    description: Option<String>,
    #[serde(default)]
    parameters: Option<Value>,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct ToolDefinition {
    name: String,
    description: String,
    parameters: Value,
}

pub(crate) fn normalize_tools(tools: Option<&[OpenAiTool]>) -> Result<Vec<ToolDefinition>, String> {
    let Some(tools) = tools else {
        return Ok(Vec::new());
    };
    if tools.len() > 64 {
        return Err("tools must contain at most 64 definitions".to_string());
    }

    let mut names = HashSet::new();
    tools
        .iter()
        .map(|tool| {
            if tool.kind != "function" {
                return Err("only tools with type `function` are supported".to_string());
            }

            let name = tool.function.name.trim();
            if name.is_empty()
                || name.len() > 64
                || !name.chars().all(|character| {
                    character.is_ascii_alphanumeric() || matches!(character, '_' | '-')
                })
            {
                return Err("tool function names must match [A-Za-z0-9_-]{1,64}".to_string());
            }
            if !names.insert(name.to_string()) {
                return Err(format!("tool function `{name}` is defined more than once"));
            }

            let parameters = tool
                .function
                .parameters
                .clone()
                .unwrap_or_else(|| json!({ "type": "object", "properties": {} }));
            if parameters.get("type").and_then(Value::as_str) != Some("object") {
                return Err(format!(
                    "tool function `{name}` parameters must be a JSON schema with type `object`"
                ));
            }

            Ok(ToolDefinition {
                name: name.to_string(),
                description: tool.function.description.clone().unwrap_or_default(),
                parameters,
            })
        })
        .collect()
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct AssistantToolCall {
    id: String,
    #[serde(rename = "type")]
    kind: &'static str,
    function: AssistantToolFunction,
}

#[derive(Clone, Debug, Serialize)]
struct AssistantToolFunction {
    name: String,
    arguments: String,
}

#[derive(Serialize)]
pub(crate) struct AssistantToolCallDelta {
    index: usize,
    id: String,
    #[serde(rename = "type")]
    kind: &'static str,
    function: AssistantToolFunction,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ToolCallEnvelope {
    tool_calls: Vec<RequestedToolCall>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RequestedToolCall {
    name: String,
    arguments: Value,
}

pub(crate) fn parse_tool_calls(
    content: &str,
    tools: &[ToolDefinition],
) -> Option<Vec<AssistantToolCall>> {
    if tools.is_empty() {
        return None;
    }

    let envelope: ToolCallEnvelope = serde_json::from_str(content).ok()?;
    if envelope.tool_calls.is_empty() || envelope.tool_calls.len() > 32 {
        return None;
    }

    envelope
        .tool_calls
        .into_iter()
        .map(|call| {
            if !call.arguments.is_object() || !tools.iter().any(|tool| tool.name == call.name) {
                return None;
            }

            Some(AssistantToolCall {
                id: format!("call_{}", Uuid::new_v4().simple()),
                kind: "function",
                function: AssistantToolFunction {
                    name: call.name,
                    arguments: serde_json::to_string(&call.arguments).ok()?,
                },
            })
        })
        .collect()
}

pub(crate) fn tool_call_deltas(tool_calls: &[AssistantToolCall]) -> Vec<AssistantToolCallDelta> {
    tool_calls
        .iter()
        .enumerate()
        .map(|(index, call)| AssistantToolCallDelta {
            index,
            id: call.id.clone(),
            kind: call.kind,
            function: call.function.clone(),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::openai::AssistantMessage;

    fn declared_tools() -> Vec<ToolDefinition> {
        let tools: Vec<OpenAiTool> = serde_json::from_value(json!([
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the weather for a city.",
                    "parameters": {
                        "type": "object",
                        "properties": { "city": { "type": "string" } },
                        "required": ["city"]
                    }
                }
            }
        ]))
        .expect("test tools should deserialize");

        normalize_tools(Some(&tools)).expect("test tools should normalize")
    }

    #[test]
    fn parses_only_declared_client_side_tool_calls() {
        let tools = declared_tools();
        let calls = parse_tool_calls(
            r#"{"tool_calls":[{"name":"get_weather","arguments":{"city":"Seattle"}}]}"#,
            &tools,
        )
        .expect("declared tool call should parse");

        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].kind, "function");
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[0].function.arguments, r#"{"city":"Seattle"}"#);
        let deltas = tool_call_deltas(&calls);
        assert_eq!(deltas[0].index, 0);
        assert_eq!(deltas[0].function.name, "get_weather");
        let message = serde_json::to_value(AssistantMessage {
            role: "assistant",
            content: None,
            tool_calls: Some(calls),
        })
        .expect("tool-call response should serialize");
        assert!(message["content"].is_null());
        assert_eq!(message["tool_calls"][0]["type"], "function");
        assert_eq!(message["tool_calls"][0]["function"]["name"], "get_weather");
        assert!(
            parse_tool_calls(
                r#"{"tool_calls":[{"name":"not_declared","arguments":{}}]}"#,
                &tools,
            )
            .is_none()
        );
    }

    #[test]
    fn rejects_invalid_tool_parameter_schema() {
        let tools: Vec<OpenAiTool> = serde_json::from_value(json!([
            {
                "type": "function",
                "function": {
                    "name": "invalid_schema",
                    "parameters": { "type": "string" }
                }
            }
        ]))
        .expect("test tools should deserialize");

        assert!(normalize_tools(Some(&tools)).is_err());
    }
}

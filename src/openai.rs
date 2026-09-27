use std::collections::HashSet;

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use uuid::Uuid;

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

pub(crate) struct PromptParts {
    pub(crate) system_message: Option<String>,
    pub(crate) prompt: String,
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

pub(crate) fn compile_prompt(
    messages: &[ChatMessage],
    tools: &[ToolDefinition],
) -> Result<PromptParts, String> {
    let mut system_messages = Vec::new();
    let mut conversation = String::from(
        "Continue the following chat conversation. Answer the latest user request directly.\n\n",
    );

    for message in messages {
        let role = message.role.trim().to_ascii_lowercase();
        let content = message_content(message.content.as_ref())?;

        match role.as_str() {
            "system" | "developer" => system_messages.push(content),
            "user" => {
                conversation.push_str(&format!("[USER]\n{content}\n\n"));
            }
            "assistant" => {
                conversation.push_str(&format!("[ASSISTANT]\n{content}\n\n"));
                if let Some(tool_calls) = &message.tool_calls {
                    let tool_calls = serde_json::to_string(tool_calls).map_err(|error| {
                        format!("Could not encode assistant tool calls: {error}")
                    })?;
                    conversation.push_str(&format!("[ASSISTANT TOOL_CALLS]\n{tool_calls}\n\n"));
                }
            }
            "tool" => {
                let result = json!({
                    "tool_call_id": message.tool_call_id,
                    "content": content,
                });
                conversation.push_str(&format!("[TOOL RESULT]\n{result}\n\n"));
            }
            _ => return Err(format!("Unsupported message role: {}", message.role)),
        }
    }

    if !conversation.contains("[USER]") {
        return Err("messages must contain at least one user message".to_string());
    }

    if !tools.is_empty() {
        system_messages.push(tool_calling_instruction(tools)?);
    }

    conversation.push_str("[ASSISTANT]\n");
    Ok(PromptParts {
        system_message: (!system_messages.is_empty()).then(|| system_messages.join("\n\n")),
        prompt: conversation,
    })
}

fn tool_calling_instruction(tools: &[ToolDefinition]) -> Result<String, String> {
    let declarations = serde_json::to_string(tools)
        .map_err(|error| format!("Could not encode tool declarations: {error}"))?;
    Ok(format!(
        "You may request client-side functions, but this service cannot execute them. \
         When a function is needed, respond with exactly one JSON object and no Markdown or other text: \
         {{\"tool_calls\":[{{\"name\":\"function_name\",\"arguments\":{{}}}}]}}. \
         Use only these declared functions, and make each arguments value a JSON object. \
         If no function is needed, respond normally. Function declarations:\n{declarations}"
    ))
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

pub(crate) fn assistant_content(data: &Value) -> Option<String> {
    data.get("content")
        .and_then(Value::as_str)
        .or_else(|| data.get("message").and_then(Value::as_str))
        .map(ToOwned::to_owned)
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
            tool_call_id: None,
            tool_calls: None,
        }
    }

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
    fn compiles_system_and_conversation_messages() {
        let prompt = compile_prompt(
            &[
                message("system", json!("Be concise.")),
                message("user", json!("Hello")),
                message("assistant", json!("Hi")),
                message("user", json!("Explain Axum")),
            ],
            &[],
        )
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
    fn normalizes_supported_reasoning_effort() {
        assert_eq!(
            normalize_reasoning_effort(Some(" HIGH ")),
            Ok(Some("high".to_string()))
        );
        assert_eq!(normalize_reasoning_effort(None), Ok(None));
        assert!(normalize_reasoning_effort(Some("extra-high")).is_err());
    }

    #[test]
    fn rejects_unknown_message_roles() {
        let result = compile_prompt(&[message("function", json!("ignored"))], &[]);
        assert!(result.is_err());
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
    fn includes_tool_history_and_declarations_in_prompt() {
        let tools = declared_tools();
        let messages = vec![
            message("user", json!("What is the weather in Seattle?")),
            ChatMessage {
                role: "assistant".to_string(),
                content: None,
                tool_call_id: None,
                tool_calls: Some(vec![IncomingToolCall {
                    id: "call_weather".to_string(),
                    kind: "function".to_string(),
                    function: IncomingToolFunction {
                        name: "get_weather".to_string(),
                        arguments: r#"{"city":"Seattle"}"#.to_string(),
                    },
                }]),
            },
            ChatMessage {
                role: "tool".to_string(),
                content: Some(json!("Rain, 12 C")),
                tool_call_id: Some("call_weather".to_string()),
                tool_calls: None,
            },
        ];

        let prompt = compile_prompt(&messages, &tools).expect("prompt should compile");
        assert!(prompt.prompt.contains("[ASSISTANT TOOL_CALLS]"));
        assert!(prompt.prompt.contains("[TOOL RESULT]"));
        assert!(prompt.system_message.unwrap().contains("get_weather"));
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
        .expect("test tool should deserialize");

        assert!(normalize_tools(Some(&tools)).is_err());
    }
}

use serde_json::{Value, json};

use super::{ChatMessage, ToolDefinition};

pub(crate) struct PromptParts {
    pub(crate) system_message: Option<String>,
    pub(crate) prompt: String,
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

trait Pipe: Sized {
    fn pipe<T>(self, function: impl FnOnce(Self) -> T) -> T {
        function(self)
    }
}

impl<T> Pipe for T {}

#[cfg(test)]
mod tests {
    use super::super::{
        request::{IncomingToolCall, IncomingToolFunction},
        tools::{OpenAiTool, normalize_tools},
    };
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
    fn rejects_unknown_message_roles() {
        let result = compile_prompt(&[message("function", json!("ignored"))], &[]);
        assert!(result.is_err());
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
}

use github_copilot_sdk::{SessionConfig, SystemMessageConfig, session::Session};

use crate::{error::ApiError, openai::PromptParts, routes::AppState};

const SERVICE_NAME: &str = "copilot-openai-api";

pub(super) async fn create_session(
    state: &AppState,
    model: &str,
    prompt_parts: &PromptParts,
    streaming: bool,
    reasoning_effort: Option<&str>,
) -> Result<Session, ApiError> {
    state
        .client
        .create_session(session_config(
            model,
            prompt_parts.system_message.clone(),
            streaming,
            reasoning_effort,
        ))
        .await
        .map_err(ApiError::upstream)
}

pub(super) fn session_config(
    model: &str,
    system_message: Option<String>,
    streaming: bool,
    reasoning_effort: Option<&str>,
) -> SessionConfig {
    let mut config = SessionConfig::default()
        .with_model(model)
        .with_client_name(SERVICE_NAME)
        .with_streaming(streaming)
        .deny_all_permissions();

    if let Some(content) = system_message {
        config = config.with_system_message(SystemMessageConfig::new().with_content(content));
    }
    if let Some(reasoning_effort) = reasoning_effort {
        config = config.with_reasoning_effort(reasoning_effort);
    }

    config
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn session_config_forwards_reasoning_effort() {
        let config = session_config("gpt-5", None, true, Some("high"));

        assert_eq!(config.reasoning_effort.as_deref(), Some("high"));
    }
}

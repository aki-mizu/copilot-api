use axum::{
    Json,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use serde_json::{Value, json};
use tracing::error;

#[derive(Debug)]
pub(crate) struct ApiError {
    status: StatusCode,
    message: String,
    error_type: &'static str,
    code: Option<&'static str>,
}

impl ApiError {
    pub(crate) fn invalid_request(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: message.into(),
            error_type: "invalid_request_error",
            code: None,
        }
    }

    pub(crate) fn unauthorized() -> Self {
        Self {
            status: StatusCode::UNAUTHORIZED,
            message: "Invalid API key".to_string(),
            error_type: "authentication_error",
            code: Some("invalid_api_key"),
        }
    }

    pub(crate) fn unsupported(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: message.into(),
            error_type: "invalid_request_error",
            code: Some("unsupported_parameter"),
        }
    }

    pub(crate) fn upstream(error: impl std::fmt::Display) -> Self {
        error!(%error, "Copilot SDK request failed");
        Self {
            status: StatusCode::BAD_GATEWAY,
            message: "The Copilot backend could not complete the request".to_string(),
            error_type: "api_error",
            code: Some("copilot_backend_error"),
        }
    }

    pub(crate) fn timeout() -> Self {
        Self {
            status: StatusCode::GATEWAY_TIMEOUT,
            message: "The Copilot backend did not finish before the request timeout".to_string(),
            error_type: "timeout_error",
            code: Some("copilot_timeout"),
        }
    }

    fn as_json(&self) -> Value {
        json!({
            "error": {
                "message": self.message,
                "type": self.error_type,
                "code": self.code,
            }
        })
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.status, Json(self.as_json())).into_response()
    }
}

mod config;
mod copilot;
mod error;
mod openai;
mod routes;

use std::sync::Arc;

use anyhow::{Context, Result};
use github_copilot_sdk::{Client, ClientOptions};
use tokio::net::TcpListener;
use tracing::{info, warn};

use crate::{
    config::Config,
    routes::{AppState, app},
};

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    let config = Arc::new(Config::from_env()?);
    if config.api_key.is_none() && !config.listen_addr.ip().is_loopback() {
        warn!(address = %config.listen_addr, "API_KEY is unset on a non-loopback address");
    }

    let client = Arc::new(
        Client::start(ClientOptions::default())
            .await
            .context("failed to start the GitHub Copilot SDK client")?,
    );
    let state = AppState {
        client: Arc::clone(&client),
        config: Arc::clone(&config),
    };

    let app = app(state);

    let listener = TcpListener::bind(config.listen_addr)
        .await
        .with_context(|| format!("failed to bind {}", config.listen_addr))?;
    info!(address = %config.listen_addr, "OpenAI-compatible API is listening");

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .context("HTTP server failed")?;

    client
        .stop()
        .await
        .context("failed to stop the Copilot SDK client")?;
    Ok(())
}

async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C signal handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install SIGTERM signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        () = ctrl_c => {},
        () = terminate => {},
    }
}

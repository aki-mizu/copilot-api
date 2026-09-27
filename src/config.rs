use std::{env, net::SocketAddr, time::Duration};

use anyhow::{Context, Result};

#[derive(Debug)]
pub(crate) struct Config {
    pub(crate) listen_addr: SocketAddr,
    pub(crate) default_model: String,
    pub(crate) request_timeout: Duration,
    pub(crate) api_key: Option<String>,
}

impl Config {
    pub(crate) fn from_env() -> Result<Self> {
        let host = env::var("HOST").unwrap_or_else(|_| "127.0.0.1".to_string());
        let port = env::var("PORT")
            .unwrap_or_else(|_| "3000".to_string())
            .parse::<u16>()
            .context("PORT must be a valid TCP port")?;
        let request_timeout_seconds = env::var("REQUEST_TIMEOUT_SECONDS")
            .unwrap_or_else(|_| "120".to_string())
            .parse::<u64>()
            .context("REQUEST_TIMEOUT_SECONDS must be a positive integer")?;

        if request_timeout_seconds == 0 {
            anyhow::bail!("REQUEST_TIMEOUT_SECONDS must be greater than zero");
        }

        let listen_addr = format!("{host}:{port}")
            .parse()
            .context("HOST and PORT must form a valid socket address")?;
        let default_model = non_empty_env("DEFAULT_MODEL").unwrap_or_else(|| "auto".to_string());

        Ok(Self {
            listen_addr,
            default_model,
            request_timeout: Duration::from_secs(request_timeout_seconds),
            api_key: non_empty_env("API_KEY"),
        })
    }
}

fn non_empty_env(name: &str) -> Option<String> {
    env::var(name).ok().filter(|value| !value.trim().is_empty())
}

# Copilot OpenAI API

An OpenAI-compatible Chat Completions service built with Axum and the [GitHub Copilot SDK for Rust](https://github.com/github/copilot-sdk/tree/main/rust).

## Requirements

- Rust 1.94 or newer
- A GitHub Copilot subscription and a usable Copilot authentication session

The SDK bundles a compatible Copilot runtime by default. A clean build can download roughly 150 MB of runtime artifacts.

## Run

```bash
cargo run --release
```

The service listens at `http://127.0.0.1:3000` by default. Configure it with environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `HOST` | `127.0.0.1` | Address to bind |
| `PORT` | `3000` | TCP port to bind |
| `API_KEY` | unset | Optional Bearer token required for `/v1/*` |
| `DEFAULT_MODEL` | `auto` | Model used when a request omits `model` |
| `REQUEST_TIMEOUT_SECONDS` | `120` | Maximum duration for a completion |

For a network-accessible deployment, set `API_KEY` and a non-loopback `HOST` explicitly.

## API

`GET /health` reports process health.

`GET /v1/models` advertises the configured default model.

`POST /v1/chat/completions` accepts the OpenAI Chat Completions message format, including `stream: true` for Server-Sent Events.

```bash
curl http://127.0.0.1:3000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "auto",
    "messages": [{"role": "user", "content": "Explain what Axum does in one sentence."}]
  }'
```

When `API_KEY` is configured, add `-H "Authorization: Bearer $API_KEY"` to `/v1/*` calls.

The initial surface intentionally supports text messages, one completion (`n: 1`), and streaming. Tool calling is rejected rather than silently exposing Copilot tools. Every session uses the SDK's deny-all permission policy, so this service does not approve filesystem, shell, browser, or MCP actions on behalf of an HTTP caller.
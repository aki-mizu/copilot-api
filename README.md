# Copilot OpenAI API

An OpenAI-compatible Chat Completions service built with Axum and the [GitHub Copilot SDK for Rust](https://github.com/github/copilot-sdk/tree/main/rust).

## Requirements

- A GitHub Copilot subscription and a usable Copilot authentication session or a fine-grained GitHub token with the **Copilot Requests** permission

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
| `COPILOT_GITHUB_TOKEN` | unset | Fine-grained GitHub token used for non-interactive Copilot authentication |
| `DEFAULT_MODEL` | `auto` | Model used when a request omits `model` |
| `REQUEST_TIMEOUT_SECONDS` | `120` | Maximum duration for a completion |

For a network-accessible deployment, set `API_KEY` and a non-loopback `HOST` explicitly.

## Deploy With Portainer

 [`docker-compose.yml`](docker-compose.yml) pulls the prebuilt GHCR image and is intended for a Docker Standalone environment. Deploy it through **Stacks** > **Add stack** > **Git Repository**, with `docker-compose.yml` as the Compose path.

Set these stack environment variables in Portainer instead of committing a `.env` file:

| Variable | Required | Purpose |
| --- | --- | --- |
| `COPILOT_GITHUB_TOKEN` | Yes | Fine-grained token with **Copilot Requests** permission |
| `API_KEY` | Yes | Bearer token required from clients calling `/v1/*` |
| `PUBLISHED_PORT` | No | Host port to publish, default `3000` |
| `DEFAULT_MODEL` | No | Model name, default `auto` |
| `REQUEST_TIMEOUT_SECONDS` | No | Completion timeout in seconds, default `120` |

The container runs as an unprivileged user and keeps the downloaded Copilot runtime in the `copilot-state` named volume. The first deployment can take longer while that runtime is downloaded. Do not add your private `.env` file or either token to the Git repository.

## Publish To GHCR

Pushing to `main`, pushing a `v*` semantic-version tag, or manually running [`.github/workflows/publish-image.yml`](.github/workflows/publish-image.yml) publishes an image to `ghcr.io/aki-mizu/copilot-api`. The workflow uses the repository `GITHUB_TOKEN`; no registry credential is stored in this repository.

For Portainer to pull the published image, deploy [`docker-compose.yml`](docker-compose.yml) and set the same stack variables listed above. `GHCR_IMAGE` defaults to `ghcr.io/aki-mizu/copilot-api:latest`; set it to a versioned tag such as `ghcr.io/aki-mizu/copilot-api:1.2.3` to pin a deployment.

The image is linked to this repository, so it inherits repository access permissions. Package visibility is separate: a first GHCR publish is private even when the repository is public. To allow Portainer to pull it without registry credentials, open the package's **Package settings**, choose **Change visibility**, and select **Public**. GitHub does not allow a public package to become private again.

If you keep the GHCR package private, add `ghcr.io` as a registry in Portainer with credentials that have package read access, then select that registry for the stack. The Copilot token and API key remain stack environment variables, not registry credentials.

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
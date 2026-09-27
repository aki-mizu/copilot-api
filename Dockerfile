# syntax=docker/dockerfile:1
FROM rust:1.94-bookworm AS builder

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential perl pkg-config \
    && rm -rf /var/lib/apt/lists/*

COPY Cargo.toml Cargo.lock ./
COPY src ./src

RUN cargo build --locked --release

FROM debian:bookworm-slim AS runtime

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --uid 10001 --create-home --home-dir /var/lib/copilot copilot

COPY --from=builder /app/target/release/copilot-openai-api /usr/local/bin/copilot-openai-api

USER copilot
WORKDIR /var/lib/copilot

ENV HOME=/var/lib/copilot \
    COPILOT_HOME=/var/lib/copilot \
    HOST=0.0.0.0 \
    PORT=3000 \
    RUST_LOG=info

EXPOSE 3000

HEALTHCHECK --interval=30s --timeout=5s --start-period=2m --retries=3 \
    CMD curl --fail --silent http://127.0.0.1:3000/health || exit 1

ENTRYPOINT ["/usr/local/bin/copilot-openai-api"]
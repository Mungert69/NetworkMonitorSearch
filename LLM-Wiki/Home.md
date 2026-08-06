# NetworkMonitorSearch

`NetworkMonitorSearch` is the NetworkMonitor background service that turns text into embeddings, stores and searches them in OpenSearch, and exposes those workflows over RabbitMQ. It has no HTTP endpoints of its own; callers communicate through message exchanges.

## Start here

- [Architecture](Architecture) — components and startup flow.
- [RabbitMQ contract](RabbitMQ-Contract) — accepted operations and response routing.
- [OpenSearch and indexing](OpenSearch-and-Indexing) — supported index types, update modes, and search.
- [Embeddings](Embeddings) — local ONNX, Novita, token sizing, and caching.
- [LLM history memory](LLM-History-Memory) — persistent conversation history and semantic recall.
- [Operations](Operations) — configuration, build, deployment, and troubleshooting.
- [Development](Development) — code map, tests, and safe change points.

## What is in this repository

The executable is a .NET 10 worker (`Program.cs`, `Startup.cs`). Service code is under `Services/`; models, configuration types, RabbitMQ primitives, and request contracts come from the adjacent `NetworkMonitorLib` and `NetworkMonitorData` projects. Two embedding-model bundles are checked in: `stsb-bert-tiny-onnx` and `qwen3-embed-0.6-onnx`.

## Service boundary

```text
NetworkMonitor callers / LLM service
              | RabbitMQ requests
              v
       NetworkMonitorSearch
        | embeddings          | OpenSearch REST/client
        v                     v
 local ONNX or Novita     document and history indexes
```

Every request is authenticated using the shared LLM encryption key and carries an application ID. Responses return on an explicit response exchange or an operation-specific default; see [RabbitMQ contract](RabbitMQ-Contract).

## Scope and source of truth

This wiki documents the tracked application source as inspected on 2026-08-06. Configuration keys and message DTO field definitions are owned by the referenced shared projects, so their code remains the authoritative contract. Large JSON files in the repository are sample/import data, not this wiki's source of operational truth.

See the [index](index) for the full catalog and [log](log) for maintenance history.

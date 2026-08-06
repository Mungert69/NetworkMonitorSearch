# NetworkMonitorSearch

NetworkMonitorSearch is the NetworkMonitor service responsible for embedding text, indexing and querying it in OpenSearch, and providing LLM conversation-memory retrieval. It is a .NET 10 background service with a RabbitMQ interface—callers do not use an HTTP API exposed by this project.

## What it does

- Generates embeddings with a local ONNX model or the Novita embedding API.
- Stores and searches documents, MITRE data, security books, quantum books, and blog content in OpenSearch.
- Supports vector search and configurable hybrid lexical/vector reranking.
- Persists LLM conversation histories and semantically searches individual turns.
- Caches generated embeddings in OpenSearch to avoid repeated model/provider work.
- Creates OpenSearch snapshots on request.

## Architecture

```text
NetworkMonitor services / LLM clients
                | RabbitMQ requests
                v
       NetworkMonitorSearch (.NET 10)
          | embeddings          | OpenSearch
          v                     v
  local ONNX or Novita      content, history, and cache indexes
```

The service initializes its RabbitMQ connection, OpenSearch integration, and message consumers during host startup. It validates requests with the shared LLM encryption key and publishes each response to either the requested exchange or an operation-specific default.

## RabbitMQ operations

| Operation | Purpose |
|---|---|
| `createIndex` | Create or update a supported index from JSON or a configured data directory. |
| `queryIndex` | Search an indexed content collection. |
| `queryMemory` | Semantically retrieve relevant LLM-history turns. |
| `queryMemoryTurnWindow` | Retrieve turns around a known session turn. |
| `historyStore` | Upsert, retrieve, list, or delete full conversation histories. |
| `createSnapshot` | Create an OpenSearch snapshot. |

The exact request and response DTOs are defined in the shared `NetworkMonitorLib` project. See the [Wiki's RabbitMQ contract](https://github.com/Mungert69/NetworkMonitorSearch/wiki/RabbitMQ-Contract) before integrating a caller.

## Getting started

### Prerequisites

- .NET 10 SDK
- Compatible sibling checkouts of `NetworkMonitorLib` and `NetworkMonitorData`
- RabbitMQ and OpenSearch reachable through the shared NetworkMonitor configuration
- An embedding model directory or remote embedding-provider configuration

### Run locally

```bash
dotnet restore
dotnet test --nologo
dotnet run --project NetworkMonitorSearch.csproj
```

Configuration is loaded from `appsettings.json` and environment variables, with shared `SystemParams` and `MLParams` resolved by `NetworkMonitorLib`. Do not commit credentials, encryption keys, or deployment-specific configuration.

## Project layout

| Location | Purpose |
|---|---|
| `Program.cs`, `Startup.cs` | Host configuration, dependency registration, and lifecycle setup. |
| `Services/RabbitListener.cs` | RabbitMQ consumers and request dispatch. |
| `Services/OpenSearchService.cs` | Request orchestration, authorization, response publishing, and query cache. |
| `Services/OpenSearchHelper.cs` | OpenSearch mappings, indexing, searches, and history persistence. |
| `Services/Strategies/IndexStrategy.cs` | Per-index serialization, embedding, mapping, and hit-formatting strategies. |
| `Services/*Embedding*.cs` | Local/remote vector generation and persistent embedding cache. |
| `Services/Tests/` | xUnit regression tests. |
| `LLM-Wiki/` | Source Markdown for the project GitHub Wiki. |

## Deployment

The project is configured to publish OCI containers for Linux x64 and arm64.

```bash
./build-run      # Release image
./build-run-dev  # Release-Dev image
```

The scripts invoke tests but currently do not fail the publish if tests fail; run `dotnet test --nologo` as a separate release check.

## Documentation

The detailed operational and architectural documentation lives in the [NetworkMonitorSearch Wiki](https://github.com/Mungert69/NetworkMonitorSearch/wiki).

- [Architecture](https://github.com/Mungert69/NetworkMonitorSearch/wiki/Architecture)
- [OpenSearch and indexing](https://github.com/Mungert69/NetworkMonitorSearch/wiki/OpenSearch-and-Indexing)
- [Embedding providers and cache](https://github.com/Mungert69/NetworkMonitorSearch/wiki/Embeddings)
- [LLM history memory](https://github.com/Mungert69/NetworkMonitorSearch/wiki/LLM-History-Memory)
- [Operations](https://github.com/Mungert69/NetworkMonitorSearch/wiki/Operations)
- [Development guide](https://github.com/Mungert69/NetworkMonitorSearch/wiki/Development)

Future LLM maintainers should follow [`LLM-Wiki/AGENTS.md`](LLM-Wiki/AGENTS.md) when updating the Wiki source.

## Related projects

- `NetworkMonitorLLM` consumes search results for RAG.
- `NetworkMonitorService` orchestrates user-facing search requests.

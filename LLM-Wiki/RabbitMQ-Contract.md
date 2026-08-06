# RabbitMQ contract

`RabbitListener` registers six RabbitMQ exchanges/operations, each with a 60-second message timeout. The shared `RabbitListenerBase` derives queue names and handles the transport mechanics using this service's `SystemParams.ThisSystemUrl`.

| Request exchange / operation | Request type | Result |
|---|---|---|
| `createIndex` | `CreateIndexRequest` | Indexes inline JSON, a JSON file, or a configured data directory. |
| `queryIndex` | `QueryIndexRequest` | Searches a supported content index. |
| `queryMemory` | `MemoryQueryRequest` | Semantically searches indexed LLM-history turns. |
| `queryMemoryTurnWindow` | `MemoryTurnWindowRequest` | Loads turns around a session turn index. |
| `historyStore` | `HistoryStoreRequest` | Upserts, gets, lists, or deletes whole conversation histories. |
| `createSnapshot` | `CreateSnapshotRequest` | Creates an OpenSearch snapshot. |

## Authentication and responses

`OpenSearchService` checks requests with `EncryptHelper.IsBadKey(LLMEncryptKey, AuthKey, AppID)`. Invalid requests fail before work is done. The exact DTO fields and encryption protocol live in `NetworkMonitorLib/Objects`.

Unless `ResponseExchange` is supplied, successful workflows publish to these defaults:

| Operation | Default response exchange |
|---|---|
| Create index | `createIndexResult{AppID}` |
| Query index | `{AppID}QueryIndexResult` |
| Query memory | `{AppID}MemoryQueryResult` |
| Memory turn window | `{AppID}MemoryTurnWindowResult` |
| History store | `{AppID}HistoryStoreResult` |

When `RoutingKey` is present, it is also supplied when publishing the response.

## Query behavior

`queryIndex` requires an index and ordinarily a `QueryText`. An empty query is permitted for an anchored or filtered follow-up request (anchor document/chunk or metadata filters). It accepts vector search mode, top-K, optional metadata, neighbors, and filters for source, section, pages, and chunk range.

`queryMemory` requires text. If a session is supplied, recall is session-scoped; otherwise it searches the user's history and excludes the current session when provided. `IncludeToolTurns` controls whether tool-related turns participate.

## Callers

The README identifies `NetworkMonitorLLM` as the RAG consumer and `NetworkMonitorService` as a user-facing search orchestrator. Treat their current request creation code as the best source for working payload examples.

Related: [LLM history memory](LLM-History-Memory) and [OpenSearch and indexing](OpenSearch-and-Indexing).

# LLM history memory

The service maintains two related OpenSearch indexes for LLM conversation persistence and retrieval.

| Index | Purpose | Keying |
|---|---|---|
| `llm_history` | Full serialized history for a service/session. | SHA-256 of normalized `serviceId:sessionId`. |
| `llm_history_turns` | Individual semantically searchable turns. | Deterministic service/session/turn document IDs. |

## Whole-history operations

`historyStore` supports `upsert`, `get`, `delete`, and `list`. Upsert stores the raw `HistoryJson` and additionally parses/indexes its turns. List filters the stored histories by service/user as implemented in `OpenSearchHelper`.

Both history indexes are created on demand. Whole-history data includes service ID, session ID, user ID, LLM type, timestamp, and the serialized payload. Treat history content as sensitive operational data.

## Turn indexing and recall

The turn index stores role, turn index/time, text, content hash, and a `turn_embedding` KNN vector. Parsing accepts string, object, and array content shapes. Tool-call and tool-response content is recognized; a response's status is inferred (for example: success, timeout, canceled, or error).

`queryMemory` runs semantic recall over `llm_history_turns`. It returns user-global results when no session is supplied, or session-only results when a session is specified. For global recall, callers can identify a current session to exclude it. Tool turns are excluded unless `IncludeToolTurns` is true. A small contextual expansion fetches adjacent turns around selected hits.

`queryMemoryTurnWindow` is deterministic rather than semantic: it retrieves a configured number of turns before and after a specific session turn index.

## Design implications

- Update the full history before expecting new turns to appear in semantic memory.
- Changing the embedding model/dimension requires reindexing `llm_history_turns` as well as content indexes.
- Deleting a whole-history record does not, by itself, imply an external retention-policy guarantee for related turn documents; validate deletion behavior against the deployment's current code and requirements.

Related: [RabbitMQ contract](RabbitMQ-Contract), [Embeddings](Embeddings), [Operations](Operations).

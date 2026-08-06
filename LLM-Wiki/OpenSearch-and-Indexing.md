# OpenSearch and indexing

## Supported content indexes

Index handling is strategy-based. An unsupported index name cannot be deserialized, mapped, or searched through this service.

| Index | Strategy | Embedding approach |
|---|---|---|
| `documents` | `DocumentIndexingStrategy` | Content plus supported vector-search modes. |
| `mitre` | `MitreIndexingStrategy` | Embeds the MITRE item output. |
| `securitybooks` | `SecurityBookIndexingStrategy` | Multi-vector book fields; optional alternate-question fields. |
| `quantumbooks` | `QuantumBookIndexingStrategy` | Multi-vector book fields. |
| `blogs` | `BlogIndexingStrategy` | Separate title and content vectors. |
| `llm_history` | Helper-managed | Stores full history payloads by service/session. |
| `llm_history_turns` | Helper-managed | Stores vectorized individual conversation turns. |
| `llm_embedding_cache` | Cache-managed | Persistent generated-vector cache. |

The configured default index is supplied by `MLParams`; it need not be one of the application-defined names unless it is used by a strategy-based request.

## Index creation modes

`CreateIndexRequest` accepts inline JSON or a JSON file. The selected strategy deserializes the source. With `CreateFromJsonDataDir`, the service instead enumerates index subdirectories under the configured `SystemParams.DataDir`.

- **Append**: retain the index and write supplied items.
- **Full rebuild** (`RecreateIndex`): delete the target index, recreate its strategy mapping, then write items.
- **Incremental update**: retain the index, compare stable content hashes, skip unchanged items and reuse persisted embeddings when possible.

Indexing derives an effective `padToTokens` from all strategy-selected text fields, clamped to configured minimum and maximum caps. It persists the result as `index_config/{index}_padtokens.json` in `DataDir`; later queries reuse it.

## Searching

Normal searches generate an embedding and use the strategy's vector field. For index names configured in `OpenSearchHybridIndices`, the helper can instead combine lexical and vector candidates with reciprocal-rank fusion (RRF), controlled by hybrid candidate, minimum-candidate, RRF-K, and field-weight settings.

Metadata-aware document queries can filter by document ID, chunk ID, source file, section path, page interval, and chunk-index interval. An anchor plus neighbor window expands returned document context. `TopK` is bounded by the implementation (normally 1–20 for document searches).

## Snapshots

`createSnapshot` calls OpenSearch's snapshot API using the supplied repository/name. Its default index list is `documents,mitre,securitybooks,blogs`; pass an explicit list if the deployment also needs `quantumbooks` or history/cache indexes. Restore support exists in `IOpenSearchService` but is not exposed by `RabbitListener`.

Related: [Embeddings](Embeddings), [LLM history memory](LLM-History-Memory), [Operations](Operations).

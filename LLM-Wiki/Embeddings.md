# Embeddings

## Provider selection

`EmbeddingGeneratorFactory` creates the base `IEmbeddingGenerator` from shared `MLParams`. The repository carries local ONNX model bundles (`stsb-bert-tiny-onnx` and `qwen3-embed-0.6-onnx`) for local inference. `NovitaEmbeddingGenerator` and `NovitaApiClient` provide the remote alternative, including rate limiting and retry behavior.

The active generator is always wrapped in `CachedEmbeddingGenerator`. Its model identity combines provider, model directory, and configured vector dimension so vectors from distinct model configurations do not share cache entries.

## Tokenization and vector sizing

`AutoTokenizer` reads tokenizer assets from the model directory. During indexing, the service counts strategy-selected text fields and calculates a padded token length between `MinTokenLengthCap` and `MaxTokenLengthCap`. That value is persisted per index and used again for query embeddings, keeping query shape consistent with indexed vectors.

The configured `EmbeddingModelVecDim` must match the active model output and every OpenSearch `knn_vector` mapping. Changing dimension requires rebuilding affected indexes; merely switching configuration will make existing vectors incompatible.

## Persistent cache

The cache key is SHA-256 over model identity, normalized text, and padding mode/length. Lookup order is:

1. in-process concurrent dictionary;
2. OpenSearch `llm_embedding_cache` index;
3. underlying model/provider.

On a miss, a generated vector is held in memory and written with OpenSearch's create-only endpoint. A 409 conflict is benign: another caller wrote the same deterministic entry. Cache failures are logged but do not prevent a usable generated embedding from being returned.

## Operational considerations

- Cache entries are never automatically expired; version the model/provider configuration when embeddings change.
- The remote provider needs its API endpoint/model/key in shared ML configuration; do not put credentials in source or wiki pages.
- Model assets significantly increase image/repository size. The project is configured to copy both local bundles to output.

Related: [OpenSearch and indexing](OpenSearch-and-Indexing) and [Operations](Operations).

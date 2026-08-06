# Operations

## Configuration

The host requires `appsettings.json` and overlays environment variables. Shared `SystemParamsHelper` supplies `SystemParams` and `MLParams`; configure values through the deployment's established shared configuration mechanism.

Required categories include:

- RabbitMQ connection and this service's `ThisSystemUrl` identity;
- OpenSearch URL, user, and key, default index, per-index query timeouts, and hybrid-search settings;
- embedding provider/model path/vector dimension, token caps, and provider credentials if remote;
- `DataDir` for import sources and persisted token metadata;
- `LLMEncryptKey` for message authentication.

Never commit a real appsettings file, API key, OpenSearch credential, or encryption key.

## Local commands

```bash
dotnet restore
dotnet test --nologo
dotnet run --project NetworkMonitorSearch.csproj
```

The solution references sibling repositories `../NetworkMonitorLib` and `../NetworkMonitorData`; they must be present and compatible. The project targets .NET 10 and is configured for Linux x64 and arm64 container publishing.

`build-run` publishes the Release OCI container (`mungert/networkmonitorsearch:<tag>` by default); `build-run-dev` publishes the Release-Dev image. Both run tests but deliberately continue if tests fail (`dotnet test ... || true`), so use the direct test command as the deployment gate.

## Health and troubleshooting

1. Confirm the process reached Rabbit connection, OpenSearch initialization, and listener setup in logs.
2. Verify the configured OpenSearch URL, credentials, cluster reachability, and expected index/vector dimension.
3. Verify the request `AuthKey` and `AppID`; bad-key errors intentionally prevent work.
4. For stale content, distinguish process query-cache results from OpenSearch data; restart to clear the in-memory query cache.
5. For vector failures, check model assets/provider access and the saved `DataDir/index_config/*_padtokens.json` value.
6. For memory recall, confirm `historyStore` upsert succeeded and `llm_history_turns` is populated.

## Security notes

The OpenSearch client and several HTTP calls accept any TLS certificate. Use a private, trusted network and prioritize certificate validation hardening before exposing this service or cluster across untrusted networks. History and embedding caches may contain sensitive text-derived data; include them explicitly in backup and retention planning.

Related: [Architecture](Architecture), [OpenSearch and indexing](OpenSearch-and-Indexing).

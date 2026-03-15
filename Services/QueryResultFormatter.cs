using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Json;
using NetworkMonitor.Objects;

namespace NetworkMonitor.Search.Services;

public static class QueryResultFormatter
{
    private static readonly HashSet<string> LocatorKeys = new(StringComparer.OrdinalIgnoreCase)
    {
        "doc_id",
        "chunk_id",
        "chunk_index",
        "chunk_count",
        "page_start",
        "page_end"
    };

    public static string Format(QueryIndexRequest request, IReadOnlyList<QueryResultObj>? queryResults, string? fallbackMessage = null)
    {
        var results = queryResults ?? Array.Empty<QueryResultObj>();
        if (results.Count == 0)
        {
            return string.IsNullOrWhiteSpace(fallbackMessage)
                ? "No results returned from index query."
                : fallbackMessage;
        }

        if (request.IncludeMetadata)
        {
            var shapedResults = results.Select(item =>
            {
                var rawMetadata = item.Metadata ?? new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
                var metadata = rawMetadata
                    .ToDictionary(kvp => kvp.Key, kvp => kvp.Value, StringComparer.OrdinalIgnoreCase);
                bool hasLocator = metadata.Keys.Any(LocatorKeys.Contains);
                var actionable = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
                if (metadata.TryGetValue("doc_id", out var docId) && !string.IsNullOrWhiteSpace(docId))
                    actionable["anchor_doc_id"] = docId;
                if (metadata.TryGetValue("chunk_id", out var chunkId) && !string.IsNullOrWhiteSpace(chunkId))
                    actionable["anchor_chunk_id"] = chunkId;
                if (metadata.TryGetValue("source_file", out var sourceFile) && !string.IsNullOrWhiteSpace(sourceFile))
                    actionable["filter_source_file"] = sourceFile;
                if (metadata.TryGetValue("section_path", out var sectionPath) && !string.IsNullOrWhiteSpace(sectionPath))
                    actionable["filter_section_path"] = sectionPath;
                if (metadata.TryGetValue("chunk_index", out var chunkIndex) && !string.IsNullOrWhiteSpace(chunkIndex))
                {
                    actionable["filter_chunk_index_min"] = chunkIndex;
                    actionable["filter_chunk_index_max"] = chunkIndex;
                }
                if (metadata.TryGetValue("page_start", out var pageStart) && !string.IsNullOrWhiteSpace(pageStart))
                    actionable["filter_page_start"] = pageStart;
                if (metadata.TryGetValue("page_end", out var pageEnd) && !string.IsNullOrWhiteSpace(pageEnd))
                    actionable["filter_page_end"] = pageEnd;
                return new
                {
                    input = item.Input ?? string.Empty,
                    output = item.Output ?? string.Empty,
                    score = item.Score,
                    source_type = hasLocator ? "rag_chunk" : "rag_text_only",
                    actionable = actionable.Count > 0 ? actionable : null,
                    metadata = metadata.Count > 0 ? metadata : null
                };
            }).ToList();

            bool anyLocator = shapedResults.Any(r => string.Equals(r.source_type, "rag_chunk", StringComparison.Ordinal));
            var payload = new
            {
                format = "query_result_v2",
                index_name = request.IndexName,
                vector_search_mode = request.VectorSearchMode.ToString(),
                top_k = request.TopK,
                result_count = shapedResults.Count,
                locator_support = anyLocator ? "partial_or_full" : "none",
                locator_note = anyLocator
                    ? "Locator metadata is present for some or all results. Prefer actionable.anchor_doc_id/anchor_chunk_id for follow-up expansion and optional filter_* parameters for targeted narrowing."
                    : "This index returned semantic text results without locator metadata. This is common for FAQ and MITRE content.",
                followup_parameters = new[]
                {
                    "anchor_doc_id",
                    "anchor_chunk_id",
                    "neighbor_window",
                    "filter_doc_id",
                    "filter_chunk_id",
                    "filter_source_file",
                    "filter_section_path",
                    "filter_page_start",
                    "filter_page_end",
                    "filter_chunk_index_min",
                    "filter_chunk_index_max"
                },
                anchor = new
                {
                    anchor_doc_id = request.AnchorDocId,
                    anchor_chunk_id = request.AnchorChunkId,
                    neighbor_window = request.NeighborWindow
                },
                results = shapedResults
            };

            return JsonSerializer.Serialize(payload, new JsonSerializerOptions
            {
                WriteIndented = true
            });
        }

        var sb = new StringBuilder();
        sb.AppendLine("Index Query Results:");
        foreach (var item in results)
        {
            sb.AppendLine($"Input: {item.Input}");
            sb.AppendLine($"Output: {item.Output}");
            sb.AppendLine("---");
        }
        return sb.ToString();
    }
}

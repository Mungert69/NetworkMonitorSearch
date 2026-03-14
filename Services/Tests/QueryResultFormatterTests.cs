using System.Collections.Generic;
using System.Text.Json;
using NetworkMonitor.Objects;
using Xunit;

namespace NetworkMonitor.Search.Services;

public class QueryResultFormatterTests
{
    [Fact]
    public void Format_WithoutMetadata_ReturnsLegacyPlainTextShape()
    {
        var request = new QueryIndexRequest
        {
            IndexName = "documents",
            IncludeMetadata = false
        };
        var results = new List<QueryResultObj>
        {
            new QueryResultObj { Input = "q1", Output = "a1", Score = 0.8f }
        };

        var formatted = QueryResultFormatter.Format(request, results);

        Assert.Contains("Index Query Results:", formatted);
        Assert.Contains("Input: q1", formatted);
        Assert.Contains("Output: a1", formatted);
    }

    [Fact]
    public void Format_WithMetadataAndLocatorFields_ReturnsLocatorAwareJsonShape()
    {
        var request = new QueryIndexRequest
        {
            IndexName = "securitybooks",
            IncludeMetadata = true,
            TopK = 3
        };
        var results = new List<QueryResultObj>
        {
            new QueryResultObj
            {
                Input = "q1",
                Output = "a1",
                Score = 0.9f,
                Metadata = new Dictionary<string, string>
                {
                    ["doc_id"] = "doc-1",
                    ["chunk_id"] = "c-12"
                }
            }
        };

        var formatted = QueryResultFormatter.Format(request, results);
        using var doc = JsonDocument.Parse(formatted);
        var root = doc.RootElement;

        Assert.Equal("query_result_v2", root.GetProperty("format").GetString());
        Assert.Equal("securitybooks", root.GetProperty("index_name").GetString());
        Assert.Equal("partial_or_full", root.GetProperty("locator_support").GetString());
        Assert.Equal("rag_chunk", root.GetProperty("results")[0].GetProperty("source_type").GetString());
    }

    [Fact]
    public void Format_WithNoResults_UsesFallbackMessageWhenProvided()
    {
        var request = new QueryIndexRequest
        {
            IndexName = "documents"
        };

        var formatted = QueryResultFormatter.Format(request, new List<QueryResultObj>(), "fallback");

        Assert.Equal("fallback", formatted);
    }
}

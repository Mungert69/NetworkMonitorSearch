using System;
using System.Linq;
using System.IO;
using Newtonsoft.Json;
using System.Security.Cryptography;
using System.Text;
using System.Threading.Tasks;
using System.Collections.Generic;
using NetworkMonitor.Objects;

namespace NetworkMonitor.Search.Services;

public interface ITokenEstimationStrategy
{
    bool CanHandle(string indexName);
    IEnumerable<string> GetFields(object item); // ← Add this

    (int padToTokens, int actualMax) EstimatePadding(IEnumerable<string> jsonFiles, string embeddingModelDir, int maxCap, int minCap);
}
public class DocumentTokenEstimationStrategy : ITokenEstimationStrategy
{
    public bool CanHandle(string indexName) => indexName.Equals("mitre", StringComparison.OrdinalIgnoreCase);
    public IEnumerable<string> GetFields(object item)
    {
        if (item is Document doc)
            return new[] { doc.Input, doc.Output };
        return Enumerable.Empty<string>();
    }
    public (int padToTokens, int actualMax) EstimatePadding(IEnumerable<string> jsonFiles, string modelDir, int maxCap, int minCap)
    {
        var tokenizer = new AutoTokenizer(modelDir, maxCap);
        int pad = minCap;
        int maxSeen = minCap;

        foreach (var file in jsonFiles)
        {
            var docs = JsonConvert.DeserializeObject<List<Document>>(File.ReadAllText(file)) ?? new();
            foreach (var d in docs)
            {
                foreach (var text in new[] { d.Input, d.Output })
                {
                    if (!string.IsNullOrWhiteSpace(text))
                    {
                        int tokens = tokenizer.CountTokens(text);
                        maxSeen = Math.Max(maxSeen, tokens);
                        pad = Math.Max(pad, tokens);
                        if (pad >= maxCap) return (pad, maxSeen);
                    }
                }
            }
        }
        return (pad, maxSeen);
    }
}
public class SecurityBookTokenEstimationStrategy : ITokenEstimationStrategy
{
    public bool CanHandle(string indexName) => indexName.Equals("securitybooks", StringComparison.OrdinalIgnoreCase);
    public IEnumerable<string> GetFields(object item)
    {
        if (item is SecurityBook book)
            return new[] { book.Input, book.Output, book.Summary };
        return Enumerable.Empty<string>();
    }
    public (int padToTokens, int actualMax) EstimatePadding(IEnumerable<string> jsonFiles, string modelDir, int maxCap, int minCap)
    {
        var tokenizer = new AutoTokenizer(modelDir, maxCap);
        int pad = minCap;
        int maxSeen = minCap;

        foreach (var file in jsonFiles)
        {
            var books = JsonConvert.DeserializeObject<List<SecurityBook>>(File.ReadAllText(file)) ?? new();
            foreach (var sb in books)
            {
                foreach (var text in new[] { sb.Input, sb.Output, sb.Summary })
                {
                    if (!string.IsNullOrWhiteSpace(text))
                    {
                        int tokens = tokenizer.CountTokens(text);
                        maxSeen = Math.Max(maxSeen, tokens);
                        pad = Math.Max(pad, tokens);
                        if (pad >= maxCap) return (pad, maxSeen);
                    }
                }
            }
        }
        return (pad, maxSeen);
    }
}

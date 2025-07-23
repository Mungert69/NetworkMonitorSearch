//  NetworkMonitor.Search.Strategies ------------------------------------------------
using System;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Threading.Tasks;
using System.Collections.Generic;
using NetworkMonitor.Objects;
using Newtonsoft.Json;

namespace NetworkMonitor.Search.Services;

/// <summary>
/// Contract every indexable artefact must satisfy.
/// All methods are **type‑agnostic** for OpenSearchHelper,
/// but individual strategy classes know their concrete type.
/// </summary>


public interface IIndexingStrategy
{
    /// Logical name of the OpenSearch index this artefact lives in
    string IndexName { get; }

    string GetVectorField(VectorSearchMode mode);
    IReadOnlyDictionary<string, float> GetDefaultFieldWeights();
    string GetIndexMapping(int vectorDimension);
    List<object> Deserialize(string json);


    /// Ensure that the artefact has every embedding it needs; generate
    /// anything that is missing.
    Task EnsureEmbeddingsAsync(object item,
                               IEmbeddingGenerator generator,
                               int padToTokens);

    /// A deterministic, collision‑resistant ID (e.g. SHA‑256 of some field)
    string ComputeId(object item);

    /// Shape the payload to be written into OpenSearch.
    object BuildIndexDocument(object item);

    /// Returns true if this strategy can handle <paramref name="item"/>.
    bool CanHandle(object item);
    bool CanHandle(string indexName);
}

//  ------------------------------------------------------------------------------
//  Helper for computing deterministic IDs from strings
internal static class IdHelper
{
    internal static string Sha256(string text)
    {
        using var sha = SHA256.Create();
        var bytes = sha.ComputeHash(Encoding.UTF8.GetBytes(text));
        var sb = new StringBuilder(bytes.Length * 2);
        foreach (var b in bytes) sb.Append(b.ToString("x2"));
        return sb.ToString();
    }
}


//  ------------------------------------------------------------------------------
//  Strategy for plain 'Documents'
public sealed class DocumentIndexingStrategy : IIndexingStrategy
{
    public string IndexName => "documents";

    public string ContentVectorFieldName => "output_embedding";
    public string QuestionVectorFieldName => "input_embedding";

    public string GetVectorField(VectorSearchMode mode) => mode switch
    {
        VectorSearchMode.question => QuestionVectorFieldName,
        _ => ContentVectorFieldName
    };
    public IReadOnlyDictionary<string, float> GetDefaultFieldWeights() =>
          new Dictionary<string, float>
          {
              [QuestionVectorFieldName] = 1f,
              [ContentVectorFieldName] = 1f
          };
    public bool CanHandle(object item) => item is Document;
    public bool CanHandle(string indexName) => indexName.Equals("documents", StringComparison.OrdinalIgnoreCase);


    public async Task EnsureEmbeddingsAsync(object item,
                                            IEmbeddingGenerator generator,
                                            int padToTokens)
    {
        var sb = (Document)item;

        async Task Ensure(Func<List<float>> get, Action<List<float>> set, string sourceText)
        {
            if (get() is { Count: > 0 }) return;

            var emb = await generator.GenerateEmbeddingAsync(sourceText, padToTokens);
            if (emb.Count == 0)
                throw new InvalidOperationException($"Failed to generate embedding for '{sourceText}'.");
            set(emb);
        }

        await Ensure(() => sb.InputEmbedding, e => sb.InputEmbedding = e, sb.Input);
        await Ensure(() => sb.OutputEmbedding, e => sb.OutputEmbedding = e, sb.Output);

    }


    public string ComputeId(object item) =>
        IdHelper.Sha256(((Document)item).Output);

    public object BuildIndexDocument(object item)
    {
        var sb = (Document)item;
        return new
        {
            input = sb.Input,
            output = sb.Output,
            input_embedding = sb.InputEmbedding,
            output_embedding = sb.OutputEmbedding
        };
    }
    // documents
    public string GetIndexMapping(int dim) => $@"
{{
  ""settings"": {{ ""index"": {{ ""knn"": true }} }},
  ""mappings"": {{
    ""properties"": {{
      ""input""  : {{ ""type"": ""text"" }},
      ""output"" : {{ ""type"": ""text"" }}

      ""input_embedding"" :  {{ ""type"": ""knn_vector"", ""dimension"": {dim},
                               ""method"": {{ ""name"": ""hnsw"", ""space_type"": ""l2"", ""engine"": ""faiss"" }} }},

      ""output_embedding"" : {{ ""type"": ""knn_vector"", ""dimension"": {dim},
                               ""method"": {{ ""name"": ""hnsw"", ""space_type"": ""l2"", ""engine"": ""faiss"" }} }}
    }}
  }}
}}";
    public List<object> Deserialize(string json)
    {
        var list = JsonConvert.DeserializeObject<List<Document>>(json);
        return list?.Cast<object>().ToList() ?? new List<object>();
    }
}

public sealed class MitreIndexingStrategy : IIndexingStrategy
{
    public string IndexName => "mitre";
    public string ContentVectorFieldName => "embedding";
    public string GetVectorField(VectorSearchMode mode) =>
            ContentVectorFieldName;
    public IReadOnlyDictionary<string, float> GetDefaultFieldWeights() =>
            new Dictionary<string, float> { [ContentVectorFieldName] = 1f };
    public bool CanHandle(object item) => item is Mitre;
    public bool CanHandle(string indexName) => indexName.Equals("mitre", StringComparison.OrdinalIgnoreCase);


    public async Task EnsureEmbeddingsAsync(object item,
                                            IEmbeddingGenerator generator,
                                            int padToTokens)
    {
        var doc = (Mitre)item;
        if (doc.Embedding is { Count: > 0 }) return;

        doc.Embedding = await generator.GenerateEmbeddingAsync(doc.Output, padToTokens);
        if (doc.Embedding.Count == 0)
            throw new InvalidOperationException("Failed to generate embedding for Document.");
    }

    public string ComputeId(object item) =>
        IdHelper.Sha256(((Mitre)item).Output);

    public object BuildIndexDocument(object item)
    {
        var d = (Mitre)item;
        return new
        {
            input = d.Input,
            output = d.Output,
            embedding = d.Embedding
        };
    }
    // mitre
    public string GetIndexMapping(int dim) => $@"
{{
  ""settings"": {{ ""index"": {{ ""knn"": true }} }},
  ""mappings"": {{
    ""properties"": {{
      ""input""     : {{ ""type"": ""text"" }},
      ""output""    : {{ ""type"": ""text"" }},
      ""embedding"" : {{
        ""type""  : ""knn_vector"",
        ""dimension"" : {dim},
        ""method"": {{ ""name"": ""hnsw"", ""space_type"": ""l2"", ""engine"": ""faiss"" }}
      }}
    }}
  }}
}}";
    public List<object> Deserialize(string json)
    {
        var list = JsonConvert.DeserializeObject<List<Mitre>>(json);
        return list?.Cast<object>().ToList() ?? new List<object>();
    }
}

//  ------------------------------------------------------------------------------
//  Strategy for 'SecurityBook'
public sealed class SecurityBookIndexingStrategy : IIndexingStrategy
{
    public string IndexName => "securitybooks";

    public string ContentVectorFieldName => "output_embedding";
    public string QuestionVectorFieldName => "input_embedding";
    public string SummaryVectorFieldName => "summary_embedding";
    public string GetVectorField(VectorSearchMode mode) => mode switch
    {
        VectorSearchMode.question => QuestionVectorFieldName,
        VectorSearchMode.summary => SummaryVectorFieldName,
        _ => ContentVectorFieldName
    };

    public IReadOnlyDictionary<string, float> GetDefaultFieldWeights() =>
       new Dictionary<string, float>
       {
           [QuestionVectorFieldName] = 1f,
           [ContentVectorFieldName] = 1f,
           [SummaryVectorFieldName] = 1f
       };
    public bool CanHandle(object item) => item is SecurityBook;
    public bool CanHandle(string indexName) => indexName.Equals("securitybooks", StringComparison.OrdinalIgnoreCase);


    public async Task EnsureEmbeddingsAsync(object item,
                                            IEmbeddingGenerator generator,
                                            int padToTokens)
    {
        var sb = (SecurityBook)item;

        async Task Ensure(Func<List<float>> get, Action<List<float>> set, string sourceText)
        {
            if (get() is { Count: > 0 }) return;

            var emb = await generator.GenerateEmbeddingAsync(sourceText, padToTokens);
            if (emb.Count == 0)
                throw new InvalidOperationException($"Failed to generate embedding for '{sourceText}'.");
            set(emb);
        }

        await Ensure(() => sb.InputEmbedding, e => sb.InputEmbedding = e, sb.Input);
        await Ensure(() => sb.OutputEmbedding, e => sb.OutputEmbedding = e, sb.Output);
        await Ensure(() => sb.SummaryEmbedding, e => sb.SummaryEmbedding = e, sb.Summary);
    }

    public string ComputeId(object item) =>
        IdHelper.Sha256(((SecurityBook)item).Output);

    public object BuildIndexDocument(object item)
    {
        var sb = (SecurityBook)item;
        return new
        {
            input = sb.Input,
            output = sb.Output,
            summary = sb.Summary,
            input_embedding = sb.InputEmbedding,
            output_embedding = sb.OutputEmbedding,
            summary_embedding = sb.SummaryEmbedding
        };
    }
    // securitybooks
    public string GetIndexMapping(int dim) => $@"
{{
  ""settings"": {{ ""index"": {{ ""knn"": true }} }},
  ""mappings"": {{
    ""properties"": {{
      ""input""  : {{ ""type"": ""text"" }},
      ""output"" : {{ ""type"": ""text"" }},
      ""summary"": {{ ""type"": ""text"" }},

      ""input_embedding"" :  {{ ""type"": ""knn_vector"", ""dimension"": {dim},
                               ""method"": {{ ""name"": ""hnsw"", ""space_type"": ""l2"", ""engine"": ""faiss"" }} }},

      ""output_embedding"" : {{ ""type"": ""knn_vector"", ""dimension"": {dim},
                               ""method"": {{ ""name"": ""hnsw"", ""space_type"": ""l2"", ""engine"": ""faiss"" }} }},

      ""summary_embedding"" : {{ ""type"": ""knn_vector"", ""dimension"": {dim},
                                ""method"": {{ ""name"": ""hnsw"", ""space_type"": ""l2"", ""engine"": ""faiss"" }} }}
    }}
  }}
}}";
    public List<object> Deserialize(string json)
    {
        var list = JsonConvert.DeserializeObject<List<SecurityBook>>(json);
        return list?.Cast<object>().ToList() ?? new List<object>();
    }

}

public class Document
{
    public string Input { get; set; } = "";
    public string Output { get; set; } = "";
    public List<float> InputEmbedding { get; set; } = new();
    public List<float> OutputEmbedding { get; set; } = new();
}
public class Mitre
{
    public string Input { get; set; } = "";
    public string Output { get; set; } = "";
    public List<float> Embedding { get; set; } = new();
}

public class SecurityBook
{
    public string Input { get; set; } = "";
    public string Output { get; set; } = "";
    public string Summary { get; set; } = "";
    public List<float> InputEmbedding { get; set; } = new();
    public List<float> OutputEmbedding { get; set; } = new();
    public List<float> SummaryEmbedding { get; set; } = new();
}
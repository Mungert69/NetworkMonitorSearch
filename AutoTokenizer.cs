using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

public class SpecialToken
{
    [JsonProperty("content")]
    public string Content { get; set; }

    [JsonProperty("lstrip")]
    public bool Lstrip { get; set; }

    [JsonProperty("normalized")]
    public bool Normalized { get; set; }

    [JsonProperty("rstrip")]
    public bool Rstrip { get; set; }

    [JsonProperty("single_word")]
    public bool SingleWord { get; set; }
}

public class AutoTokenizer
{
    private readonly Dictionary<string, int> _vocab;
    private readonly int _maxLength;
    private readonly bool _doLowerCase;
    private readonly Dictionary<string, string> _specialTokens;

    public AutoTokenizer(string modelDir)
    {
        // Load tokenizer configuration
        var tokenizerConfigPath = Path.Combine(modelDir, "tokenizer_config.json");
        var tokenizerConfig = JsonConvert.DeserializeObject<JObject>(File.ReadAllText(tokenizerConfigPath));

        _maxLength = tokenizerConfig["model_max_length"]?.Value<int>() ?? 128;
        _doLowerCase = tokenizerConfig["do_lower_case"]?.Value<bool>() ?? false;

        // Load special tokens
        var specialTokensPath = Path.Combine(modelDir, "special_tokens_map.json");
        var specialTokenObjects = JsonConvert.DeserializeObject<Dictionary<string, SpecialToken>>(File.ReadAllText(specialTokensPath));

        // Extract the "content" field for each token
        _specialTokens = specialTokenObjects.ToDictionary(
            kvp => kvp.Key, // Key remains the same (e.g., "cls_token")
            kvp => kvp.Value.Content // Extract the "content" field
        );

        // Load vocabulary
        var vocabPath = Path.Combine(modelDir, "vocab.txt");
        _vocab = LoadVocabulary(vocabPath);
    }

    public TokenizedInput Tokenize(string text)
    {
        // Preprocess text (lowercase if needed)
        if (_doLowerCase)
            text = text.ToLowerInvariant();

        // Split into tokens (simplified WordPiece tokenization)
        var tokens = text.Split(new[] { ' ', '\t', '\n' }, StringSplitOptions.RemoveEmptyEntries)
                         .SelectMany(TokenizeWord)
                         .ToList();

        // Add special tokens ([CLS] and [SEP])
        tokens.Insert(0, _specialTokens["cls_token"]);
        tokens.Add(_specialTokens["sep_token"]);

        // Truncate to max length
        if (tokens.Count > _maxLength)
            tokens = tokens.Take(_maxLength).ToList();

        // Pad to max length
        var inputIds = tokens.Select(t => (long)_vocab.GetValueOrDefault(t, _vocab[_specialTokens["unk_token"]])).ToList();
        var attentionMask = Enumerable.Repeat(1L, inputIds.Count).ToList();

        while (inputIds.Count < _maxLength)
        {
            inputIds.Add((long)_vocab[_specialTokens["pad_token"]]);
            attentionMask.Add(0L);
        }

        return new TokenizedInput
        {
            InputIds = inputIds,
            AttentionMask = attentionMask,
            TokenTypeIds = new List<long>(new long[_maxLength])
        };
    }

    private IEnumerable<string> TokenizeWord(string word)
    {
        // Simplified WordPiece tokenization (split into subwords)
        // Replace this with actual WordPiece logic if needed
        return new[] { word };
    }

    private Dictionary<string, int> LoadVocabulary(string vocabPath)
    {
        var vocab = new Dictionary<string, int>();
        var lines = File.ReadAllLines(vocabPath);
        for (int i = 0; i < lines.Length; i++)
        {
            vocab[lines[i]] = i;
        }
        return vocab;
    }
}

public class TokenizedInput
{
    public List<long> InputIds { get; set; }
    public List<long> AttentionMask { get; set; }
    public List<long> TokenTypeIds { get; set; }
}

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
namespace NetworkMonitor.Search.Services;
public class SpecialToken
{
    [JsonProperty("content")]
    public string Content { get; set; } ="";

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
        var tokenizerConfig = JsonConvert.DeserializeObject<JObject>(File.ReadAllText(tokenizerConfigPath)) ?? new ();

        _maxLength = tokenizerConfig["model_max_length"]?.Value<int>() ?? 512;
        _doLowerCase = tokenizerConfig["do_lower_case"]?.Value<bool>() ?? true;

        // Load special tokens
        var specialTokensPath = Path.Combine(modelDir, "special_tokens_map.json");
        var specialTokensJson = File.ReadAllText(specialTokensPath);
        var specialTokensJObj = JsonConvert.DeserializeObject<JObject>(specialTokensJson) ?? new JObject();

        _specialTokens = new Dictionary<string, string>();
        foreach (var prop in specialTokensJObj.Properties())
        {
            if (prop.Value.Type == JTokenType.Object)
            {
                // Standard case: object with "content" field
                var tokenObj = prop.Value.ToObject<SpecialToken>();
                _specialTokens[prop.Name] = tokenObj?.Content ?? "";
            }
            else if (prop.Value.Type == JTokenType.Array)
            {
                // For additional_special_tokens: array of strings
                var arr = prop.Value.ToObject<List<string>>();
                if (arr != null && arr.Count > 0)
                {
                    // Store the first as the representative, or handle as needed
                    _specialTokens[prop.Name] = arr[0];
                }
            }
            else if (prop.Value.Type == JTokenType.String)
            {
                _specialTokens[prop.Name] = prop.Value.ToString();
            }
        }

        // Load vocabulary
        var vocabTxtPath = Path.Combine(modelDir, "vocab.txt");
        var vocabJsonPath = Path.Combine(modelDir, "vocab.json");
        if (File.Exists(vocabTxtPath))
        {
            _vocab = LoadVocabulary(vocabTxtPath);
        }
        else if (File.Exists(vocabJsonPath))
        {
            _vocab = JsonConvert.DeserializeObject<Dictionary<string, int>>(File.ReadAllText(vocabJsonPath));
        }
        else
        {
            throw new FileNotFoundException("No vocab.txt or vocab.json found in model directory.");
        }
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
    public List<long> InputIds { get; set; } = new ();
    public List<long> AttentionMask { get; set; } = new ();
    public List<long> TokenTypeIds { get; set; } = new ();
}

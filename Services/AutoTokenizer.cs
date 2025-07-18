using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Newtonsoft.Json.Linq;
using Tokenizers.DotNet;

namespace NetworkMonitor.Search.Services
{
    public class AutoTokenizer
    {
        private readonly Tokenizer _tokenizer;
        private readonly int _maxLength;
        private readonly uint _padTokenId;

        public AutoTokenizer(string modelDir)
        {
            // Load tokenizer.json
            var tokJson = Path.Combine(modelDir, "tokenizer.json");
            _tokenizer = new Tokenizer(tokJson);

            // Load config for max length and pad token
            var cfg = JObject.Parse(File.ReadAllText(Path.Combine(modelDir, "tokenizer_config.json")));
            _maxLength = 512;

            // Get pad token string
            var padToken = cfg["pad_token"]?.Value<string>();
            if (string.IsNullOrEmpty(padToken))
                throw new Exception("pad_token not found in tokenizer_config.json");

            // Load vocab (json or txt) to get pad token ID
            var vocabJsonPath = Path.Combine(modelDir, "vocab.json");
            var vocabTxtPath = Path.Combine(modelDir, "vocab.txt");
            Dictionary<string, uint> vocab;
            if (File.Exists(vocabJsonPath))
            {
                // vocab.json is usually string→int, convert to string→uint
                var vocabRaw = JObject.Parse(File.ReadAllText(vocabJsonPath));
                vocab = vocabRaw.Properties().ToDictionary(
                    p => p.Name,
                    p => (uint)p.Value.Value<int>());
            }
            else if (File.Exists(vocabTxtPath))
            {
                vocab = File.ReadAllLines(vocabTxtPath)
                            .Select((line, idx) => new { line, idx })
                            .ToDictionary(x => x.line, x => (uint)x.idx);
            }
            else
            {
                throw new Exception("No vocab.json or vocab.txt found in model directory.");
            }

            if (!vocab.TryGetValue(padToken, out _padTokenId))
                throw new Exception($"Pad token '{padToken}' not found in vocabulary.");
        }

        public TokenizedInput Tokenize(string text)
        {
            var ids = _tokenizer.Encode(text); // ids is uint[]
            int len = Math.Min(ids.Length, _maxLength);

            var inputIds = new long[_maxLength];
            var attentionMask = new long[_maxLength];

            for (int i = 0; i < len; i++)
            {
                inputIds[i] = ids[i];
                attentionMask[i] = 1;
            }
            for (int i = len; i < _maxLength; i++)
            {
                inputIds[i] = _padTokenId;
                attentionMask[i] = 0;
            }

            return new TokenizedInput
            {
                InputIds = inputIds.ToList(),
                AttentionMask = attentionMask.ToList(),
                TokenTypeIds = Enumerable.Repeat(0L, _maxLength).ToList()
            };
        }


    }

    public class TokenizedInput
    {
        public List<long> InputIds { get; set; } = new();
        public List<long> AttentionMask { get; set; } = new();
        public List<long> TokenTypeIds { get; set; } = new();
    }
}

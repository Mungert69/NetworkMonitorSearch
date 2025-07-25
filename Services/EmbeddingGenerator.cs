using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Newtonsoft.Json.Linq;
using Microsoft.Extensions.Logging;
using Onnx;
using NetworkMonitor.Objects;

namespace NetworkMonitor.Search.Services
{
    public interface IEmbeddingGenerator
    {
        Task<List<float>> GenerateEmbeddingAsync(string text, int padToTokens, bool pad = false);
    }

    public class EmbeddingGenerator : IEmbeddingGenerator
    {
        private readonly InferenceSession _session;
        private readonly AutoTokenizer _tokenizer;
        private readonly string _modelPath;
        private readonly int _maxTokenLengthCap;
        private readonly ILogger<EmbeddingGenerator> _logger;

        private static readonly SemaphoreSlim _embeddingSemaphore = new SemaphoreSlim(1, 1);

        public EmbeddingGenerator(MLParams mlParams, ILogger<EmbeddingGenerator> logger )
        {
            _logger = logger;
            _modelPath = Path.Combine(mlParams.EmbeddingModelDir, "model.onnx");
            var options = new SessionOptions();
            _maxTokenLengthCap = mlParams.MaxTokenLengthCap;
            options.IntraOpNumThreads = mlParams.LlmThreads;
            _session = new InferenceSession(_modelPath, options);

            _tokenizer = new AutoTokenizer(mlParams.EmbeddingModelDir, _maxTokenLengthCap);
            foreach (var o in _session.OutputMetadata)
                _logger?.LogInformation($"{o.Key}: {string.Join(", ", o.Value.Dimensions)}");
        }

        public async Task<List<float>> GenerateEmbeddingAsync(string text, int padToTokens, bool pad = false)
        {
            await _embeddingSemaphore.WaitAsync();
            try
            {
                var tokenizedInput = pad
                    ? _tokenizer.Tokenize(text, padToTokens)
                    : _tokenizer.TokenizeNoPad(text);

                foreach (var kv in _session.InputMetadata)
                    _logger.LogInformation($"{kv.Key} → {kv.Value.ElementType}, shape: [{string.Join(", ", kv.Value.Dimensions)}]");

                // Convert to tensors
                int seqLen = tokenizedInput.InputIds.Count;

                var inputIdsTensor = new DenseTensor<long>(tokenizedInput.InputIds.ToArray(), new[] { 1, seqLen });
                var attentionMaskTensor = new DenseTensor<long>(tokenizedInput.AttentionMask.ToArray(), new[] { 1, seqLen });

                // QWEN and some newer models expect position_ids:
                var positionIdsArr = new long[seqLen];
                for (int i = 0; i < seqLen; i++)
                    positionIdsArr[i] = i;
                var positionIdsTensor = new DenseTensor<long>(positionIdsArr, new[] { 1, seqLen });


                // Prepare inputs (include position_ids if model expects it)
                // Prepare inputs
                var inputs = new List<NamedOnnxValue>
                {
                    NamedOnnxValue.CreateFromTensor("input_ids", inputIdsTensor),
                    NamedOnnxValue.CreateFromTensor("attention_mask", attentionMaskTensor),
                    NamedOnnxValue.CreateFromTensor("position_ids", positionIdsTensor)
                };


                using var results = _session.Run(inputs);

                // Debug: print all returned outputs
                foreach (var result in results)
                {
                    _logger.LogInformation($"ONNX output: {result.Name}, type: {result.Value?.GetType()}");
                }

                // Try to find a float32 or float16 tensor (embedding) in the outputs
                var embeddingResultFloat = results.FirstOrDefault(r => r.Value is Tensor<float>);
                if (embeddingResultFloat != null)
                {
                    var embeddingsTensor = embeddingResultFloat.AsTensor<float>();
                    return PoolEmbeddings(embeddingsTensor, tokenizedInput.AttentionMask);
                }
                var embeddingResultF16 = results.FirstOrDefault(r => r.Value is Tensor<Float16>);
                if (embeddingResultF16 != null)
                {
                    var embeddingsTensorF16 = embeddingResultF16.AsTensor<Float16>();
                    return PoolEmbeddingsF16(embeddingsTensorF16, tokenizedInput.AttentionMask);
                }
                var embeddingResultInt8 = results.FirstOrDefault(r => r.Value is Tensor<byte>);
                if (embeddingResultInt8 != null)
                {
                    var embeddingsTensorInt8 = embeddingResultInt8.AsTensor<byte>();
                    // Replace these with actual values for your quantized model
                    float scale = 0.0027f;        // Example value, get from model
                    float zeroPoint = 128.0f;     // Example value, get from model

                    return PoolEmbeddingsUInt8(embeddingsTensorInt8, tokenizedInput.AttentionMask, scale, zeroPoint);

                }

                _logger.LogInformation("No float32 ,float16 or int8 tensor found in ONNX outputs!");
                return new List<float>();
            }
            finally
            {
                _embeddingSemaphore.Release();
            }
        }

        private List<float> PoolEmbeddings(Tensor<float> embeddingsTensor, List<long> attentionMask)
        {
            var dims = embeddingsTensor.Dimensions;
            if (dims.Length != 3)
                throw new Exception($"Unexpected embedding tensor shape: [{string.Join(", ", dims.ToArray())}]");

            int seqLen = dims[1];
            int embeddingDim = dims[2];
            var pooledEmbedding = new float[embeddingDim];

            for (int i = 0; i < embeddingDim; i++)
            {
                float sum = 0;
                int count = 0;
                for (int j = 0; j < seqLen; j++)
                {
                    if (attentionMask[j] == 1)
                    {
                        sum += embeddingsTensor[0, j, i];
                        count++;
                    }
                }
                pooledEmbedding[i] = count > 0 ? sum / count : 0;
            }
            return pooledEmbedding.ToList();
        }

        private List<float> PoolEmbeddingsF16(Tensor<Float16> embeddingsTensor, List<long> attentionMask)
        {
            var dims = embeddingsTensor.Dimensions;
            if (dims.Length != 3)
                throw new Exception($"Unexpected embedding tensor shape: [{string.Join(", ", dims.ToArray())}]");

            int seqLen = dims[1];
            int embeddingDim = dims[2];
            var pooledEmbedding = new float[embeddingDim];

            for (int i = 0; i < embeddingDim; i++)
            {
                float sum = 0;
                int count = 0;
                for (int j = 0; j < seqLen; j++)
                {
                    if (attentionMask[j] == 1)
                    {
                        sum += (float)embeddingsTensor[0, j, i];
                        count++;
                    }
                }
                pooledEmbedding[i] = count > 0 ? sum / count : 0;
            }
            return pooledEmbedding.ToList();
        }

        private List<float> PoolEmbeddingsUInt8(
      Tensor<byte> qTensor,
      List<long> attentionMask,
      float scale,
      float zeroPoint
  )
        {
            var dimsArr = qTensor.Dimensions.ToArray();

            if (dimsArr.Length == 3)
            {
                int seq = dimsArr[1], dim = dimsArr[2];
                var pooled = new float[dim];

                for (int j = 0; j < dim; j++)
                {
                    float sum = 0;
                    int count = 0;
                    for (int i = 0; i < seq; i++)
                    {
                        if (attentionMask[i] == 1)
                        {
                            byte q = qTensor[0, i, j];
                            sum += (q - zeroPoint) * scale;
                            count++;
                        }
                    }
                    pooled[j] = count > 0 ? sum / count : 0;
                }
                return pooled.ToList();
            }
            else if (dimsArr.Length == 2)
            {
                int dim = dimsArr[1];
                var pooled = new float[dim];
                for (int j = 0; j < dim; j++)
                {
                    if (attentionMask[j] == 1)
                    {
                        byte q = qTensor[0, j];
                        pooled[j] = (q - zeroPoint) * scale;
                    }
                    else
                    {
                        pooled[j] = 0;
                    }
                }
                return pooled.ToList();
            }
            else
            {
                throw new Exception($"Unexpected tensor shape: [{string.Join(',', dimsArr)}]");
            }
        }

        public async Task<List<List<float>>> GenerateBatchEmbeddingsAsync(List<string> texts, int maxTokens)
        {
            await _embeddingSemaphore.WaitAsync();
            try
            {
                var tokenized = texts.Select(t => _tokenizer.Tokenize(t, maxTokens)).ToList();
                int B = texts.Count, S = maxTokens;

                // Build data arrays
                var inputIds = new long[B, S];
                var attentionMask = new long[B, S];
                var positionIds = new long[B, S];

                for (int b = 0; b < B; b++)
                {
                    var tkn = tokenized[b];
                    for (int j = 0; j < S; j++)
                    {
                        inputIds[b, j] = tkn.InputIds[j];
                        attentionMask[b, j] = tkn.AttentionMask[j];
                        positionIds[b, j] = j;
                    }
                }

                var inputs = new List<NamedOnnxValue>
                {
                    NamedOnnxValue.CreateFromTensor("input_ids", inputIds.ToTensor()),
                    NamedOnnxValue.CreateFromTensor("attention_mask", attentionMask.ToTensor()),
                    NamedOnnxValue.CreateFromTensor("position_ids", positionIds.ToTensor())
                };
                using var results = _session.Run(inputs);

                // Find the output (float32 or float16)
                var embeddingResultFloat = results.FirstOrDefault(r => r.Value is Tensor<float>);
                var embeddingResultF16 = results.FirstOrDefault(r => r.Value is Tensor<Float16>);

                if (embeddingResultFloat != null)
                {
                    var embeddingsTensor = embeddingResultFloat.AsTensor<float>();
                    return PoolBatchEmbeddings(embeddingsTensor, tokenized.Select(x => x.AttentionMask).ToList());
                }
                if (embeddingResultF16 != null)
                {
                    var embeddingsTensorF16 = embeddingResultF16.AsTensor<Float16>();
                    return PoolBatchEmbeddingsF16(embeddingsTensorF16, tokenized.Select(x => x.AttentionMask).ToList());
                }
                throw new Exception("No float32 or float16 tensor found in ONNX outputs!");
            }
            finally
            {
                _embeddingSemaphore.Release();
            }
        }

        // Helper: Pool embeddings for batch outputs (float32)
        private List<List<float>> PoolBatchEmbeddings(Tensor<float> embeddingsTensor, List<List<long>> attentionMasks)
        {
            var dims = embeddingsTensor.Dimensions;
            if (dims.Length != 3)
                throw new Exception($"Unexpected tensor shape: [{string.Join(", ", dims.ToArray())}]");

            int batchSize = dims[0];
            int seqLen = dims[1];
            int embeddingDim = dims[2];
            var results = new List<List<float>>(batchSize);

            for (int b = 0; b < batchSize; b++)
            {
                var pooled = new float[embeddingDim];
                for (int i = 0; i < embeddingDim; i++)
                {
                    float sum = 0;
                    int count = 0;
                    for (int j = 0; j < seqLen; j++)
                    {
                        if (attentionMasks[b][j] == 1)
                        {
                            sum += embeddingsTensor[b, j, i];
                            count++;
                        }
                    }
                    pooled[i] = count > 0 ? sum / count : 0;
                }
                results.Add(pooled.ToList());
            }
            return results;
        }

        // Helper: Pool embeddings for batch outputs (float16)
        private List<List<float>> PoolBatchEmbeddingsF16(Tensor<Float16> embeddingsTensor, List<List<long>> attentionMasks)
        {
            var dims = embeddingsTensor.Dimensions;
            if (dims.Length != 3)
                throw new Exception($"Unexpected tensor shape: [{string.Join(", ", dims.ToArray())}]");


            int batchSize = dims[0];
            int seqLen = dims[1];
            int embeddingDim = dims[2];
            var results = new List<List<float>>(batchSize);

            for (int b = 0; b < batchSize; b++)
            {
                var pooled = new float[embeddingDim];
                for (int i = 0; i < embeddingDim; i++)
                {
                    float sum = 0;
                    int count = 0;
                    for (int j = 0; j < seqLen; j++)
                    {
                        if (attentionMasks[b][j] == 1)
                        {
                            sum += (float)embeddingsTensor[b, j, i];
                            count++;
                        }
                    }
                    pooled[i] = count > 0 ? sum / count : 0;
                }
                results.Add(pooled.ToList());
            }
            return results;
        }
        public void PrintEmbedding(string label, List<float> emb)
        {
            _logger.LogInformation($"{label} first 8: {string.Join(", ", emb.Take(8))}");
            _logger.LogInformation($"{label} norm: {Math.Sqrt(emb.Sum(x => x * x))}");
        }

        public float CosineSim(List<float> a, List<float> b)
        {
            float dot = 0, normA = 0, normB = 0;
            for (int i = 0; i < a.Count; i++)
            {
                dot += a[i] * b[i];
                normA += a[i] * a[i];
                normB += b[i] * b[i];
            }
            return dot / (float)(Math.Sqrt(normA) * Math.Sqrt(normB));
        }


    }
}

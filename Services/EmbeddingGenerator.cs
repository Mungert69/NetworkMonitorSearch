using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Newtonsoft.Json.Linq;
using Onnx;

namespace NetworkMonitor.Search.Services
{
    public class EmbeddingGenerator
    {
        private readonly InferenceSession _session;
        private readonly AutoTokenizer _tokenizer;
        private readonly string _modelPath;
        private int _defaultMinTokens;

        private static readonly object _embeddingLock = new object();

        public EmbeddingGenerator(string modelDir, int defaultMinTokens)
        {
            // Load the ONNX model with restricted CPU threads
            _modelPath = Path.Combine(modelDir, "model.onnx");
            var options = new SessionOptions();
            options.IntraOpNumThreads = 2;
            _defaultMinTokens = defaultMinTokens;
            _session = new InferenceSession(_modelPath, options);

            // Initialize the tokenizer with default min length
            _tokenizer = new AutoTokenizer(modelDir);
        }

        public List<float> GenerateEmbedding(string text, int? overrideMaxTokens = null)
        {
            lock (_embeddingLock)
            {
                int maxTokenLength = _defaultMinTokens;
                if (overrideMaxTokens != null) maxTokenLength = (int)overrideMaxTokens;
              
                // Tokenize the input text
                var tokenizedInput = _tokenizer.Tokenize(text, maxTokenLength);

                foreach (var kv in _session.InputMetadata)
                    Console.WriteLine($"{kv.Key} → {kv.Value.ElementType}, shape: [{string.Join(", ", kv.Value.Dimensions)}]");

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
                    Console.WriteLine($"ONNX output: {result.Name}, type: {result.Value?.GetType()}");
                }

                // Try to find a float32 or float16 tensor (embedding) in the outputs
                var embeddingResultFloat = results.FirstOrDefault(r => r.Value is Tensor<float>);
                if (embeddingResultFloat != null)
                {
                    var embeddingsTensor = embeddingResultFloat.AsTensor<float>();
                    return PoolEmbeddings(embeddingsTensor);
                }

                var embeddingResultF16 = results.FirstOrDefault(r => r.Value is Tensor<Float16>);
                if (embeddingResultF16 != null)
                {
                    var embeddingsTensorF16 = embeddingResultF16.AsTensor<Float16>();
                    return PoolEmbeddingsF16(embeddingsTensorF16);
                }
                var embeddingResultInt8 = results.FirstOrDefault(r => r.Value is Tensor<byte>);
                if (embeddingResultInt8 != null)
                {
                    var embeddingsTensorInt8 = embeddingResultInt8.AsTensor<byte>();

                    return PoolEmbeddingsUInt8(embeddingsTensorInt8);
                }

                throw new Exception("No float32 ,float16 or int8 tensor found in ONNX outputs!");
            }
        }

        private List<float> PoolEmbeddings(Tensor<float> embeddingsTensor)
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
                for (int j = 0; j < seqLen; j++)
                    sum += embeddingsTensor[0, j, i];
                pooledEmbedding[i] = sum / seqLen;
            }
            return pooledEmbedding.ToList();
        }

        private List<float> PoolEmbeddingsF16(Tensor<Float16> embeddingsTensor)
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
                for (int j = 0; j < seqLen; j++)
                    sum += (float)embeddingsTensor[0, j, i]; // Convert Float16 → float
                pooledEmbedding[i] = sum / seqLen;
            }
            return pooledEmbedding.ToList();
        }
        private List<float> PoolEmbeddingsUInt8(Tensor<byte> qTensor)
        {
            const float min = -0.300980538529f;
            const float max = 0.395263433565f;
            const float scale = (max - min) / 255f;
            var dimsArr = qTensor.Dimensions.ToArray();

            if (dimsArr.Length == 3)
            {
                int seq = dimsArr[1], dim = dimsArr[2];
                var pooled = new float[dim];

                for (int j = 0; j < dim; j++)
                {
                    float sum = 0;
                    for (int i = 0; i < seq; i++)
                    {
                        byte q = qTensor[0, i, j];
                        sum += q * scale + min;
                    }
                    pooled[j] = sum / seq;
                }
                return pooled.ToList();
            }
            else if (dimsArr.Length == 2)
            {
                int dim = dimsArr[1];
                var pooled = new float[dim];
                for (int j = 0; j < dim; j++)
                {
                    byte q = qTensor[0, j];
                    pooled[j] = q * scale + min;
                }
                return pooled.ToList();
            }
            else
            {
                throw new Exception($"Unexpected tensor shape: [{string.Join(',', dimsArr)}]");
            }
        }

        public void PrintEmbedding(string label, List<float> emb)
        {
            Console.WriteLine($"{label} first 8: {string.Join(", ", emb.Take(8))}");
            Console.WriteLine($"{label} norm: {Math.Sqrt(emb.Sum(x => x * x))}");
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

        public void CompareEmbeddings(EmbeddingGenerator floatGen, EmbeddingGenerator quantGen, string text)
        {
            var floatEmb = floatGen.GenerateEmbedding(text);
            var quantEmb = quantGen.GenerateEmbedding(text);
            PrintEmbedding("FLOAT32", floatEmb);
            PrintEmbedding("UINT8 ", quantEmb);
            Console.WriteLine($"Cosine similarity: {CosineSim(floatEmb, quantEmb)}");
        }


    }
}

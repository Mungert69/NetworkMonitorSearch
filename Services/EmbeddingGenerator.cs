using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Newtonsoft.Json.Linq;

namespace NetworkMonitor.Search.Services
{
    public class EmbeddingGenerator
    {
        private readonly InferenceSession _session;
        private readonly AutoTokenizer _tokenizer;

        public EmbeddingGenerator(string modelDir)
        {
            // Load the ONNX model
            var modelPath = Path.Combine(modelDir, "model.onnx");
            _session = new InferenceSession(modelPath);

            // Initialize the tokenizer
            _tokenizer = new AutoTokenizer(modelDir);
        }

        public List<float> GenerateEmbedding(string text)
        {
            // Tokenize the input text
            var tokenizedInput = _tokenizer.Tokenize(text);

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
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("input_ids", inputIdsTensor),
                NamedOnnxValue.CreateFromTensor("attention_mask", attentionMaskTensor),
                //NamedOnnxValue.CreateFromTensor("position_ids", positionIdsTensor)
                NamedOnnxValue.CreateFromTensor("token_type_ids", positionIdsTensor)
                
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

            throw new Exception("No float32 or float16 tensor found in ONNX outputs!");
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
    }
}

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
public class EmbeddingGenerator
{
    private readonly InferenceSession _session;
    private readonly AutoTokenizer _tokenizer;

    public EmbeddingGenerator(string modelDir)
    {
        // Load the ONNX model
        var modelPath = Path.Combine(modelDir,"onnx", "model.onnx");
        _session = new InferenceSession(modelPath);

        // Initialize the tokenizer
        _tokenizer = new AutoTokenizer(modelDir);
    }

    public List<float> GenerateEmbedding(string text)
    {
        // Tokenize the input text
        var tokenizedInput = _tokenizer.Tokenize(text);

        // Convert to tensors
        var inputIdsTensor = new DenseTensor<long>(tokenizedInput.InputIds.ToArray(), new[] { 1, tokenizedInput.InputIds.Count });
        var attentionMaskTensor = new DenseTensor<long>(tokenizedInput.AttentionMask.ToArray(), new[] { 1, tokenizedInput.AttentionMask.Count });
        var tokenTypeIdsTensor = new DenseTensor<long>(tokenizedInput.TokenTypeIds.ToArray(), new[] { 1, tokenizedInput.TokenTypeIds.Count });

        // Prepare inputs for the ONNX model
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("input_ids", inputIdsTensor),
            NamedOnnxValue.CreateFromTensor("attention_mask", attentionMaskTensor),
            NamedOnnxValue.CreateFromTensor("token_type_ids", tokenTypeIdsTensor)
        };

        // Run the model and get the output
        using var results = _session.Run(inputs);
        var embeddingsTensor = results.First().AsTensor<float>();

        // Apply mean pooling
        var pooledEmbedding = PoolEmbeddings(embeddingsTensor);

        return pooledEmbedding;
    }

    private List<float> PoolEmbeddings(Tensor<float> embeddingsTensor)
    {
        var embeddingDimension = embeddingsTensor.Dimensions[2];
        var pooledEmbedding = new float[embeddingDimension];

        for (int i = 0; i < embeddingDimension; i++)
        {
            float sum = 0;
            for (int j = 0; j < embeddingsTensor.Dimensions[1]; j++)
            {
                sum += embeddingsTensor[0, j, i];
            }
            pooledEmbedding[i] = sum / embeddingsTensor.Dimensions[1];
        }

        return pooledEmbedding.ToList();
    }
}


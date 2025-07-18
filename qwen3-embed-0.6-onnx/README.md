---
library_name: transformers.js
base_model: Qwen/Qwen3-Embedding-0.6B
pipeline_tag: feature-extraction
---

https://huggingface.co/Qwen/Qwen3-Embedding-0.6B with ONNX weights to be compatible with Transformers.js.

## Usage (Transformers.js)

If you haven't already, you can install the [Transformers.js](https://huggingface.co/docs/transformers.js) JavaScript library from [NPM](https://www.npmjs.com/package/@huggingface/transformers) using:
```bash
npm i @huggingface/transformers
```

You can then compute embeddings as follows:
```js
import { pipeline, matmul } from "@huggingface/transformers";

// Create a feature extraction pipeline
const extractor = await pipeline(
  "feature-extraction",
  "onnx-community/Qwen3-Embedding-0.6B-ONNX",
  {
    dtype: "fp32", // Options: "fp32", "fp16", "q8"
    // device: "webgpu",
  },
);

function get_detailed_instruct(task_description, query) {
  return `Instruct: ${task_description}\nQuery:${query}`;
}

// Each query must come with a one-sentence instruction that describes the task
const task = "Given a web search query, retrieve relevant passages that answer the query";
const queries = [
  get_detailed_instruct(task, "What is the capital of China?"),
  get_detailed_instruct(task, "Explain gravity"),
];

// No need to add instruction for retrieval documents
const documents = [
  "The capital of China is Beijing.",
  "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.",
];
const input_texts = [...queries, ...documents];

// Extract embeddings for queries and documents
const output = await extractor(input_texts, {
  pooling: "last_token",
  normalize: true,
});
const scores = await matmul(
  output.slice([0, queries.length]), // Query embeddings
  output.slice([queries.length, null]).transpose(1, 0), // Document embeddings
);
console.log(scores.tolist());
// [
//   [ 0.7645590305328369, 0.14142560958862305 ],
//   [ 0.13549776375293732, 0.599955141544342 ]
// ]
```

---

Note: Having a separate repo for ONNX weights is intended to be a temporary solution until WebML gains more traction. If you would like to make your models web-ready, we recommend converting to ONNX using [🤗 Optimum](https://huggingface.co/docs/optimum/index) and structuring your repo like this one (with ONNX weights located in a subfolder named `onnx`).
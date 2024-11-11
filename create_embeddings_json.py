from transformers import AutoTokenizer
from onnxruntime import InferenceSession
import json
import numpy as np
import os

def load_model_and_tokenizer(model_dir):
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    # Load ONNX model
    model_path = os.path.join(model_dir, "onnx", "model.onnx")
    session = InferenceSession(model_path)
    return tokenizer, session

def generate_embedding(text, tokenizer, session):
    # Tokenize the input text
    tokens = tokenizer(text, padding="max_length", truncation=True, max_length=128, return_tensors="np")
    input_ids = tokens["input_ids"]
    attention_mask = tokens["attention_mask"]
    token_type_ids = tokens.get("token_type_ids", np.zeros_like(input_ids))

    # Prepare input for the ONNX model
    inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids
    }

    # Run the model and get embeddings
    outputs = session.run(None, inputs)
    # Apply mean pooling along the sequence length dimension (axis=1)
    embedding = np.mean(outputs[0], axis=1).squeeze().tolist()  # Get a single 1D array
    return embedding

def process_input_file(input_file, output_file, model_dir):
    # Load the model and tokenizer
    tokenizer, session = load_model_and_tokenizer(model_dir)

    # Load input data
    with open(input_file, "r") as f:
        data = json.load(f)

    # Process each entry to generate embeddings
    results = []
    for entry in data:
        input_text = entry.get("input", "")
        embedding = generate_embedding(input_text, tokenizer, session)
        
        # Append results with original data and new embeddings
        results.append({
            "instruction": entry.get("instruction", ""),
            "input": input_text,
            "output": entry.get("output", ""),
            "embedding": embedding
        })

    # Save results to output file
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Embeddings generated and saved to {output_file}")

# Usage
if __name__ == "__main__":
    input_file = "input_data.json"
    output_file = "output_with_embeddings.json"
    model_dir = "stsb-bert-tiny-onnx"  # Directory containing tokenizer and model.onnx

    process_input_file(input_file, output_file, model_dir)


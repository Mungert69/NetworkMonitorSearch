import sys
from transformers import AutoTokenizer
from onnxruntime import InferenceSession
import numpy as np
import os
import json

def load_model_and_tokenizer(model_dir="stsb-bert-tiny-onnx"):
    # Load tokenizer and ONNX model
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model_path = os.path.join(model_dir, "onnx", "model.onnx")
    session = InferenceSession(model_path)
    return tokenizer, session

def generate_embedding(text, tokenizer, session):
    # Tokenize the input text
    tokens = tokenizer(text, padding="max_length", truncation=True, max_length=128, return_tensors="np")
    input_ids = tokens["input_ids"]
    attention_mask = tokens["attention_mask"]
    token_type_ids = tokens.get("token_type_ids", np.zeros_like(input_ids))

    # Prepare inputs for ONNX model
    inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids
    }

    # Run the model and get the embedding
    outputs = session.run(None, inputs)
    # Apply mean pooling across the sequence length (axis 1)
    embedding = np.mean(outputs[0], axis=1).squeeze().tolist()  # Squeeze to get a 1D array
    return embedding

def main():
    if len(sys.argv) < 2:
        print("Please provide the input text to generate embedding.")
        print("Usage: python3 make_embedding.py \"Your input text here\"")
        sys.exit(1)

    input_text = sys.argv[1]

    # Load model and tokenizer
    tokenizer, session = load_model_and_tokenizer()

    # Generate embedding
    embedding = generate_embedding(input_text, tokenizer, session)

    # Print the embedding or save to a file if needed
    print("Generated Embedding:", embedding)

    # Optionally, save the embedding to a JSON file
    with open("query_embedding.json", "w") as f:
        json.dump({"text": input_text, "embedding": embedding}, f)
        print("Embedding saved to query_embedding.json")

if __name__ == "__main__":
    main()


import json

def get_embedding_dims(file_path="query_embedding.json"):
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
            embedding = data.get("embedding")
            
            if embedding is None:
                print("No 'embedding' field found in the JSON file.")
                return

            dims = len(embedding)
            print(f"Embedding dimensions (dims): {dims}")
            return dims

    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
    except json.JSONDecodeError:
        print("Error decoding JSON file. Please ensure the file is in the correct format.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Run the function
get_embedding_dims()


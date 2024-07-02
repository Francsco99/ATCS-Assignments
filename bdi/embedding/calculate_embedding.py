import os
import json
import time
from transformers import AutoTokenizer, AutoModel
import torch

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)


def embed_model_text(string, device):

    inputs = tokenizer(string, return_tensors="pt", truncation=True, padding=True).to(device)
    model.to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    # Use the mean of the token embeddings as the sentence embedding
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    # Convert the embedding to a list
    list = embedding.tolist()
    return list

def process_embeddings_from_json(item_title, device):
    item_embedding = {}
    
    for item, pagetitle in item_title.items():
        embedding = embed_model_text(pagetitle, device)
        if embedding is not None:
            item_embedding[item] = embedding
                
    return item_embedding

def count_files(directory_path):

    count = 0
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.json'):
                count += 1
    return count

def assign_device(device_str):
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    elif device_str == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def save_embedding_dictionary(filepath2embedding, output_dir, output_file_name):
    """Save the embedding dictionary to a JSON file in the specified directory."""
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    # Define the file path for the JSON file
    json_file_path = os.path.join(output_dir, output_file_name)
    # Write the dictionary to the JSON file
    with open(json_file_path, 'w') as json_file:
        json.dump(filepath2embedding, json_file, indent=4)

def main(input_file, output_dir):
    print(f"{count_files('monitor/monitor_specs/')} files found")

    # Determine the device to use
    device = assign_device("cuda" if torch.cuda.is_available() else "cpu")

    with open(input_file, encoding='utf-8') as f:
        item2pagetitles = json.load(f)

    # Process the specified directory into a dictionary of embeddings
    start = time.time()
    
    embeddings = process_embeddings_from_json(item2pagetitles, device)

    end = time.time()
    print(f"Processing time (using {device}): {end - start:.2f} seconds")
    
    print("Saving to json...")
    
    file_name = "embeddings_distilbert_base_uncased.json"
    
    save_embedding_dictionary(embeddings, output_dir, file_name)
    
    print(f"Embeddings saved to {output_dir}{file_name}")


if __name__ == "__main__":
    input_file = "preprocessing/preprocessed_dataset_final.json"
    output_dir = "embedding/"
    main(input_file, output_dir)

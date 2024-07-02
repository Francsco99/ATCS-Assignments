import json
import time
import torch
from hdbscan import HDBSCAN

def load_embeddings(file_path):
    """
    Carica gli embedding da un file JSON.
    """
    with open(file_path, 'r') as f:
        embeddings = json.load(f)
    return embeddings

def cluster_embeddings(embeddings, min_cluster_size=5, min_samples=1):
    """
    Esegue il clustering sugli embedding utilizzando HDBSCAN.
    """
    embeddings_array = torch.tensor([value for value in embeddings.values()])
    clusterer = HDBSCAN(metric='euclidean', min_cluster_size=min_cluster_size, min_samples=min_samples)
    cluster_labels = clusterer.fit_predict(embeddings_array.numpy())
    return cluster_labels

def save_clusters(item2embedding, cluster_labels, output_path):
    """
    Salva i risultati del clustering in un file JSON.
    """
    print("Saving clusters to JSON...")
    clusters = {}
    for i, label in enumerate(cluster_labels):
        filename = list(item2embedding.keys())[i]
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(filename)

    # Convert cluster labels to int (if necessary) for JSON serialization
    clusters = {int(k): v for k, v in clusters.items()}

    with open(output_path, 'w') as f:
        json.dump(clusters, f, indent=4)

    print(f"Clusters saved successfully to " + output_path)

def main():
    input_file = "embedding/embeddings_distilbert_base_uncased.json"
    output_file = "clustering/hdbscan_clusters.json"
    
    item2embedding = load_embeddings(input_file)
    print("Embeddings loaded successfully")

    start = time.time()

    cluster_labels = cluster_embeddings(item2embedding, min_cluster_size=5, min_samples=1)

    print("Clustering completed")

    save_clusters(item2embedding, cluster_labels, output_file)

    end = time.time()
    print(f"CLUSTERING --- Processing time: {end - start:.2f} seconds")

if __name__ == "__main__":
    print("Elaboration starting...")
    main()

import json
import paths
from itertools import combinations
import requests

def load_clusters(file_path):
    with open(file_path, 'r') as f:
        clusters = json.load(f)
    return clusters

def load_item2pagetitle(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        item2pagetitle = json.load(f)
    return item2pagetitle

def query_huggingface(title1, title2, begin, huggingface_api_key):
    headers = {
        "Authorization": f"Bearer {huggingface_api_key}",
        "Content-Type": "application/json"
    }
    
    prompt = f'''You are a helpful assistant that can tell if two monitors are the same object just by analyzing their product webpage title.
    To help yourself, search for model names or alphanumerical strings and try to ignore the webpage name in the webpage titles.
    If the page titles represent the same entity your answer MUST BE "yes".
    If the page titles do not represent the same entity your answer MUST BE "no".

    Example 1:
    first page title: "Hp Hewlett Packard HP Z22i D7Q14AT ABB Planet Computer.it"
    second page title: "HP Z22i - MrHighTech Shop"
    "yes"
    Example 2:
    first page title: "Hp Hewlett Packard HP Z22i D7Q14AT ABB Planet Computer.it"
    second page title: "C4D33AA#ABA - Hp Pavilion 20xi Ips Led Backlit Monitor - PC-Canada.com"
    "no"
    Now tell me if these two webpage titles represent the same object:
    first webpage title: "{title1}"
    second webpage title: "{title2}"
    Answers MUST BE "yes" or "no"
    '''
    
    data = {
        "inputs": prompt,
        "parameters": {
            "max_length": 50,
            "temperature": 0.15
        }
    }
    
    response = requests.post("https://api-inference.huggingface.co/models/gpt-3.5-turbo", headers=headers, json=data)
    response.raise_for_status()
    answer = response.json()[0]["generated_text"].strip()
    return answer

def pairwise_matching(clusters, item2pagetitle, num_clusters_to_analyze, huggingface_api_key):
    matches = {}
    begin = True
    
    for cluster_id, items in clusters.items():
        if num_clusters_to_analyze == 0:
            break
        matches[cluster_id] = []
        item_pairs = combinations(items, 2)
        for item1, item2 in item_pairs:
            title1 = item2pagetitle[item1]
            title2 = item2pagetitle[item2]
            match_status = query_huggingface(title1, title2, begin, huggingface_api_key)
            result = (item1, item2, match_status)
            print(result)
            matches[cluster_id].append(result)
            begin = False
        num_clusters_to_analyze -= 1
    
    return matches

def save_matches(matches, output_file):
    with open(output_file, 'w') as f:
        json.dump(matches, f, indent=4)

def calculate_combinations_for_cluster(cluster):
    return len(list(combinations(cluster, 2)))

def calculate_combinations_for_clusters(clusters):
    return sum(calculate_combinations_for_cluster(cluster) for cluster in clusters.values())

def main():
    huggingface_api_key = "YOUR_HUGGINGFACE_API_KEY"
    
    clusters_file = paths.RESULTS_DIR + "/clustering/hdbscan/hdbscan_clusters.json"
    clusters = load_clusters(clusters_file)
    
    del clusters["-1"]
    
    print(f"Calcolo delle combinazioni da effettuare (su {str(len(clusters))} cluster): {calculate_combinations_for_clusters(clusters)}")
    
    item2pagetitle_file = paths.RESULTS_DIR + "/preprocessing/item2pagetitle.json"
    item2pagetitle = load_item2pagetitle(item2pagetitle_file)
    
    num_clusters_to_analyze = 1
    matches = pairwise_matching(clusters, item2pagetitle, num_clusters_to_analyze, huggingface_api_key)
    
    output_file = paths.RESULTS_DIR + "/pairwise_matching/matches.json"
    save_matches(matches, output_file)

if __name__ == "__main__":
    main()

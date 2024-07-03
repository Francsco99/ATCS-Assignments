import json
import pandas as pd

# Carica il file JSON
with open('linkage/matches_model_name.json', 'r') as file:
    data = json.load(file)

# Inizializza una lista per contenere le coppie
pairs = []

# Itera attraverso il dizionario e aggiungi le coppie alla lista
for key, value in data.items():
    for pair in value:
        # Estrai le prime due voci di ogni sottolista
        left_item_spec = pair[0]
        right_item_spec = pair[1]
        pairs.append([left_item_spec, right_item_spec])

# Crea un DataFrame con i dati
df = pd.DataFrame(pairs, columns=["left_item_spec", "right_item_spec"])

# Salva il DataFrame in un file CSV
csv_path = 'linkage/item_specs_gpt2_model_name.csv'
df.to_csv(csv_path, index=False)

print(f"File CSV salvato come {csv_path}")

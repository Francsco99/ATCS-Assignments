import json
import csv

#Leggi i dati dai JSON
with open('preprocessing/preprocessed_dataset_final.json', 'r', encoding='utf-8') as f:
    models_data = json.load(f)

with open('preprocessing/item_to_producer.json', 'r', encoding='utf-8') as f:
    brands_data = json.load(f)

#Crea il file CSV e scrivi le colonne
with open('preprocessing/output.csv', 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['item', 'model', 'brand']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for item, model in models_data.items():
        brand = brands_data.get(item, '')  # Ottieni il brand corrispondente, se esiste
        writer.writerow({'item': item, 'model': model, 'brand': brand})
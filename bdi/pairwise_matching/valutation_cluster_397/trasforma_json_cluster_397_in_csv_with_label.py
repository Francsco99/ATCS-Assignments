import json
import csv

# Leggi il file JSON
with open('matches_cluster_397_MATCH_NO_MATCH.json', 'r') as json_file:
    data = json.load(json_file)

# Apri un file CSV per la scrittura
with open('item_specs_lama_cluster_397_with_label.csv', 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    
    # Scrivi l'intestazione del CSV
    csv_writer.writerow(['left_item_spec', 'right_item_spec', 'label'])
    
    # Trasforma i dati e scrivi nel CSV
    for item in data['397']:
        left_spec_id = item[0]
        right_spec_id = item[1]
        label = 1 if item[2] == "MATCH" else 0
        csv_writer.writerow([left_spec_id, right_spec_id, label])

print("Conversione completata. File CSV salvato come 'item_specs_lama_cluster_397_with_label.csv'.")

import csv
from itertools import combinations

# Dati di input (sostituisci con il percorso del tuo file se necessario)
data = """entity_id,spec_id
ENTITY#010,www.best-deal-items.com//346
ENTITY#010,www.best-deal-items.com//1174
ENTITY#010,www.best-deal-items.com//1351
ENTITY#010,www.best-deal-items.com//1478
ENTITY#010,www.best-deal-items.com//1765
ENTITY#010,www.best-deal-items.com//2563
ENTITY#010,www.ebay.com//9342
ENTITY#010,www.ebay.com//14838
ENTITY#010,www.ebay.com//19295
ENTITY#010,www.ebay.com//21363
ENTITY#010,www.ebay.com//22111
ENTITY#011,www.best-deal-items.com//117
ENTITY#011,www.best-deal-items.com//775
ENTITY#011,www.best-deal-items.com//1413
ENTITY#011,www.best-deal-items.com//1548
ENTITY#011,www.ebay.com//11620
ENTITY#011,www.ebay.com//14209
ENTITY#011,www.ebay.com//17066
ENTITY#011,www.ebay.com//18840
ENTITY#011,www.ebay.com//20888
ENTITY#011,www.ebay.com//20923
ENTITY#011,www.ebay.com//21596
ENTITY#011,www.ebay.com//22417
"""

# Creazione di un dizionario per memorizzare i spec_id per ogni entity_id
entity_dict = {}

# Lettura dei dati
reader = csv.reader(data.strip().split('\n'))
header = next(reader)  # Salta l'intestazione

for row in reader:
    entity_id, spec_id = row
    if entity_id not in entity_dict:
        entity_dict[entity_id] = []
    entity_dict[entity_id].append(spec_id)

# Generazione delle coppie e stampa dei risultati
with open('ground_truth_ridotto_397.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['left_spec_id', 'right_spec_id'])
    
    for entity_id, spec_ids in entity_dict.items():
        for left_spec_id, right_spec_id in combinations(spec_ids, 2):
            writer.writerow([left_spec_id, right_spec_id])

print("Coppie generate e salvate in ground_truth_ridotto_397.csv.")

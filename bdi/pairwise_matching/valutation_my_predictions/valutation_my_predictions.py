import pandas as pd

# Leggi i file CSV
predizioni_mie = pd.read_csv('predizioni_mie.csv')
ground_truth = pd.read_csv('ground_truth.csv')

# Converte il ground_truth in un dizionario per accesso rapido
ground_truth_dict = {}
for _, row in ground_truth.iterrows():
    pair = frozenset([row['left_spec_id'], row['right_spec_id']])
    ground_truth_dict[pair] = row['label']

# Inizializza i contatori
true_positive = 0
false_positive = 0
considered_pairs = set()

# Scorre le predizioni e confronta con il ground_truth
for _, row in predizioni_mie.iterrows():
    pair = frozenset([row['left_spec_id'], row['right_spec_id']])
    if pair in ground_truth_dict:
        considered_pairs.add(pair)
        if ground_truth_dict[pair] == 1:
            true_positive += 1
        else:
            false_positive += 1

# Calcola i falsi negativi
false_negative = 0
for pair, label in ground_truth_dict.items():
    if label == 1 and pair not in considered_pairs:
        false_negative += 1

# Calcola le metriche di precision, recall e F1 measure
precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
f1_measure = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

# Stampa i risultati
print(f'True Positives: {true_positive}')
print(f'False Positives: {false_positive}')
print(f'False Negatives: {false_negative}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Measure: {f1_measure:.4f}')

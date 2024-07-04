import matplotlib.pyplot as plt

# Valori di precision, recall e f1-score
precision = 0.64
recall = 1
f1_score = 0.77

# Nomi delle metriche
metrics = ['Precision', 'Recall', 'F1-Score']
# Valori delle metriche
values = [precision, recall, f1_score]

# Colori delle barre
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

# Creazione dell'istogramma
plt.bar(metrics, values, color=colors)

# Aggiunta dei valori sopra le barre, con un piccolo offset
for i, value in enumerate(values):
    plt.text(i, value - 0.05, f'{value:.2f}', ha='center', va='bottom', color='black')

# Aggiunta del titolo e delle etichette degli assi
plt.title('Precision, Recall and F1-Score')

plt.ylabel('Values')

# Mostra il grafico
plt.ylim(0, 1)  # Imposta il limite dell'asse y da 0 a 1
plt.show()

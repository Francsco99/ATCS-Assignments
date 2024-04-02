import pandas as pd
import numpy as np
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_subgraphs(csv_file):
    # Carica i dati dal CSV
    data = pd.read_csv(csv_file)

    # Estrai i valori degli user_id e delle misurazioni per ogni iterazione
    user_ids = data.columns[1:4].tolist()
    iterations = data['iteration'].unique()

    # Calcola il numero totale di iterazioni
    num_iterations = len(iterations)
    
    # Imposta la dimensione della figura in base al numero di iterazioni
    plt.figure(figsize=(8*num_iterations, 10))

    # Crea un sottografo per ogni iterazione
    for i, iteration in enumerate(iterations, 1):
        # Filtra i dati per l'iterazione corrente
        iteration_data = data[data['iteration'] == iteration]
        
        # Estrai le misurazioni per l'iterazione corrente
        iteration_measurements = iteration_data.iloc[:, 1:4]
        
        # Crea un nuovo sottografo
        ax = plt.subplot(1, num_iterations, i)
        
        # Disegna un grafico a barre per le misurazioni
        colors = plt.cm.tab10(range(len(user_ids)))
        bar_width = 0.9
        x_positions = np.arange(len(user_ids))
        
        for j, (user_id, measurements) in enumerate(zip(user_ids, iteration_measurements.iloc[0])):
            ax.bar(j, measurements, color=colors[j], width=bar_width)
            ax.text(j, measurements + 0.01, f"{measurements:.2f}", ha='center')
        
        # Aggiungi titolo ed etichette agli assi
        ax.set_title(f"Iteration {iteration}", fontsize=20)
        ax.set_xlabel('UserId', fontsize=15)
        ax.set_ylabel('Satisfaction', fontsize=15)
        
        # Imposta valori più granulari sull'asse y
        ax.set_ylim(0, 1)
        ax.set_yticks(np.arange(0, 1.1, 0.1))
        
        # Etichette degli utenti sull'asse x
        ax.set_xticks(np.arange(len(user_ids)))
        ax.set_xticklabels(user_ids, rotation=45, ha='right')
        
        # Aggiungi una griglia sull'asse y
        ax.grid(axis='y', linestyle='--', alpha=0.8)

    plt.savefig("group-reccomandations/data/graphs/sat-plot.png")

def plot_group_scores(csv_file):
    # Lettura dei dati dal file CSV utilizzando pandas
    data = pd.read_csv(csv_file)

    # Creazione del grafico
    plt.figure(figsize=(10, 6))

    # Linea per group_sat
    plt.plot(data['iteration'], data['group_sat'], marker='o', label='Group Satisfaction', color='tab:blue')

    # Linea per group_dis
    plt.plot(data['iteration'], data['group_dis'], marker='o', label='Group Disagreement', color='tab:orange')
    plt.xticks(data['iteration'])
    plt.yticks([i/10 for i in range(11)])  # Valori da 0 a 1 con passo di 0.1
    # Aggiunta di titoli e label agli assi
    plt.title('Group Satisfaction and Disagreement over Iterations')
    plt.xlabel('Iterations')
    plt.ylabel('Score')

    # Mostra griglia su entrambi gli assi
    plt.grid(True,linestyle="--")

    # Mostra leggenda in alto a destra
    plt.legend()

    plt.savefig("group-reccomandations/data/graphs/dis-sat-plot.png")
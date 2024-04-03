import csv

def load_matrix_from_csv(file_path):
    matrix = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Salta la prima riga (intestazioni delle colonne)
        for row in reader:
            matrix.append([float(value) for value in row[1:]])  # Ignora la prima colonna
    return matrix
def find_similar_users(matrix):
    n = len(matrix)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                #if matrix[i][j] > 0.6 and matrix [i][k] > 0.6 and matrix[j][k] > 0.6 and i!=0:
                #if i!=0 and ((matrix[i][j] > 0.6 and matrix[i][k] > 0.6 and matrix[j][k] < 0.4) or
    #(matrix[i][j] > 0.6 and matrix[i][k] < 0.4 and matrix[j][k] > 0.6) or
    #(matrix[i][j] < 0.4 and matrix[i][k] > 0.6 and matrix[j][k] > 0.6)):
                if i!=0 and matrix[i][j] < 0.4 and matrix [i][k] < 0.4 and matrix[j][k] > 0.4 and matrix[i][j] > 0 and matrix [i][k] > 0 and matrix[j][k] > 0:
                    return [i, j, k]
    return None

matrix = load_matrix_from_csv('data/matrices/pearson2.csv')
group = find_similar_users(matrix)

if group:
    print("Gruppo di tre userId con similarità > 0.6:")
    print(group)
else:
    print("Nessun gruppo di tre userId con similarità > 0.6 trovato.")
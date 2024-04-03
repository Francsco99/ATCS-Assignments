import single as sin
df = sin.load_data("data/ml-latest-small/ratings.csv")
matrix = sin.calculate_user_similarity_matrix(df)
sin.save_similarity_matrix_to_csv(matrix,'data/matrices/pearson2.csv')
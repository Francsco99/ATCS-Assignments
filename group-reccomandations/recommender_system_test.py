import recommender_system as rs

RATINGS_PATH="group-reccomandations-1/ml-latest-small/ratings.csv"
MOVIES_PATH="group-reccomandations-1/ml-latest-small/movies.csv"

data_ratings = rs.load_data(RATINGS_PATH)
data_movies= rs.load_data(MOVIES_PATH)
avgs = rs.mean_ratings(data_ratings)
medians = rs.median_ratings(data_ratings)

output_path = "group-reccomandations-1/matrices/jac-constr-pearson.csv"

matrix = rs.calculate_user_similarity_matrix(data_ratings)
rs.save_similarity_matrix_to_csv(matrix,output_path)
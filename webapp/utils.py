import pandas as pd
import os

def load_data(path):
    return pd.read_csv(path)

def find_all_users(df):
    return df['userId'].unique().tolist()

def get_sim_matrix(sim_type,matricesPath):
    PearsonMatrixPath = os.path.join(matricesPath,"pearson.csv")
    JaccardMatrixPath =os.path.join(matricesPath,"jaccard.csv")
    CosineMatrixPath=os.path.join(matricesPath,"cosine.csv")
    ConstrPearsonMatrixPath=os.path.join(matricesPath,"constr-pearson.csv")
    JaccardConstrPearsonMatrixPath=os.path.join(matricesPath,"jac-constr-pearson.csv")

    if sim_type == "pearson":
        similarity_matrix_path = PearsonMatrixPath
    elif sim_type == "cosine":
        similarity_matrix_path = CosineMatrixPath
    elif sim_type == "jaccard":
        similarity_matrix_path = JaccardMatrixPath
    elif sim_type == "constrPearson":
        similarity_matrix_path = ConstrPearsonMatrixPath
    elif sim_type == "jaccard-constrPearson":
        similarity_matrix_path = JaccardConstrPearsonMatrixPath
    else:
        print("Error: Unsupported similarity type")
        return None

    return load_data(similarity_matrix_path)

def mean_ratings(df):
    return df.groupby('userId')['rating'].mean().to_dict()

def get_user_similarity_from_matrix(user_id,similarity_matrix,k):
    user_similarity_row = similarity_matrix.loc[user_id-1]
    user_similarity_dict = user_similarity_row.to_dict()
    del user_similarity_dict["Unnamed: 0"]
    user_similarity_dict = {int(k): v for k, v in user_similarity_dict.items()}
    del user_similarity_dict[user_id]
    sorted_similarity = sorted(user_similarity_dict.items(), key=lambda item: abs(item[1]), reverse=True)
    if k is not None:
        return dict(sorted_similarity[:k])
    else:
        return dict(sorted_similarity)

def find_movies_not_seen_by_user(ratings_df, user, user_subset):
    """
    Find movies not watched by a user but watched by a subset of users.

    Parameters:
    - ratings_df (DataFrame): DataFrame containing ratings data.
    - user (int): ID of the user.
    - user_subset (dict): Dictionary containing user IDs as keys.

    Returns:
    - set: A set of movie IDs not watched by the user but watched by the subset.
    """
    # All the movies watched by the user subset
    movies_seen_by_others = ratings_df[ratings_df['userId'].isin(user_subset.keys())]['movieId'].unique()
    # All the movies watched by the user
    movies_seen_by_user = ratings_df[ratings_df['userId'] == user]['movieId'].unique()
    # Intersection between the previous sets
    movies_not_seen_by_user = set(movies_seen_by_others) - set(movies_seen_by_user)
    return movies_not_seen_by_user

def predict_single_rating(ratings_df, movie, user_a, neighbors, means):
    """
    Predict the rating for a single movie given a user and their neighbors.

    Parameters:
    - ratings_df (DataFrame): DataFrame containing ratings data.
    - movie (int): ID of the movie.
    - user_a (int): ID of the user.
    - neighbors (dict): Dictionary of user IDs and their similarity values.
    - means (dict): Dictionary of user IDs mapped to their average ratings.

    Returns:
    - float: Predicted rating for the movie.
    """
    avg_a = means[user_a]
    #Sums for the prediction function
    numerator_sum = 0
    denominator_sum = 0
    
    #Iterate on all the neighbors
    for user_b, similarity in neighbors.items():
        #Check wether the neighbor has seen the movie
        if has_user_seen_movie(ratings_df, user_b, movie):
            #Get his score for the movie
            rating_b = ratings_df[(ratings_df['userId'] == user_b) & (ratings_df['movieId'] == movie)]['rating'].values[0]
            avg_b = means[user_b]
            #Update sums
            numerator_sum += similarity * (rating_b - avg_b)
            denominator_sum += abs(similarity)
    
    if denominator_sum != 0:
        return avg_a + (numerator_sum / denominator_sum)
    else:
        return -1

def top_k_suggestions_matrix_names(ratings_df,movies_df, similarity_matrix, user_a, k_neighbors,k_movies):
    means = mean_ratings(ratings_df)
    #find the top k_neighbors similar users of user_a
    neighbors = get_user_similarity_from_matrix(user_a,similarity_matrix,k_neighbors)
    #find the movies that user_a has not seen but his neighbors have seen
    unwatched = find_movies_not_seen_by_user(ratings_df,user_a,neighbors)

    suggestions={} #dict to store the suggestions 
    
    #Iterate on all the movies not seen by user_a and predict a score 
    for movie in unwatched:
        suggestions[movie] = predict_single_rating(ratings_df,movie,user_a,neighbors,means)

    suggestions_names={} #dict to store the movie names and their predicted score
    for movie,rating in suggestions.items():
        movie_name =  movies_df.loc[movies_df['movieId'] == movie, 'title'].values[0]
        suggestions_names[movie_name]=rating
    suggested_movies_names = dict(sorted(suggestions_names.items(), key=lambda x: x[1], reverse=True)[:k_movies])
    return suggested_movies_names
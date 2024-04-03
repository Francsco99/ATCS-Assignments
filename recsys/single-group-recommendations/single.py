import pandas as pd
import numpy as np

'''a) Data loading and visualization'''
def load_data(PATH):
    """
    Load data from a CSV file.

    Parameters:
    - PATH (str): Path to the CSV file.

    Returns:
    - DataFrame: Loaded data as a Pandas DataFrame.
    """
    return pd.read_csv(PATH)

def visualize_data(df):
    """
    Visualize the loaded data.

    Parameters:
    - df (DataFrame): Input DataFrame.

    Returns:
    - None
    """
    # Display first 10 rows
    print(df.head(10))
    # Display number of ratings
    print("Number of rows ",len(df))

"""Useful functions"""
def mean_ratings(ratings_df):
    """
    Calculate the average rating for each user.

    Parameters:
    - ratings_df (DataFrame): DataFrame containing ratings data.

    Returns:
    - dict: A dictionary with the average rating for each user {userId: avg_rating}.
    """
    return ratings_df.groupby('userId')['rating'].mean().to_dict()

def median_ratings(ratings_df):
    """
    Calculate the median rating for each user.

    Parameters:
    - ratings_df (DataFrame): DataFrame containing ratings data.

    Returns:
    - dict: A dictionary with the median rating for each user {userId: median_rating}.
    """
    return ratings_df.groupby('userId')['rating'].median().to_dict()

def find_common_movies(ratings_df,user1,user2):
    """
    Find movies watched by both users.

    Parameters:
    - ratings_df (DataFrame): DataFrame containing ratings data.
    - user1 (int): ID of the first user.
    - user2 (int): ID of the second user.

    Returns:
    - list: A list of movie IDs watched by both users.
    """
    # Filter dataframe by userId
    user1_movies = set(ratings_df[ratings_df['userId'] == user1]['movieId'])
    user2_movies = set(ratings_df[ratings_df['userId'] == user2]['movieId'])
    # Find intersection of movieIds
    return list(user1_movies.intersection(user2_movies))

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

def has_user_seen_movie(ratings_df, user_id, movie_id):
    """
    Check if a user has seen a particular movie.

    Parameters:
    - ratings_df (DataFrame): DataFrame containing ratings data.
    - user_id (int): ID of the user.
    - movie_id (int): ID of the movie.

    Returns:
    - bool: True if the user has seen the movie, False otherwise.
    """
    return not ratings_df[(ratings_df['userId'] == user_id) & (ratings_df['movieId'] == movie_id)].empty

def top_rated_movies_dict(ratings_df, movies_df, user_id, k):
    """
    Get a dictionary of top-rated movies for a user.

    Parameters:
    - ratings_df (DataFrame): DataFrame containing ratings data.
    - movies_df (DataFrame): DataFrame containing movie data.
    - user_id (int): ID of the user.
    - k (int): Number of top-rated movies to retrieve.

    Returns:
    - dict: A dictionary with the names of the top k movies rated by the user and their ratings.
    """
    user_movies = ratings_df[ratings_df['userId'] == user_id]
    top_movies = user_movies.sort_values(by='rating', ascending=False).head(k)
    top_movie_ids = top_movies['movieId'].tolist()
    top_movie_titles = movies_df[movies_df['movieId'].isin(top_movie_ids)]['title'].tolist()
    top_movie_ratings = top_movies['rating'].tolist()
    top_movies_dict = {title: rating for title, rating in zip(top_movie_titles, top_movie_ratings)}
    return top_movies_dict

def find_all_users(ratings_df):
    """
    Get a list of all user IDs in the dataset.

    Parameters:
    - ratings_df (DataFrame): DataFrame containing ratings data.

    Returns:
    - list: A list of all user IDs present in the dataset.
    """
    return ratings_df['userId'].unique().tolist()

"""b) user-based collaborative filtering approach, using the Pearson
correlation function for computing similarities between users"""

def pearson_similarity(ratings_df,user_a,user_b,means):
    """
    Calculate the Pearson correlation coefficient between two users.

    Parameters:
    - ratings_df (DataFrame): DataFrame containing ratings data.
    - user_a (int): ID of the first user.
    - user_b (int): ID of the second user.
    - means (dict): Dictionary of user IDs mapped to their average ratings.

    Returns:
    - float: Pearson correlation coefficient between the two users.
    """
    #find the common movies i.e. the movies rated by both users
    common_movies = find_common_movies(ratings_df,user_a,user_b)
    if not common_movies or len(common_movies)<5:
        return 0
    
    #Calculate averages for each user
    mean_a = means[user_a]
    mean_b = means[user_b]

    #Variables for numerator and the denominator
    numerator_sum = 0
    sum_square_a=0
    sum_square_b=0

    #Iterate on the common movies
    for movie in common_movies:
        rating_a = ratings_df.loc[(ratings_df['userId'] == user_a) & (ratings_df['movieId'] == movie), 'rating'].values[0]
        rating_b = ratings_df.loc[(ratings_df['userId'] == user_b) & (ratings_df['movieId'] == movie), 'rating'].values[0]
        #distances from the mean value
        dist_a = rating_a-mean_a
        dist_b = rating_b-mean_b
        #update the numerator
        numerator_sum+= (dist_a*dist_b)
        #update values for the denominator
        sum_square_a += dist_a**2
        sum_square_b += dist_b**2
    
    #calculate the denominator
    denominator = (np.sqrt(sum_square_a))*(np.sqrt(sum_square_b))

    #return if division is possible
    if denominator != 0:
        return numerator_sum/denominator
    return 0

"""d) another similarity function"""
def cosine_similarity(ratings_df,user_a,user_b):
    """
    Calculate the cosine similarity between two users.

    Parameters:
    - ratings_df (DataFrame): DataFrame containing ratings data.
    - user_a (int): ID of the first user.
    - user_b (int): ID of the second user.

    Returns:
    - float: Cosine similarity between the two users.
    """
    #find the common movies i.e. the movies rated by both users
    common_movies = find_common_movies(ratings_df,user_a,user_b)
    if not common_movies:
        return -1
    
    numerator_sum=0
    sum_square_a=0
    sum_square_b=0

    for movie in common_movies:
        rating_a = ratings_df.loc[(ratings_df['userId'] == user_a) & (ratings_df['movieId'] == movie), 'rating'].values[0]
        rating_b = ratings_df.loc[(ratings_df['userId'] == user_b) & (ratings_df['movieId'] == movie), 'rating'].values[0]

        numerator_sum+=rating_a*rating_b
        sum_square_a+=rating_a**2
        sum_square_b+=rating_b**2
    
    denominator_sum= np.sqrt(sum_square_a)*np.sqrt(sum_square_b)

    if denominator_sum!=0:
        return numerator_sum/denominator_sum
    else:
        return 0

def jaccard_similarity(ratings_df,user_a,user_b):
    ratings_a = ratings_df[ratings_df['userId'] == user_a].set_index('movieId')['rating']
    ratings_b = ratings_df[ratings_df['userId'] == user_b].set_index('movieId')['rating']

    intersection = ratings_a.index.intersection(ratings_b.index)

    jaccard_index = len(intersection) / (len(ratings_a) + len(ratings_b) - len(intersection))

    return jaccard_index

def jaccard_pearson_similarity(ratings_df,user_a,user_b,means):

    jaccard_index = jaccard_similarity(ratings_df,user_a,user_b)

    if jaccard_index == 0:
        return 0
    
    pearson_sim = pearson_similarity(ratings_df,user_a,user_b,means)

    jaccard_pearson = jaccard_index * pearson_sim

    return jaccard_pearson

def constrained_pearson_similarity(ratings_df,user_a,user_b,medians):
    """
    Calculate the Constrained Pearson correlation coefficient between two users.

    Parameters:
    - ratings_df (DataFrame): DataFrame containing ratings data.
    - user_a (int): ID of the first user.
    - user_b (int): ID of the second user.
    - medians (dict): Dictionary of user IDs mapped to their medians ratings.

    Returns:
    - float: Pearson correlation coefficient between the two users.
    """
    #find the common movies i.e. the movies rated by both users
    common_movies = find_common_movies(ratings_df,user_a,user_b)
    if not common_movies:
        return -1
    
    #Calculate averages for each user
    median_a = medians[user_a]
    median_b = medians[user_b]

    #Variables for numerator and the denominator
    numerator_sum = 0
    sum_square_a=0
    sum_square_b=0

    #Iterate on the common movies
    for movie in common_movies:
        rating_a = ratings_df.loc[(ratings_df['userId'] == user_a) & (ratings_df['movieId'] == movie), 'rating'].values[0]
        rating_b = ratings_df.loc[(ratings_df['userId'] == user_b) & (ratings_df['movieId'] == movie), 'rating'].values[0]
        #distances from the mean value
        dist_a = rating_a-median_a
        dist_b = rating_b-median_b
        #update the numerator
        numerator_sum+= (dist_a*dist_b)
        #update values for the denominator
        sum_square_a += dist_a**2
        sum_square_b += dist_b**2
    
    #calculate the denominator
    denominator = (np.sqrt(sum_square_a))*(np.sqrt(sum_square_b))

    #return if division is possible
    if denominator != 0:
        return numerator_sum/denominator
    return 0

def jaccard_constrained_pearson_similarity(ratings_df,user_a,user_b,medians):
    jaccard_index = jaccard_similarity(ratings_df,user_a,user_b)

    if jaccard_index==0:
        return 0
    constr_p_sim = constrained_pearson_similarity(ratings_df,user_a,user_b,medians)

    jaccard_constr_sim = jaccard_index * constr_p_sim

    return jaccard_constr_sim

def find_similarity_values(ratings_df,user_a,sim_type,k):
    """
    Find similarity values between a user and other users.

    Parameters:
    - ratings_df (DataFrame): DataFrame containing ratings data.
    - user_a (int): ID of the user.
    - sim_type (str): Type of similarity metric to use ('pearson' or 'cosine').
    - k (int, optional): Number of top similar users to retrieve. Default is None.

    Returns:
    - dict: A dictionary of user IDs and their corresponding similarity values.
    """
    #find all users
    unique_users = ratings_df['userId'].unique()
    means = mean_ratings(ratings_df)
    medians = median_ratings(ratings_df)
    similarities = {} #to store the similarity values
    #Iterate on all the users
    for user_b in unique_users:
        if user_a != user_b:
            if sim_type == "pearson":
                similarities[int(user_b)] = pearson_similarity(ratings_df, user_a, user_b, means)
            elif sim_type == "cosine":
                similarities[int(user_b)] = cosine_similarity(ratings_df, user_a, user_b)
            elif sim_type == "jaccard":
                similarities[int(user_b)] = jaccard_similarity(ratings_df, user_a, user_b)
            elif sim_type == "jaccard-pearson":
                similarities[int(user_b)] = jaccard_pearson_similarity(ratings_df, user_a, user_b, means)
            elif sim_type == "constrPearson":
                similarities[int(user_b)] = constrained_pearson_similarity(ratings_df, user_a, user_b, medians)
            elif sim_type == "jaccard-constrPearson":
                similarities[int(user_b)] = jaccard_constrained_pearson_similarity(ratings_df, user_a, user_b, medians)
            else:
                print("Error: Unsupported similarity type")
                return None
    #Return first k values 
    if k is not None:
        return dict(sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:k])
    #Return all values
    else:
        return dict(sorted(similarities.items(), key=lambda x: x[1], reverse=True))

"""c) the prediction function presented in class for predicting movies scores
We assume that user_a did not watch the movie and
that the neighbors are the 50 most similar users to user_a
Neighbors is a dict in the form of {userId : similarity_value}"""

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

def top_k_suggestions(ratings_df,movies_df,user_a,k_movies,k_neighbors,sim_type):
    """
    Get top k movie suggestions for a user.

    Parameters:
    - ratings_df (DataFrame): DataFrame containing ratings data.
    - movies_df (DataFrame): DataFrame containing movie data.
    - user_a (int): ID of the user.
    - k_movies (int): Number of top movies to suggest.
    - k_neighbors (int): Number of top similar users to consider.
    - sim_type (str): Type of similarity metric to use ('pearson' or 'cosine').

    Returns:
    - dict: A dictionary of top k suggested movies and their predicted scores.
    """
    means = mean_ratings(ratings_df)
    #find the top k_neighbors similar users of user_a
    neighbors = find_similarity_values(ratings_df,user_a,sim_type,k_neighbors)
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

def all_suggestions(ratings_df,user_a,k_neighbors,sim_type):
    """
    Get all movie suggestions for a user.

    Parameters:
    - ratings_df (DataFrame): DataFrame containing ratings data.
    - user_a (int): ID of the user.
    - k_neighbors (int): Number of top similar users to consider.
    - sim_type (str): Type of similarity metric to use ('pearson' or 'cosine').

    Returns:
    - dict: A dictionary of all suggested movies and their predicted scores.
    """
    means = mean_ratings(ratings_df)
    #find the top k_neighbors similar users of user_a
    neighbors = find_similarity_values(ratings_df,user_a,sim_type,k_neighbors)
    #find the movies that user_a has not seen but his neighbors have seen
    unwatched = find_movies_not_seen_by_user(ratings_df,user_a,neighbors)

    suggestions={} #dict to store the suggestions 
    
    #Iterate on all the movies not seen by user_a and predict a score 
    for movie in unwatched:
        suggestions[movie] = predict_single_rating(ratings_df,movie,user_a,neighbors,means)

    return suggestions

"""EXTRA: USE SIMILARITY MATRIX INSTEAD OF CALCULATING SIMILARITIES ON THE GO"""
def get_user_similarity_from_matrix(user_id,similarity_matrix,k):
    """
    Get user similarities for a given user ID from a similarity matrix.

    Parameters:
    - user_id (int): ID of the user.
    - similarity_matrix (DataFrame): DataFrame containing user similarity matrix.

    Returns:
    - dict: A dictionary where keys are other user IDs (int) and values are similarity values (float).
    """
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

def all_suggestions_sim_matrix(ratings_df, similarity_matrix, user_a, k_neighbors):
    """
    Get all movie suggestions for a user using a precomputed user similarity matrix.

    Parameters:
    - ratings_df (DataFrame): DataFrame containing ratings data.
    - similarity_matrix (DataFrame): DataFrame containing user similarity matrix.
    - user_a (int): ID of the user.
    - k_neighbors (int): Number of top similar users to consider.

    Returns:
    - dict: A dictionary of all suggested movies and their predicted scores.
    """
    means = mean_ratings(ratings_df)

    # Get top k_neighbors similar users of user_a from the similarity matrix
    neighbors = get_user_similarity_from_matrix(user_a,similarity_matrix,k_neighbors)

    # Find the movies that user_a has not seen but his neighbors have seen
    unwatched = find_movies_not_seen_by_user(ratings_df, user_a, neighbors)

    suggestions = {}  # Dict to store the suggestions

    # Iterate over all the movies not seen by user_a and predict a score
    for movie in unwatched:
        suggestions[movie] = predict_single_rating(ratings_df, movie, user_a, neighbors, means)
    sorted_suggestions = dict(sorted(suggestions.items(), key=lambda x: x[1], reverse=True))
    return sorted_suggestions

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

def top_k_suggestions_matrix(ratings_df, similarity_matrix, user_a, k_neighbors,k_movies):
    means = mean_ratings(ratings_df)
    #find the top k_neighbors similar users of user_a
    neighbors = get_user_similarity_from_matrix(user_a,similarity_matrix,k_neighbors)
    #find the movies that user_a has not seen but his neighbors have seen
    unwatched = find_movies_not_seen_by_user(ratings_df,user_a,neighbors)

    suggestions={} #dict to store the suggestions 
    
    #Iterate on all the movies not seen by user_a and predict a score 
    for movie in unwatched:
        suggestions[movie] = predict_single_rating(ratings_df,movie,user_a,neighbors,means)
    sorted_suggestions = dict(sorted(suggestions.items(), key=lambda x: x[1], reverse=True)[:k_movies])
    return sorted_suggestions

def calculate_user_similarity_matrix(ratings_df):
    # Get all unique user IDs
    users = ratings_df['userId'].unique()
    means = mean_ratings(ratings_df)
    # Initialize an empty DataFrame to store user similarities
    similarity_matrix = pd.DataFrame(index=users, columns=users)

    # Iterate over all user pairs
    for i, user_a in enumerate(users):
        for j, user_b in enumerate(users):
            if j >= i:  # Avoid calculating similarity for the same pair twice (symmetric matrix)
                similarity = pearson_similarity(ratings_df,user_a,user_b,means)
                similarity_matrix.at[user_a, user_b] = similarity
                similarity_matrix.at[user_b, user_a] = similarity

    return similarity_matrix

def save_similarity_matrix_to_csv(similarity_matrix, output_file):
    similarity_matrix.to_csv(output_file)

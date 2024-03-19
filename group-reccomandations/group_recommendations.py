import recommender_system
import itertools

def common_movies_in_group(filtered_df,group):
    """
    Find common movies rated by all users in the group.

    Parameters:
    - filtered_df (DataFrame): DataFrame containing filtered ratings data.
    - group (list): List of user IDs in the group.

    Returns:
    - set: A set of common movie IDs rated by all users in the group.
    """

    unique_movie_ids = filtered_df['movieId'].unique().tolist()
    common_movie_ids= set() #set to store the common movie ids
    
    #Iterate on each movie id
    for movie_id in unique_movie_ids:
        #If all the users of the group have a rating for that movie, add it in the set
        if all(filtered_df[(filtered_df['movieId'] == movie_id) & (filtered_df['userId'] == user)].shape[0] > 0 for user in group):
            common_movie_ids.add(movie_id)
    return common_movie_ids

def predict_group_ratings(ratings_df, sim_matrix, group,k_neighbors):
    """
    Predict ratings for a group of users based on similarity matrix.

    Parameters:
    - ratings_df (DataFrame): DataFrame containing user ratings data.
    - sim_matrix (DataFrame): DataFrame containing user similarity matrix.
    - group (list): List of user IDs in the group.

    Returns:
    - dict: A dictionary containing predicted ratings for each user in the group.
    """
    predictions = {}
    for user in group:
        predictions[user] = recommender_system.all_suggestions_sim_matrix(ratings_df, sim_matrix, user, k_neighbors)
    return predictions

def normalize_predictions(user_predictions):
    """
    Normalize rating predictions using min-max scaling.

    Parameters:
    - user_predictions (dict): Dictionary containing rating predictions for a user.

    Returns:
    - None
    """
    min_rating = float("inf")
    max_rating = float("-inf")

    for movie,rating in user_predictions.items():
        # Update minimum and maximum predicted ratings
        min_rating = min(min_rating, rating)
        max_rating = max(max_rating, rating) 

    for movie_id, predicted_rating in user_predictions.items():
        # Normalize using min-max scaling
        normalized_rating = 0.5 + ((predicted_rating - min_rating) / (max_rating - min_rating)) * 4.5
        user_predictions[movie_id] = normalized_rating

def extract_and_predict_ratings(ratings_df, sim_matrix, output_file, user_ids,k_neighbors):
    """
    Extract user data, predict ratings using similarity matrix, and save the results to a CSV file.

    Parameters:
    - ratings_df (DataFrame): DataFrame containing user ratings data.
    - sim_matrix (DataFrame): DataFrame containing user similarity matrix.
    - output_file (str): Path to the output CSV file.
    - user_ids (list): List of user IDs to extract and predict ratings for.

    Returns:
    - DataFrame: DataFrame containing the extracted data and predicted ratings.
    """
    # Filter rows corresponding to specified user IDs
    filtered_df = ratings_df[ratings_df['userId'].isin(user_ids)].drop('timestamp', axis=1)

    # Make predictions for the group
    predictions = predict_group_ratings(ratings_df, sim_matrix, user_ids,k_neighbors)
    #for pred in predictions:
        #normalize_predictions(predictions[pred])
    # Add predictions to the filtered DataFrame
    for user, suggestion in predictions.items():
        for movie_id, rating in suggestion.items():
            filtered_df = filtered_df._append({'userId': user, 'movieId': movie_id, 'rating': rating}, ignore_index=True)
    
    # Convert the 'movieId' and 'userId' columns to integers and sort the DataFrame by 'userId'
    filtered_df = filtered_df.astype({'movieId': int, 'userId': int}).sort_values(by='userId')
    
    if output_file is not None:
    # Save the resulting DataFrame to a new CSV file
        filtered_df.to_csv(output_file, index=False)
    return filtered_df

def top_k_suggestions_average(output_file,sim_matrix,ratings_df,movies_df,group,k_movies,k_neighbors):
    """
    Calculate top k movie suggestions based on average ratings across a user group.

    Parameters:
    - output_file (str): Path to the output CSV file.
    - sim_matrix (DataFrame): DataFrame containing user similarity matrix.
    - ratings_df (DataFrame): DataFrame containing user ratings data.
    - movies_df (DataFrame): DataFrame containing movie data.
    - group (list): List of user IDs.
    - k_movies (int): Number of top movie suggestions to return.

    Returns:
    - dict: A dictionary containing the top k movie suggestions and their average ratings.
    """
    #Extract and predict group ratings
    filtered_df = extract_and_predict_ratings(ratings_df,sim_matrix,output_file,group,k_neighbors)
    #Find all common movie ids in the filtered dataframe
    common_movie_ids= common_movies_in_group(filtered_df,group)
    
    average_ratings={} #dict to store the ratings
    #Iterate on each common movie
    for movie_id in common_movie_ids:
        #Extract the ratings for that movie
        movie_ratings = filtered_df[filtered_df['movieId'] == movie_id]['rating']
        average_rating = movie_ratings.mean() #Calculate the average
        average_ratings[movie_id]=average_rating #Add the average to the dict
    
    suggestions_names={} #dict to store the movie names and their predicted score

    for movie,rating in average_ratings.items():
        movie_name =  movies_df.loc[movies_df['movieId'] == movie, 'title'].values[0]
        suggestions_names[movie_name]=rating
    suggested_movies_names = dict(sorted(suggestions_names.items(), key=lambda x: x[1], reverse=True)[:k_movies])
    
    return suggested_movies_names

def top_k_suggestions_least_misery(output_file,sim_matrix,ratings_df,movies_df,group,k_movies,k_neighbors):
    """
    Calculate top k movie suggestions based on least misery ratings across a user group.

    Parameters:
    - output_file (str): Path to the output CSV file.
    - sim_matrix (DataFrame): DataFrame containing user similarity matrix.
    - ratings_df (DataFrame): DataFrame containing user ratings data.
    - movies_df (DataFrame): DataFrame containing movie data.
    - group (list): List of user IDs.
    - k_movies (int): Number of top movie suggestions to return.

    Returns:
    - dict: A dictionary containing the top k movie suggestions and their average ratings.
    """
    #Extract and predict group ratings
    filtered_df = extract_and_predict_ratings(ratings_df,sim_matrix,output_file,group,k_neighbors)
    #Find all common movie ids in the filtered dataframe
    common_movie_ids= common_movies_in_group(filtered_df,group)
    
    minimum_ratings={} #dict to store the ratings
    #Iterate on each common movie
    for movie_id in common_movie_ids:
        #Extract the ratings for that movie
        movie_ratings = filtered_df[filtered_df['movieId'] == movie_id]['rating']
        minimum_rating = min(movie_ratings) #Calculate the average
        minimum_ratings[movie_id]=minimum_rating #Add the average to the dict
    
    suggestions_names={} #dict to store the movie names and their predicted score

    for movie,rating in minimum_ratings.items():
        movie_name =  movies_df.loc[movies_df['movieId'] == movie, 'title'].values[0]
        suggestions_names[movie_name]=rating
    suggested_movies_names = dict(sorted(suggestions_names.items(), key=lambda x: x[1], reverse=True)[:k_movies])
    
    return suggested_movies_names

def pariwise_disagreement(filtered_df,movie_id,group):
    """
    Calculate the pairwise disagreement for a movie among the users in a group.

    Parameters:
    - filtered_df (DataFrame): DataFrame containing filtered ratings data.
    - movie_id (int): ID of the movie.
    - group (list): List of user IDs in the group.

    Returns:
    - float: The pairwise disagreement value for the movie among the users in the group.
    """
    num_members = len(group)

    abs_sum = 0
    for user_a,user_b in itertools.combinations(group,2):
        if user_a != user_b:
            rating_a = filtered_df.loc[(filtered_df['userId'] == user_a) & (filtered_df['movieId'] == movie_id), 'rating'].values[0]
            rating_b = filtered_df.loc[(filtered_df['userId'] == user_b) & (filtered_df['movieId'] == movie_id), 'rating'].values[0]
            abs_sum += abs(rating_a-rating_b)

    return (2/(num_members*(num_members-1)))*abs_sum

def average_pairwise_disagreement(filtered_df,movie_id,group,w):
    """
    Calculate the pairwise disagreement for a movie among the users in a group.

    Parameters:
    - filtered_df (DataFrame): DataFrame containing filtered ratings data.
    - movie_id (int): ID of the movie.
    - group (list): List of user IDs in the group.
    - w (float): Weight value
    Returns:
    - float: The pairwise disagreement value for the movie among the users in the group.
    """
    movie_ratings = filtered_df[filtered_df['movieId'] == movie_id]['rating']
    
    average_rating = movie_ratings.mean() #Calculate the average
    
    disagreement_value = pariwise_disagreement(filtered_df,movie_id,group)
    
    return ((1-w) * average_rating)+(w * disagreement_value)

def top_k_suggestions_avg_disagreement(output_file,sim_matrix,ratings_df,movies_df,group,k_movies,k_neighbors,w):
    """
    Calculate top k movie suggestions based on average misery ratings across a user group taking into account also the disagreement.

    Parameters:
    - output_file (str): Path to the output CSV file.
    - sim_matrix (DataFrame): DataFrame containing user similarity matrix.
    - ratings_df (DataFrame): DataFrame containing user ratings data.
    - movies_df (DataFrame): DataFrame containing movie data.
    - group (list): List of user IDs.
    - k_movies (int): Number of top movie suggestions to return.
    - w (float): Weight for the disagreement score

    Returns:
    - dict: A dictionary containing the top k movie suggestions and their average ratings.
    """
    #Extract and predict group ratings
    filtered_df = extract_and_predict_ratings(ratings_df,sim_matrix,output_file,group,k_neighbors)
    #Find all common movie ids in the filtered dataframe
    common_movie_ids= common_movies_in_group(filtered_df,group)
    
    avg_disagreement_ratings={} #dict to store the ratings
    #Iterate on each common movie
    for movie_id in common_movie_ids:
        #Extract the ratings for that movie
        rating = average_pairwise_disagreement(filtered_df,movie_id,group,w)
        avg_disagreement_ratings[movie_id]=rating #Add the average to the dict
    
    suggestions_names={} #dict to store the movie names and their predicted score

    for movie,rating in avg_disagreement_ratings.items():
        movie_name =  movies_df.loc[movies_df['movieId'] == movie, 'title'].values[0]
        suggestions_names[movie_name]=rating
    suggested_movies_names = dict(sorted(suggestions_names.items(), key=lambda x: x[1], reverse=True)[:k_movies])
    
    return suggested_movies_names
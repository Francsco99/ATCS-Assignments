import pandas as pd
import recommender as rc
import time
import pandas as pd
import numpy as np
from itertools import combinations

def single_user_satisfaction(user,df,group_predictions):
    """
    Calculate the satisfaction score of a single user within a group based on their ratings compared to group predictions.

    Args:
    - user (int): The user ID for whom satisfaction score needs to be calculated.
    - df (DataFrame): DataFrame containing user ratings data with columns: userId, movieId, rating.
    - group_predictions (dict): Dictionary containing movieId -> rating predictions for the group.

    Returns:
    - float: Satisfaction score of the user within the group.
    """
    numerator=0
    #iterate on each predicted movie for the group
    for movie_id in group_predictions.keys():
        #get the rating of the user for that movie
        rating = df[(df['userId'] == user) & (df['movieId'] == movie_id)]['rating'].values
        numerator+=rating
    
    #how many movies in the group predictions
    top_k = len(group_predictions)
    #get the ratings of the user
    top_k_user_ratings = df[df['userId'] == user].head(top_k)
    denominator = top_k_user_ratings['rating'].sum()
    return numerator[0]/denominator if denominator!=0 else 0

def all_users_satisfactions(group,df,group_predictions):
    """
    Calculate the satisfaction scores of a group of users based on their ratings compared to group predictions.

    Args:
    - group (list): List of user IDs in the group.
    - df (DataFrame): DataFrame containing user ratings data with columns: userId, movieId, rating.
    - group_predictions (dict): Dictionary containing movieId -> rating predictions for the group.

    Returns:
    - list: List of satisfaction scores for each user in the group.
    """
    group_sat=[]
    for user in group:
        group_sat.append(single_user_satisfaction(user,df,group_predictions))
    return group_sat

def group_disagreement(all_users_sat):
    """
    Calculate the disagreement among satisfaction scores within a group.

    Args:
    - group_sat (list): List of satisfaction scores for users in the group.

    Returns:
    - float: Disagreement among satisfaction scores within the group.
    """
    return max(all_users_sat)-min(all_users_sat)

def new_alpha_disagreement(all_users_sat):
    """
    Obtain the new value for the "alpha" parameter by considering the Disagreements between the Satisfactions of the Users in the Group.

    Parameters:
    - all_users_sat (list): A list of Satisfactions of the Users in the Group, computed for the current iteration.
    """
    values = all_users_sat

    value_pairs = list(combinations(values, 2))  # Pairs: (Satisfaction User_i, Satisfaction User_j)
    disagreements = np.zeros(len(value_pairs))

    for i, pair in enumerate(value_pairs):
        # disagreement(User_i, User_j) = |Satisfaction User_i - Satisfaction User_j|
        disagreements[i] = np.abs(pair[0] - pair[1])
    
    disagreement_pairs = list(combinations(disagreements, 2))   # Pairs: (Disagreement UserPair_k, Disagreement UserPair_l)
    averages = np.zeros(len(values))
    for i, pair in enumerate(disagreement_pairs):
        averages[i] = np.mean(pair) # For each pair of disagreements, we compute the average
    
    # The new value for "alpha" is the median of the averages
    return np.median(averages)

def update_users_ratings(df,group_pred):
    """
    Update the user ratings DataFrame by removing movies for which group predictions have been made.

    Args:
    - df (DataFrame): DataFrame containing user ratings data with columns: userId, movieId, rating.
    - group_pred (dict): Dictionary containing movieId -> rating predictions made for the group.

    Returns:
    - DataFrame: Updated DataFrame after removing movies for which predictions have been made.
    """
    movie_ids_to_remove = list(group_pred.keys())
    filtered_df = df[~df['movieId'].isin(movie_ids_to_remove)]
    return filtered_df

def process_df(df):
    """
    Process the DataFrame to calculate mean and minimum ratings for each movie.

    Args:
    - df (DataFrame): DataFrame containing user ratings data with columns: userId, movieId, rating.

    Returns:
    - DataFrame: Processed DataFrame with mean and minimum ratings calculated for each movie.
    """
    pivot_df = df.pivot(index='movieId', columns='userId', values='rating')
    pivot_df['mean'] = pivot_df.mean(axis=1)
    pivot_df['min'] = pivot_df.min(axis=1)
    pivot_df.reset_index(inplace=True)
    pivot_df.dropna(inplace=True)
    return pivot_df

def save_to_csv(group,all_users_sat,path):
    group_dis = {}
    group_sat = {}
    iterations = list(all_users_sat.keys())
    for k,v in all_users_sat.items():
        group_dis[k]=group_disagreement(v)
        group_sat[k]=np.mean(v)
    
    df = pd.DataFrame(all_users_sat)

    df=df.T
    df.columns = group

    df.insert(0, 'iteration', iterations)
    # Add 'group_sat' column
    df['group_sat'] = pd.Series(group_sat)
    
    # Add 'group_dis' column
    df['group_dis'] = pd.Series(group_dis)

    # Save the DataFrame to a CSV file
    df.to_csv(path, index=False)

def sdda(df,alpha): 
    """
    Calculate group predictions using the Sequential Deterministic Decremental Aggregation (SDDA) algorithm.

    Args:
    - df (DataFrame): DataFrame containing user ratings data with columns: userId, movieId, rating.
    - alpha (float): Weight parameter for balancing average and minimum ratings.

    Returns:
    - dict: Dictionary containing movieId -> group rating predictions.
    """
    #dict {movieId -> score} to store predicted group ratings
    group_rec={}
    #enrich users_ratings with mean and min values
    df_proc= process_df(df)
    for index,row in df_proc.iterrows():
        movie_id = int(row['movieId'])
        avg_rating = row['mean']
        min_rating = row['min']
        #calculate group rating
        group_rating = ((1-alpha)*avg_rating) + (alpha*min_rating)
        group_rec[movie_id]=group_rating
    #sort the ratings
    sorted_group_rec = dict(sorted(group_rec.items(), key=lambda x: x[1], reverse=True))
    return sorted_group_rec

def sequential_recommender(group,iterations,df,top_k):
    """
    Perform sequential recommendation process for a group of users.

    Args:
    - group (list): List of user IDs in the group.
    - iterations (int): Number of iterations to perform for recommendations.
    - df (DataFrame): DataFrame containing user ratings data with columns: userId, movieId, rating.
    - top_k (int): Number of top predictions to make in each iteration.

    Returns:
    - dict: Dictionary containing all group predictions made in each iteration.
    """
    start_time = time.time()
    print("Starting group predictions...")
    users_ratings_df = rc.enrich_group_dataset(df,group)
    print("Ending group predictions...")
    print("Time elapsed:",time.time()-start_time,"seconds\n")
    
    #Start sequential recommendations loop
    users_sat=[]
    alpha=0
    all_group_pred={}
    all_users_sat={}
    for j in range (1,iterations+1):
        current_time = time.time()
        print("Starting iteration",j)
        group_predictions = sdda(users_ratings_df,alpha)
        top_k_group_predictions = dict(list(group_predictions.items())[:10])
        all_group_pred[j]=top_k_group_predictions
        print("Top",top_k,"group predictions for iter.:",j)
        print(list(top_k_group_predictions.keys()))
        users_sat = all_users_satisfactions(group,users_ratings_df,dict(list(group_predictions.items())[:30]))
        all_users_sat[j]=users_sat
        print("Group satisfactions for iter.",j,users_sat)
        alpha = new_alpha_disagreement(users_sat)
        print("Group disagreement for iter.",j,alpha)
        print("Updating users ratings...")
        users_ratings_df = update_users_ratings(users_ratings_df,top_k_group_predictions)
        print("Ending iteration",j)
        print("Time elapsed:",time.time()-current_time,"seconds\n")
    return all_users_sat

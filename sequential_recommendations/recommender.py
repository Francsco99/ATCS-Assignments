import pandas as pd
import numpy as np
import heapq
import time

def open_file(path):
    return pd.read_csv(path)

def get_common_items(df,user_a,user_b):
    # Filter dataframe by userId
    a_ratings = df[df['userId'] == user_a]
    b_ratings = df[df['userId'] == user_b]

    return pd.merge(a_ratings, b_ratings, on='movieId', how='inner')

def get_mean_values(df,user):
    selected_rows = df[df['userId'] == user]
    return selected_rows['rating'].mean()

def pearson_similarity(df,user_a,user_b):
    common_movies=get_common_items(df,user_a,user_b)
    if common_movies.empty:
        return 0
    ratings_a = common_movies['rating_x']
    ratings_b = common_movies['rating_y']

    mean_a = get_mean_values(df,user_a)
    mean_b = get_mean_values(df,user_b)

    denominator = np.sqrt(np.sum((ratings_a-mean_a)**2)) * np.sqrt(np.sum((ratings_b-mean_b)**2))
    return np.sum((ratings_a-mean_a)*(ratings_b-mean_b))/denominator if denominator != 0 else 0

def get_neighbors(df,user,size,score_function=pearson_similarity):
    top_scores = []
    candidate_neighbors = df.loc[df['userId'] != user, 'userId'].unique()
    for neighbor in candidate_neighbors:
        sim_value = score_function(df,user,neighbor)
        heapq.heappush(top_scores,(sim_value,neighbor))

        if len(top_scores) > size:
            heapq.heappop(top_scores)

    neighbors={}
    for sim_value,user_id in top_scores:
        neighbors[user_id]=sim_value
    
    return {k: v for k, v in sorted(neighbors.items(), key=lambda item: item[1], reverse=True)} 

def make_prediction(df,user,item,neighbors):
    mean = get_mean_values(df,user)

    numerator =0
    denominator=0
    for user_id,score in neighbors.items():
        rating_df = df[(df['userId'] == user_id) & (df['movieId'] == item)]['rating']
        if not rating_df.empty:
            user_mean = get_mean_values(df,user_id)
            numerator += score *(rating_df.values[0]-user_mean)
            denominator += abs(score)
    return mean + numerator/denominator if denominator!=0 else 0

def get_recommended_items(df, user, neighbors, size=None):
    items_rated_user = set(df[df['userId'] == user]['movieId'])
    items_rated_neighbors = set()
    
    for neighbor in neighbors.keys():
        n_ratings = set(df[df['userId'] == neighbor]['movieId'])
        items_rated_neighbors.update(n_ratings)
    
    not_rated_items_user = list(items_rated_neighbors - items_rated_user)

    top_predictions = []
    for item in not_rated_items_user:
        prediction = make_prediction(df, user, item, neighbors)
        heapq.heappush(top_predictions, (prediction, item))

        if size is not None and len(top_predictions) > size:
            heapq.heappop(top_predictions)

    predictions = {}
    for prediction, item in top_predictions:
        predictions[item] = prediction
    
    sorted_predictions = {k: v for k, v in sorted(predictions.items(), key=lambda item: item[1], reverse=True)}
    
    return sorted_predictions

#aggiungi gli score predetti a quelli votati dall'utente
def enrich_dataset(df,user):
    neighbors = get_neighbors(df,user,75,pearson_similarity)
    user_ratings = df[df['userId'] == user][['userId','movieId', 'rating']]
    predictions = get_recommended_items(df,user,neighbors)
    pred_df = pd.DataFrame({
    'userId': [user] * len(predictions),
    'movieId': list(predictions.keys()),
    'rating': list(predictions.values())
    }, columns=['userId', 'movieId', 'rating'])
    final_df = pd.concat([user_ratings, pred_df], ignore_index=True)
    sorted_df = final_df.sort_values(by=['userId', 'rating'], ascending=[True, False])
    #sorted_df=sorted_df.drop(columns=['Unnamed: 0'])
    return sorted_df

#dataframe di gruppo
def enrich_group_dataset(df,group):
    final_df = pd.DataFrame()
    for user in group:
        print("Enriching user",user)
        user_ratings = enrich_dataset(df,user)
        final_df = pd.concat([final_df,user_ratings],ignore_index=True)
    return final_df
# print("inizio")
# ratings_df = open_file('group-reccomandations/data/ml-latest-small/ratings.csv')
# group=[1,2]
# final= enrich_group_dataset(ratings_df,group)
# df_proc= final.groupby('movieId')['rating'].agg(['mean', 'min']).reset_index()
# print(df_proc)
# final.to_csv("group-reccomandations/data/csv/final.csv")
# df_proc.to_csv("group-reccomandations/data/csv/df_proc.csv")











# common_movies = get_common_items(ratings_df,1,2)
# print(common_movies['rating_x'],common_movies['rating_y'])      
# print(get_mean_values(ratings_df,1))
# print(get_mean_values(ratings_df,2))
# print(pearson_similarity(ratings_df,1,45))
# start = time.time()
# neighbors = get_neighbors(ratings_df,7,40,pearson_similarity)
# print(get_recommended_items(ratings_df,7,neighbors,10))
# print("\n",time.time()-start)



    

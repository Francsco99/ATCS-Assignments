import pandas as pd
import recommender as rc
import time

def single_user_satisfaction(user,df,group_predictions):
    #df: dataframe con userId , movieId , rating
    #group_predictions: mappa con movieId -> rating
    numerator=0

    for movie_id in group_predictions.keys():
        rating = df[(df['userId'] == user) & (df['movieId'] == movie_id)]['rating'].values
        if len(rating)>0:
            score = rating[0]
        else:
            score = 0
        numerator+=rating
    
    top_k= len(group_predictions)
    top_k_user_ratings = df[df['userId'] == user].head(top_k)
    denominator = top_k_user_ratings['rating'].sum()
    #print("Numerator",numerator)
    #print("Denominator",denominator)
    return numerator[0]/denominator if denominator!=0 else 0

def group_satisfaction(group,df,group_predictions):
    group_sat=[]
    for user in group:
        group_sat.append(single_user_satisfaction(user,df,group_predictions))
    return group_sat

def group_disagreement(group_sat):
    return max(group_sat)-min(group_sat)

def update_users_ratings(df,group_pred):
    movie_ids_to_remove = list(group_pred.keys())
    filtered_df = df[~df['movieId'].isin(movie_ids_to_remove)]
    return filtered_df

def process_df(df):
    pivot_df = df.pivot(index='movieId', columns='userId', values='rating')
    pivot_df['mean'] = pivot_df.mean(axis=1)
    pivot_df['min'] = pivot_df.min(axis=1)
    pivot_df.reset_index(inplace=True)
    pivot_df.dropna(inplace=True)
    return pivot_df
def sdda(df,alpha): 
    group_rec={}
    df_proc= process_df(df)
    for index,row in df_proc.iterrows():
        movie_id = int(row['movieId'])
        avg_rating = row['mean']
        min_rating = row['min']
        group_rating = ((1-alpha)*avg_rating) + (alpha*min_rating)
        group_rec[movie_id]=group_rating
    sorted_group_rec = dict(sorted(group_rec.items(), key=lambda x: x[1], reverse=True))
    return sorted_group_rec

def sequential_recommender(group,iterations,df,top_k):
    start_time = time.time()
    print("Starting group predictions...")
    users_ratings_df = rc.enrich_group_dataset(df,group)
    print("Ending group predictions...")
    print("Time elapsed:",time.time()-start_time,"seconds\n")
    
    #Start sequential recommendations loop
    group_sat=[]
    alpha=0
    all_group_pred={}
    for j in range (1,iterations+1):
        current_time = time.time()
        print("Starting iteration",j)
        group_predictions = sdda(users_ratings_df,alpha)
        top_k_group_predictions = dict(list(group_predictions.items())[:10])
        all_group_pred[j]=top_k_group_predictions
        print("Top",top_k,"group predictions for iter.:",j)
        print(list(top_k_group_predictions.keys()))
        group_sat = group_satisfaction(group,users_ratings_df,dict(list(group_predictions.items())[:50]))
        print("Group satisfactions for iter.",j,group_sat)
        alpha = group_disagreement(group_sat)
        print("Group disagreement for iter.",j,alpha)
        print("Updating users ratings...")
        users_ratings_df = update_users_ratings(users_ratings_df,top_k_group_predictions)
        print("Ending iteration",j)
        print("Time elapsed:",time.time()-current_time,"seconds\n")

df = rc.open_file('group-reccomandations/data/ml-latest-small/ratings.csv')
group=[23,88,530]
sequential_recommender(group,5,df,10)
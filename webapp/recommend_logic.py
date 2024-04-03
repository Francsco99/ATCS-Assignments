import pandas as pd
import os
import heapq


def load_data(path):
    return pd.read_csv(path)

#SINGLE RECOMMENDATIONS

def get_neighbors_from_matrix(user,matrix,k=None):
    # Estrae la riga associata all'userId
    user_row = matrix[matrix['Unnamed: 0'] == user].iloc[:, 1:]
    
    # Ordina i valori delle celle della riga
    sorted_values = user_row.squeeze().sort_values(ascending=False)
    
    if k is None:
        return {int(user_id): float(similarity) for user_id, similarity in sorted_values.items() if user_id != user}
    
    # Altrimenti, restituisce i top k utenti simili, escludendo l'utente stesso (userId)
    else:
        top_k_users = sorted_values.iloc[2:k+2]
        return {int(user_id): float(similarity) for user_id, similarity in top_k_users.items()}

def get_sim_matrix(sim_type,matricesPath):
    PearsonMatrixPath = os.path.join(matricesPath,"pearson2.csv")
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

def top_rated_movies_dict(ratings_df, movies_df, user_id, k):
    user_movies = ratings_df[ratings_df['userId'] == user_id]
    top_movies = user_movies.sort_values(by='rating', ascending=False).head(k)
    top_movie_ids = top_movies['movieId'].tolist()
    top_movie_titles = movies_df[movies_df['movieId'].isin(top_movie_ids)]['title'].tolist()
    top_movie_ratings = top_movies['rating'].tolist()
    top_movies_dict = {title: rating for title, rating in zip(top_movie_titles, top_movie_ratings)}
    return top_movies_dict

def get_all_users(ratings_df):
    return ratings_df['userId'].unique().tolist()

def get_mean_values(df,user):
    selected_rows = df[df['userId'] == user]
    return selected_rows['rating'].mean()

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

def get_recommended_items_names(sim_matrix,k_neighbors,df,movies_df,user,k_movies=None):
    neighbors=get_neighbors_from_matrix(user,sim_matrix,k_neighbors)
    predictions = get_recommended_items(df,user,neighbors,k_movies)
    
    names={}
    for movie,score in predictions.items():
        movie_name =  movies_df.loc[movies_df['movieId'] == movie, 'title'].values[0]
        names[movie_name]=score
    return names




#GROUP RECOMMENDATIONS
#aggiungi gli score predetti a quelli votati dall'utente
def enrich_dataset_matrix(df,user,matrix,size):
    neighbors = get_neighbors_from_matrix(user,matrix,size)
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
def enrich_group_dataset_matrix(df,matrix,group,size):
    final_df = pd.DataFrame()
    for user in group:
        print("Enriching user",user)
        user_ratings = enrich_dataset_matrix(df,user,matrix,size)
        final_df = pd.concat([final_df,user_ratings],ignore_index=True)
    return final_df

def process_df(df):

    pivot_df = df.pivot(index='movieId', columns='userId', values='rating')
    pivot_df['mean'] = pivot_df.mean(axis=1)
    pivot_df['min'] = pivot_df.min(axis=1)
    pivot_df.reset_index(inplace=True)
    pivot_df.dropna(inplace=True)
    return pivot_df

def group_rating_matrix(df,group,matrix,k_neighbors,k_movies,type,w):
    group_rec={}
    df=enrich_group_dataset_matrix(df,matrix,group,k_neighbors)
    df_proc= process_df(df)
    for index,row in df_proc.iterrows():
        movie_id = int(row['movieId'])
        avg_rating = row['mean']
        min_rating = row['min']
        #calculate group rating
        if type=="leastMisery":
            group_rating = min_rating
        elif type=="average":
            group_rating=avg_rating
        elif type == "AverageDisagreement":
            group_rating = (1-w)*avg_rating + w*min_rating
        group_rec[movie_id]=group_rating
    #sort the ratings
    sorted_group_rec = dict(sorted(group_rec.items(), key=lambda x: x[1], reverse=True)[:k_movies])
    return sorted_group_rec

def group_rating_names(movies_df,df,group,matrix,k_neighbors,k_movies,type,w):
    group_ratings= group_rating_matrix(df,group,matrix,k_neighbors,k_movies,type,w)
    names={}
    for movie,rating in group_ratings.items():
        movie_name =  movies_df.loc[movies_df['movieId'] == movie, 'title'].values[0]
        names[movie_name]=rating
    return names    

#group=[1,45,64]
#df = load_data('data/ml-latest-small/ratings.csv')
#matrix = load_data('data/matrices/pearson2.csv')
#print(enrich_group_dataset_matrix(df,matrix,group,40))
import group_recommendations as gr
import time

start=time.time()
# Esempio di utilizzo:
ratings_path = 'group-reccomandations-1/ml-latest-small/ratings.csv'
movies_path='group-reccomandations-1/ml-latest-small/movies.csv'
output_file = 'group-reccomandations-1/ml-latest-small/filtered.csv'
matrix_path = 'group-reccomandations-1/ml-latest-small/similarity_matrix_pearson.csv'

user_ids = [1, 2, 3]  # Lista degli userId da estrarre e per i quali eseguire le previsioni
ratings_df = gr.recommender_system.load_data(ratings_path)
movies_df = gr.recommender_system.load_data(movies_path)
matrix = gr.recommender_system.load_data(matrix_path)

avg_method = gr.top_k_suggestions_average(output_file,matrix,ratings_df,movies_df,user_ids,10)
avg_method_dis = gr.top_k_suggestions_avg_disagreement(output_file,matrix,ratings_df,movies_df,user_ids,10,0.4)
end=time.time()
print("Average method for group",user_ids)
for k,v in avg_method.items():
    print(k,v)
print("-------------------------------------")
print("Average method disagreement for group",user_ids)
for k,v in avg_method_dis.items():
    print(k,v)
print("tempo trascorso",(end-start)/60)
import group_recommendations as gr
import recommender_system as rs
import os

"""File paths"""
absPath = os.path.dirname(os.path.abspath(__file__))
datasetPath = os.path.join(absPath,"data","ml-latest-small")
matricesPath = os.path.join(absPath,"data","matrices")
userRatingsPath = os.path.join(datasetPath,"ratings.csv")
moviesPath = os.path.join(datasetPath,"movies.csv")
PearsonMatrixPath = os.path.join(matricesPath,"pearson.csv")
JaccardMatrixPath =os.path.join(matricesPath,"jaccard.csv")
CosineMatrixPath=os.path.join(matricesPath,"cosine.csv")
ConstrPearsonMatrixPath=os.path.join(matricesPath,"constr-pearson.csv")
JaccardConstrPearsonMatrixPath=os.path.join(matricesPath,"jac-constr-pearson.csv")

"""Data loading"""
ratings_df = rs.load_data(userRatingsPath)
movies_df = rs.load_data(moviesPath)

avgs = rs.mean_ratings(ratings_df)
medians = rs.median_ratings(ratings_df)

#Similarity matrices
pearson_matrix = rs.load_data(PearsonMatrixPath)
jaccard_matrix = rs.load_data(JaccardMatrixPath)
cosine_matrix = rs.load_data(CosineMatrixPath)
constr_pearson_matrix = rs.load_data(ConstrPearsonMatrixPath)
jac_constr_pearson_matrix = rs.load_data(JaccardConstrPearsonMatrixPath)

def print_result(result):
    for k,v in result.items():
        print (k,v)

assignment_choice = None
while assignment_choice not in [1, 2]:
    try:
        assignment_choice = int(input("Choose assignment (1 for Assignment 1, 2 for Assignment 2): "))
        if assignment_choice not in [1, 2]:
            print("Invalid choice. Please choose either 1 or 2.")
    except ValueError:
        print("Invalid input. Please enter a number.")
        
if assignment_choice==1:
    """Assignment 1"""
    user = int(input("Insert userId (1 to 610)"))
    print("Showing results for user",user)
    print("-----------------------------------------------")
    print("Top 10 similar users using Pearson Correlation Coefficient")
    print_result(rs.get_user_similarity_from_matrix(user,pearson_matrix,10))
    print("-----------------------------------------------")
    print("Top 10 similar users using Jaccard - Constrained Pearson Correlation Coefficient")
    print_result(rs.get_user_similarity_from_matrix(user,jac_constr_pearson_matrix,10))
    print("-----------------------------------------------")
    print("Top 10 suggestions using Pearson Correlation Coefficient (Considering 30 similar neighbors)")
    print_result(rs.top_k_suggestions_matrix(ratings_df,movies_df,pearson_matrix,user,30,10))
    print("-----------------------------------------------")
    print("Top 10 suggestions using Jaccard - Pearson Constrained Correlation Coefficient (Considering 30 similar neighbors)")
    print_result(rs.top_k_suggestions_matrix(ratings_df,movies_df,jac_constr_pearson_matrix,user,30,10))
    print("-----------------------------------------------")

elif assignment_choice==2:
    """Assignment 2"""
    group=[]
    group_size=3
    for i in range(group_size):
        group.append(int(input("Insert userId (1 to 610)")))
    print("Showing results for group",group)
    print("-----------------------------------------------")
    print("Top 10 suggestions using Average Method")
    print_result(gr.top_k_suggestions_average(None,pearson_matrix,ratings_df,movies_df,group,10,30))
    print("-----------------------------------------------")
    print("Top 10 suggestions using Least Misery Method")
    print_result(gr.top_k_suggestions_least_misery(None,pearson_matrix,ratings_df,movies_df,group,10,30))
    print("-----------------------------------------------")
    print("Top 10 suggestions using Average Method with Group Disagreements")
    print_result(gr.top_k_suggestions_avg_disagreement(None,pearson_matrix,ratings_df,movies_df,group,10,30,0.3))

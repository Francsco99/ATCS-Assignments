from flask import Flask,render_template, request, jsonify
import recommender_system as rs
import group_recommendations as rg
import os

app = Flask(__name__,static_folder='static')

absPath = os.path.dirname(os.path.abspath(__file__))
datasetPath = os.path.join(absPath,"ml-latest-small")
matricesPath = os.path.join(absPath,"matrices")
userRatingsPath = os.path.join(datasetPath,"ratings.csv")
moviesPath = os.path.join(datasetPath,"movies.csv")


ratings_data = rs.load_data(userRatingsPath)
movies_data = rs.load_data(moviesPath)
users = rs.find_all_users(ratings_data)

min_user_id = min(users)
max_user_id = max(users)

def get_sim_matrix(sim_type):
    PearsonMatrixPath = os.path.join(matricesPath,"pearson.csv")
    JaccardMatrixPath =os.path.join(matricesPath,"jaccard.csv")
    CosineMatrixPath=os.path.join(matricesPath,"cosine.csv")
    ConstrPearsonMatrixPath=os.path.join(matricesPath,"constr-pearson.csv")

    if sim_type == "pearson":
        similarity_matrix_path = PearsonMatrixPath
    elif sim_type == "cosine":
        similarity_matrix_path = CosineMatrixPath
    elif sim_type == "jaccard":
        similarity_matrix_path = JaccardMatrixPath
    elif sim_type == "constrPearson":
        similarity_matrix_path = ConstrPearsonMatrixPath
    else:
        print("Error: Unsupported similarity type")
        return None

    return rs.load_data(similarity_matrix_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/users')
def get_users():
    return jsonify(users)

@app.route('/single_recommendations')
def single_recommendations_page():
    # Codice per la pagina di raccomandazioni singole
    return render_template('single.html',min_user_id=min_user_id,max_user_id=max_user_id)

@app.route('/group_recommendations')
def group_recommendations_page():
    # Codice per la pagina di raccomandazioni di gruppo
    return render_template('group.html',min_user_id=min_user_id,max_user_id=max_user_id)

@app.route('/group-recommendations', methods=['GET'])
def group_recommendations():
    user_id1 = int(request.args.get('userId1'))
    user_id2 = int(request.args.get('userId2'))
    user_id3 = int(request.args.get('userId3'))
    group = [user_id1,user_id2,user_id3]
    top_k_movies = int(request.args.get('topKMovies'))
    top_k_neighbors = int(request.args.get('topKNeighbors'))
    sim_type = str(request.args.get("similarityType"))
    group_type = str(request.args.get("groupType"))
    weight = float(request.args.get('weight'))
    sim_matrix = get_sim_matrix(sim_type)
    if group_type=="average":
        predictions = rg.top_k_suggestions_average(None,sim_matrix,ratings_data,movies_data,group,top_k_movies,top_k_neighbors)
    elif group_type=="leastMisery":
        predictions = rg.top_k_suggestions_least_misery(None,sim_matrix,ratings_data,movies_data,group,top_k_movies,top_k_neighbors)
    elif group_type=="AverageDisagreement":
        predictions = rg.top_k_suggestions_avg_disagreement(None,sim_matrix,ratings_data,movies_data,group,top_k_movies,top_k_neighbors,weight)
    return jsonify(predictions)

@app.route('/recommendations', methods=['GET'])
def recommendations():
    user_id = int(request.args.get('userId'))
    top_k_movies = int(request.args.get('topKMovies'))
    top_k_neighbors = int(request.args.get('topKNeighbors'))
    sim_type = str(request.args.get("similarityType"))
    sim_matrix = get_sim_matrix(sim_type)
    predictions=rs.top_k_suggestions_matrix(ratings_data,movies_data,sim_matrix,user_id,top_k_neighbors,top_k_movies)
    return jsonify(predictions)

@app.route('/top_rated_movies')
def top_rated_movies():
    user_id = int(request.args.get('userId'))
    top_movies = rs.top_rated_movies_dict(ratings_data, movies_data, user_id,10)
    return jsonify(top_movies)

@app.route('/top_similar_users')
def top_similar_users():
    user_id = int(request.args.get('userId'))
    sim_type = str(request.args.get('similarityType'))
    sim_matrix = get_sim_matrix(sim_type)
    top_users = rs.get_user_similarity_from_matrix(user_id,sim_matrix,10)
    return jsonify(top_users)
    

if __name__ == '__main__':
    app.run(debug=True)

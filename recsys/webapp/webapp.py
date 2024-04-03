from flask import Flask,render_template, request, jsonify
import recommend_logic as rl
import os


app = Flask(__name__,static_folder='static')
data_dir = os.path.abspath("recsys/data")
datasetPath = os.path.join(data_dir,"ml-latest-small")
matricesPath = os.path.join(data_dir,"matrices")
userRatingsPath = os.path.join(datasetPath,"ratings.csv")
moviesPath = os.path.join(datasetPath,"movies.csv")

ratings_data = rl.load_data(userRatingsPath)
movies_data = rl.load_data(moviesPath)
users = rl.get_all_users(ratings_data)

min_user_id = min(users)
max_user_id = max(users)

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

@app.route('/recommendations', methods=['GET'])
def recommendations():
    user_id = int(request.args.get('userId'))
    top_k_movies = int(request.args.get('topKMovies'))
    top_k_neighbors = int(request.args.get('topKNeighbors'))
    sim_type = str(request.args.get("similarityType"))
    sim_matrix = rl.get_sim_matrix(sim_type,matricesPath)
    predictions=rl.get_recommended_items_names(sim_matrix,top_k_neighbors,ratings_data,movies_data,user_id,top_k_movies)
    return jsonify(predictions)

@app.route('/top_rated_movies')
def top_rated_movies():
    user_id = int(request.args.get('userId'))
    top_movies = rl.top_rated_movies_dict(ratings_data, movies_data, user_id,10)
    return jsonify(top_movies)

@app.route('/top_similar_users')
def top_similar_users():
    user_id = int(request.args.get('userId'))
    sim_type = str(request.args.get('similarityType'))
    sim_matrix = rl.get_sim_matrix(sim_type,matricesPath)
    top_users = rl.get_neighbors_from_matrix(user_id,sim_matrix,10)
    return jsonify(top_users)

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
    sim_matrix = rl.get_sim_matrix(sim_type,matricesPath)
    predictions = rl.group_rating_names(movies_data,ratings_data,group,sim_matrix,top_k_neighbors,top_k_movies,group_type,weight)
    return jsonify(predictions)

if __name__ == '__main__':
    app.run(debug=True)

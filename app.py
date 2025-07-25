from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import re
app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')
users_df = pd.read_csv(r"C:\Users\Smile\Desktop\newbeat\users.csv", encoding='latin-1')
songs_df = pd.read_csv(r"C:\Users\Smile\Desktop\newbeat\songs.csv", encoding='latin-1')
data = pd.merge(users_df, songs_df, on='Song_ID', how='inner')
data.rename(columns={'User_ID_x': 'User_ID', 'User_ID_y': 'User_ID_y'}, inplace=True)
data.drop(columns=['User_ID_y'], inplace=True)
print("Merged DataFrame columns after renaming and dropping:", data.columns)
user_item_matrix = data.pivot_table(index='User_ID', columns='Song_ID', values='Ratings').fillna(0)
user_item_matrix_sparse = csr_matrix(user_item_matrix)
knn = NearestNeighbors(metric='cosine', algorithm='brute')
knn.fit(user_item_matrix_sparse)
songs_df['combined_features'] = songs_df['Artist'] + " " + songs_df['Genre']
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(songs_df['combined_features'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
def get_collaborative_recommendations(user_id, num_recommendations=5):
    if user_id not in user_item_matrix.index:
        return []
    user_index = user_item_matrix.index.get_loc(user_id)
    distances, indices = knn.kneighbors(user_item_matrix.iloc[user_index, :].values.reshape(1, -1), n_neighbors=num_recommendations + 1)
    valid_indices = indices.flatten()[1:] 
    song_ids = user_item_matrix.columns[valid_indices].tolist()
    return songs_df[songs_df['Song_ID'].isin(song_ids)]['Title'].tolist()
def get_content_based_recommendations(song_id, num_recommendations=5):
    if song_id not in songs_df['Song_ID'].values:
        return []
    song_index = songs_df[songs_df['Song_ID'] == song_id].index[0]
    sim_scores = list(enumerate(cosine_sim[song_index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    song_indices = [i[0] for i in sim_scores[1:num_recommendations + 1]]  
    return songs_df['Title'].iloc[song_indices].tolist()
def get_hybrid_recommendations(user_id, song_id, num_recommendations=5):
    collab_recommendations = get_collaborative_recommendations(user_id, num_recommendations)
    content_recommendations = get_content_based_recommendations(song_id, num_recommendations)
    hybrid_recommendations = list(set(collab_recommendations + content_recommendations))
    return hybrid_recommendations[:num_recommendations]
@app.route('/', methods=['GET', 'POST'])
def recommend():
    if request.method == 'POST':
        song_name = request.form['song_name']
        escaped_song_name = re.escape(song_name)
        song_row = songs_df[songs_df['Title'].str.contains(escaped_song_name, case=False, na=False)]
        if not song_row.empty:
            song_id = song_row.iloc[0]['Song_ID']
            user_id = users_df['User_ID'].iloc[0] 
            recommendations = get_hybrid_recommendations(user_id, song_id, num_recommendations=5)
            return render_template('index.html', recommendations=recommendations)
        else:
            return render_template('index.html', error="Song not found.")
    return render_template('index.html')
if __name__ == "__main__":
    app.run(debug=True)

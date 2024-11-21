from flask import Flask, render_template, request, redirect, url_for, session
from models.recommender import get_recommendations
from models.llm_api import predict_preferences
from models.data_manager import load_dataset, add_rating

app = Flask(__name__)
app.secret_key = '' 

# Load static dataset at startup
static_data = load_dataset()
dynamic_data = []

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

app = Flask(__name__)

# data = {
#     'user_id': [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
#     'movie_id': [1, 2, 3, 1, 2, 4, 2, 3, 4, 1, 3, 4],
#     'rating': [5, 4, 1, 4, 5, 1, 2, 4, 5, 4, 3, 5]
# }

# df = pd.DataFrame(data)
df = pd.read_csv('./data/ratings.csv')

# Sample movie data
# movies = {
#     'movie_id': [1, 2, 3, 4, 5],
#     'title': ['Dilwale Dulhania Le Jayenge', 'Kabhi Khushi Kabhie Gham', 'Dangal', 'Lagaan', 'Gully Boy'],
#     'poster': [
#         'static/Pictures/ddlj.jpg',
#         'static/Pictures/kabhi khushi kabhi gam poster.jpg',
#         'static/Pictures/dangal.jpg',
#         'static/Pictures/lagan.jpg',
#         'static/Pictures/gully boy poster.jpg'
#     ]
# }

# movies_df = pd.DataFrame(movies)
movies_df = pd.read_csv('./data/movies.csv')

# Create user-item matrix
user_item_matrix = df.pivot(index='userId', columns='movieId', values='rating').fillna(0)

# Convert to scipy.sparse.csr_matrix
user_item_matrix_sparse = csr_matrix(user_item_matrix.values)

# Normalize by subtracting mean
user_ratings_mean = np.mean(user_item_matrix_sparse, axis=1)
R_demeaned = user_item_matrix_sparse - user_ratings_mean.reshape(-1, 1)

# Perform SVD
U, sigma, Vt = svds(R_demeaned, k=2)

# Convert sigma to diagonal matrix
sigma = np.diag(sigma)

# Reconstruct matrix
all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
preds_df = pd.DataFrame(all_user_predicted_ratings, columns=user_item_matrix.columns)

def recommend_movies(predictions_df, user_id, movies_df, original_ratings_df, num_recommendations=5):
    user_row_number = user_id #- 1
    sorted_user_predictions = predictions_df.iloc[user_row_number].sort_values(ascending=False)

    user_data = original_ratings_df[original_ratings_df.userId == user_id]
    user_full = (user_data.merge(movies_df, how='left', left_on='movieId', right_on='movieId').
                 sort_values(['rating'], ascending=False))

    recommendations = (movies_df[~movies_df['movieId'].isin(user_full['movieId'])].
                       merge(pd.DataFrame(sorted_user_predictions).reset_index(), how='left',
                             left_on='movieId',
                             right_on='movieId').
                       rename(columns={user_row_number: 'Predictions'}).
                       sort_values('Predictions', ascending=False).
                       iloc[:num_recommendations, :-1])

    return user_full, recommendations


@app.route('/')
def login():
    return render_template('login.html')

@app.route('/main', methods=['GET', 'POST'])
def main():
    # if 'user_id' not in session:
    #     return redirect(url_for('login'))

    # if request.method == 'POST':
    #     # Add user rating to dynamic dataset
    #     movie_id = request.form['movie_id']
    #     rating = request.form['rating']
    #     add_rating(session['user_id'], movie_id, rating, dynamic_data)
    
    # if len(dynamic_data) >= 10:  # Threshold for dynamic recommendations
    #     recommendations = get_recommendations(session['user_id'], dynamic_data)
    # else:
    #     recommendations = predict_preferences(session['user_id'], static_data)
    user_id = 1
    already_rated, recommendations = recommend_movies(preds_df, user_id, movies_df, df)
    all_movies = movies_df[['title','poster']]
    return render_template('index.html', user_id=user_id, already_rated=already_rated, recommendations=recommendations, all_movies=all_movies)

    # return render_template('main.html', recommendations=recommendations)

@app.route('/login', methods=['POST'])
def authenticate():
    user_id = request.form['user_id']
    session['user_id'] = user_id
    return redirect(url_for('main'))

if __name__ == '__main__':
    app.run(debug=True)

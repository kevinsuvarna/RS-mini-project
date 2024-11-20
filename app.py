from flask import Flask, render_template, request, redirect, url_for, session
from models.recommender import get_recommendations
from models.llm_api import predict_preferences
from models.data_manager import load_dataset, add_rating

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with a secure key

# Load static dataset at startup
static_data = load_dataset()
dynamic_data = []

@app.route('/')
def login():
    return render_template('login.html')

@app.route('/main', methods=['GET', 'POST'])
def main():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        # Add user rating to dynamic dataset
        movie_id = request.form['movie_id']
        rating = request.form['rating']
        add_rating(session['user_id'], movie_id, rating, dynamic_data)
    
    if len(dynamic_data) >= 10:  # Threshold for dynamic recommendations
        recommendations = get_recommendations(session['user_id'], dynamic_data)
    else:
        recommendations = predict_preferences(session['user_id'], static_data)

    return render_template('main.html', recommendations=recommendations)

@app.route('/login', methods=['POST'])
def authenticate():
    user_id = request.form['user_id']
    session['user_id'] = user_id
    return redirect(url_for('main'))

if __name__ == '__main__':
    app.run(debug=True)

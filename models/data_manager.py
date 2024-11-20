import pandas as pd
from config import STATIC_DATASET_PATH

def load_dataset():
    """
    Load the static dataset from a file (e.g., MovieLens).
    Returns:
    - DataFrame containing movie data
    """
    return pd.read_csv(STATIC_DATASET_PATH)

def add_rating(user_id, movie_id, rating, dynamic_data):
    """
    Add a user's rating to the dynamic dataset.
    Parameters:
    - user_id: ID of the user
    - movie_id: ID of the movie
    - rating: Rating given by the user
    - dynamic_data: List storing dynamic ratings
    """
    dynamic_data.append({'user_id': user_id, 'movie_id': movie_id, 'rating': rating})

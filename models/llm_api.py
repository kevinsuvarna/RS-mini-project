import requests
from config import GEMINI_API_KEY

def predict_preferences(user_id, static_data):
    """
    Use Gemini's API to predict preferences for new users.
    Parameters:
    - user_id: ID of the user
    - static_data: The static dataset (e.g., MovieLens)
    Returns:
    - Predicted list of preferred movies
    """
    # Example API call structure
    response = requests.post(
        "https://api.gemini.com/predict",
        headers={"Authorization": f"Bearer {GEMINI_API_KEY}"},
        json={"user_id": user_id, "data": static_data}
    )
    return response.json().get('recommendations', [])

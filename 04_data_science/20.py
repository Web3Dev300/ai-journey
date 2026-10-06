import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

def generate_streaming_data():
    """Generates synthetic users, items, and rating interactions."""
    np.random.seed(42)
    
    # 1. Users (Demographics)
    users = pd.DataFrame({
        'user_id': range(1, 101),
        'age': np.random.randint(18, 65, 100),
        'country': np.random.choice(['US', 'UK', 'CA', 'AU'], 100)
    })
    
    # 2. Items (Content)
    movies = pd.DataFrame({
        'movie_id': range(1, 51),
        'title': [f"Movie_{i}" for i in range(1, 51)],
        'genre': np.random.choice(['Action', 'Comedy', 'Drama', 'Sci-Fi', 'Horror'], 50)
    })
    
    # 3. Interactions (Ratings 1-5)
    # Simulating a sparse matrix where users haven't rated everything
    interactions = []
    for user in users['user_id']:
        # Each user rates between 5 and 20 movies
        num_ratings = np.random.randint(5, 21)
        rated_movies = np.random.choice(movies['movie_id'], num_ratings, replace=False)
        for movie in rated_movies:
            # Random rating from 1 to 5
            interactions.append({'user_id': user, 'movie_id': movie, 'rating': np.random.randint(1, 6)})
            
    ratings_df = pd.DataFrame(interactions)
    return users, movies, ratings_df

def build_recommendation_model(ratings_df):
    """Builds a Collaborative Filtering model using Matrix Factorization (SVD)."""
    
    # 1. Create the User-Item Matrix
    # Rows: users, Columns: movies, Values: ratings
    user_item_matrix = ratings_df.pivot(index='user_id', columns='movie_id', values='rating')
    
    # Fill missing values (unseen movies) with 0 for matrix math
    # Note: In advanced systems, we might fill with user or item averages
    user_item_matrix_filled = user_item_matrix.fillna(0)
    
    # 2. Apply Matrix Factorization (Truncated SVD)
    # This compresses the matrix into latent (hidden) features, 
    # capturing the underlying tastes of users and characteristics of movies.
    svd = TruncatedSVD(n_components=10, random_state=42)
    latent_matrix = svd.fit_transform(user_item_matrix_filled)
    
    # Reconstruct the matrix to get predicted ratings for ALL user-movie pairs
    reconstructed_matrix = np.dot(latent_matrix, svd.components_)
    
    # Convert back to a DataFrame for easy lookup
    predicted_ratings_df = pd.DataFrame(
        reconstructed_matrix, 
        index=user_item_matrix.index, 
        columns=user_item_matrix.columns
    )
    
    return user_item_matrix, predicted_ratings_df

def get_recommendations(user_id, user_item_matrix, predicted_ratings_df, movies_df, top_n=5):
    """Generates Top-N recommendations for a specific user."""
    
    if user_id not in predicted_ratings_df.index:
        return "User not found in the training data (Cold Start scenario)."

    # Get the user's predicted ratings for all movies
    user_predictions = predicted_ratings_df.loc[user_id]
    
    # Find out which movies the user has *already* watched/rated
    # We drop NA values to get only the items they actually interacted with
    movies_already_watched = user_item_matrix.loc[user_id].dropna().index
    
    # Filter out movies the user has already watched
    unseen_predictions = user_predictions.drop(movies_already_watched, errors='ignore')
    
    # Sort the unseen movies by predicted rating in descending order
    top_movie_ids = unseen_predictions.sort_values(ascending=False).head(top_n).index
    
    # Get the actual movie titles and genres
    recommended_movies = movies_df[movies_df['movie_id'].isin(top_movie_ids)][['title', 'genre']]
    
    return recommended_movies

def main():
    print("1. Generating Data...")
    users_df, movies_df, ratings_df = generate_streaming_data()
    print(f"Generated {len(ratings_df)} ratings across {len(users_df)} users and {len(movies_df)} movies.\n")
    
    print("2. Building Collaborative Filtering Model (SVD)...")
    user_item_matrix, predicted_ratings = build_recommendation_model(ratings_df)
    
    # Let's recommend movies for a random user, e.g., User ID 10
    target_user = 10
    print(f"\n3. Generating Recommendations for User {target_user}...")
    
    # What did they like? (Let's show highly rated movies they already saw)
    user_history = ratings_df[(ratings_df['user_id'] == target_user) & (ratings_df['rating'] >= 4)]
    watched_details = user_history.merge(movies_df, on='movie_id')
    print(f"\nUser {target_user}'s Highly Rated History:")
    print(watched_details[['title', 'genre', 'rating']].head(3).to_string(index=False))
    
    # What do we recommend?
    recs = get_recommendations(target_user, user_item_matrix, predicted_ratings, movies_df, top_n=5)
    
    print(f"\nTop 5 Recommended Movies for User {target_user}:")
    print(recs.to_string(index=False))

if __name__ == "__main__":
    main()


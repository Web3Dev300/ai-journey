import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD

GENRES = ['Action', 'Comedy', 'Drama', 'Sci-Fi', 'Horror']
AGE_BINS = [17, 29, 44, 65]
AGE_LABELS = ['18-29', '30-44', '45-64']


def generate_streaming_data(num_users=300, num_movies=60):
    """Generates synthetic users, movies and ratings in which every user has a real genre taste."""
    rng = np.random.default_rng(42)

    # 1. Users (Demographics). Taste depends partly on the age group.
    users = pd.DataFrame({
        'user_id': range(1, num_users + 1),
        'age': rng.integers(18, 65, num_users),
        'country': rng.choice(['US', 'UK', 'CA', 'AU'], num_users)
    })
    users['age_group'] = pd.cut(users['age'], bins=AGE_BINS, labels=AGE_LABELS)
    likely_favourites = {'18-29': ['Action', 'Sci-Fi', 'Horror'],
                         '30-44': ['Comedy', 'Action', 'Drama'],
                         '45-64': ['Drama', 'Comedy']}
    favourite_genre = {row.user_id: rng.choice(likely_favourites[row.age_group]) for row in users.itertuples()}

    # 2. Items (Content)
    movies = pd.DataFrame({
        'movie_id': range(1, num_movies + 1),
        'title': [f"Movie_{i}" for i in range(1, num_movies + 1)],
        'genre': rng.choice(GENRES, num_movies)
    })
    movie_genre = movies.set_index('movie_id')['genre']

    # 3. Interactions (Ratings 1-5): high for the favourite genre, lower for the rest, plus noise.
    #    Each user rates only some of the movies, so the matrix has gaps.
    interactions = []
    for user in users['user_id']:
        rated_movies = rng.choice(movies['movie_id'], rng.integers(15, 31), replace=False)
        for movie in rated_movies:
            liked = movie_genre[movie] == favourite_genre[user]
            rating = (4.5 if liked else 2.5) + rng.normal(0, 0.7)
            interactions.append({'user_id': user, 'movie_id': movie, 'rating': int(np.clip(round(rating), 1, 5))})

    ratings_df = pd.DataFrame(interactions)
    return users, movies, ratings_df


def build_recommendation_model(ratings_df, n_components=5, n_iterations=10):
    """Builds a Collaborative Filtering model using Matrix Factorization (SVD)."""

    # 1. Create the User-Item Matrix (rows: users, columns: movies, values: ratings)
    user_item_matrix = ratings_df.pivot(index='user_id', columns='movie_id', values='rating')

    # 2. Centre each user's ratings on their own average, so that 0 means "no opinion".
    #    (Filling the raw matrix with 0 would tell the model that unseen movies are hated.)
    user_means = user_item_matrix.mean(axis=1)
    centred = user_item_matrix.sub(user_means, axis=0)
    is_rated = centred.notna().to_numpy()
    known_values = centred.fillna(0).to_numpy()

    # 3. Apply Matrix Factorization (Truncated SVD) to find latent (hidden) taste features.
    #    Start with 0 in the empty cells, then replace those guesses with the model's own
    #    predictions and refit a few times.
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    filled = known_values.copy()
    for _ in range(n_iterations):
        latent_matrix = svd.fit_transform(filled)
        reconstructed_matrix = np.dot(latent_matrix, svd.components_)
        filled = np.where(is_rated, known_values, reconstructed_matrix)

    # 4. Add the averages back to get predicted ratings on the 1-5 scale for ALL user-movie pairs
    predicted_ratings_df = pd.DataFrame(
        reconstructed_matrix + user_means.to_numpy()[:, None],
        index=user_item_matrix.index,
        columns=user_item_matrix.columns
    ).clip(1, 5)

    return user_item_matrix, predicted_ratings_df


def rmse(predicted, actual):
    """Root mean squared error: the typical size of a prediction error, in rating points."""
    return float(np.sqrt(np.mean((np.asarray(predicted) - np.asarray(actual)) ** 2)))


def evaluate(ratings_df):
    """Hides 20% of the ratings, trains on the rest and measures the error on the hidden ones."""
    test = ratings_df.sample(frac=0.2, random_state=42)
    train = ratings_df.drop(test.index)
    user_item_matrix, predicted_ratings_df = build_recommendation_model(train)

    model_predictions = [predicted_ratings_df.at[row.user_id, row.movie_id] for row in test.itertuples()]
    # Baseline: ignore the movie and always predict the user's own average rating
    baseline_predictions = test['user_id'].map(user_item_matrix.mean(axis=1))
    return rmse(model_predictions, test['rating']), rmse(baseline_predictions, test['rating'])


def get_recommendations(user_id, user_item_matrix, predicted_ratings_df, movies_df, top_n=5):
    """Generates Top-N recommendations for a known user, best first."""
    # Movies the user has already rated are not recommended again
    already_rated = user_item_matrix.loc[user_id].dropna().index
    unseen_predictions = predicted_ratings_df.loc[user_id].drop(already_rated)

    # Sort the unseen movies by predicted rating, highest first, and keep that order
    top = unseen_predictions.sort_values(ascending=False).head(top_n).rename('predicted_rating').reset_index()
    return top.merge(movies_df, on='movie_id')[['title', 'genre', 'predicted_rating']]


def recommend_for_new_user(age, users_df, ratings_df, movies_df, top_n=5):
    """Cold start: a new user has no ratings yet, so use demographics instead.
    Recommends the movies rated highest by existing users in the same age group."""
    age_group = pd.cut([age], bins=AGE_BINS, labels=AGE_LABELS)[0]
    similar_users = users_df.loc[users_df['age_group'] == age_group, 'user_id']
    group_ratings = ratings_df[ratings_df['user_id'].isin(similar_users)]
    stats = group_ratings.groupby('movie_id')['rating'].agg(['mean', 'count'])
    top = stats[stats['count'] >= 10].sort_values('mean', ascending=False).head(top_n)  # Skip rarely rated movies
    top = top.rename(columns={'mean': 'avg_rating_in_age_group'}).reset_index()
    return top.merge(movies_df, on='movie_id')[['title', 'genre', 'avg_rating_in_age_group']]


def main():
    print("1. Generating Data...")
    users_df, movies_df, ratings_df = generate_streaming_data()
    print(f"Generated {len(ratings_df)} ratings across {len(users_df)} users and {len(movies_df)} movies.\n")

    print("2. Evaluating on ratings the model has not seen...")
    model_rmse, baseline_rmse = evaluate(ratings_df)
    print(f"RMSE of the model: {model_rmse:.2f}")
    print(f"RMSE of the baseline (each user's average rating): {baseline_rmse:.2f}\n")

    print("3. Building Collaborative Filtering Model (SVD) on all ratings...")
    user_item_matrix, predicted_ratings = build_recommendation_model(ratings_df)

    # Let's recommend movies for one user, e.g., User ID 10
    target_user = 10

    # What did they like? (Let's show highly rated movies they already saw)
    user_history = ratings_df[(ratings_df['user_id'] == target_user) & (ratings_df['rating'] >= 4)]
    watched_details = user_history.merge(movies_df, on='movie_id')
    print(f"\nUser {target_user}'s Highly Rated History:")
    print(watched_details[['title', 'genre', 'rating']].head(5).to_string(index=False))

    # What do we recommend?
    recs = get_recommendations(target_user, user_item_matrix, predicted_ratings, movies_df, top_n=5)
    print(f"\nTop 5 Recommended Movies for User {target_user} (best first):")
    print(recs.round(2).to_string(index=False))

    print("\n4. Cold start: a new 50-year-old user with no ratings yet")
    print(recommend_for_new_user(50, users_df, ratings_df, movies_df).round(2).to_string(index=False))


if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

# Create synthetic data
np.random.seed(42)
n_samples = 1000

locations = ['Downtown', 'Suburb', 'Rural']
amenities_levels = ['Basic', 'Standard', 'Premium']

data = {
    'location': np.random.choice(locations, n_samples),
    'size': np.random.normal(1500, 500, n_samples).astype(int),
    'amenities': np.random.choice(amenities_levels, n_samples),
    'year_built': np.random.randint(1950, 2024, n_samples),
    'num_bedrooms': np.random.randint(1, 6, n_samples),
    'num_bathrooms': np.random.randint(1, 4, n_samples)
}
df = pd.DataFrame(data)
df['size'] = np.clip(df['size'], 500, 5000)

base_price = 100000
location_multiplier = {'Downtown': 2.0, 'Suburb': 1.2, 'Rural': 0.8}
amenities_multiplier = {'Basic': 1.0, 'Standard': 1.1, 'Premium': 1.3}

prices = (
    base_price
    + (df['size'] * 150)
    + (df['num_bedrooms'] * 20000)
    + (df['num_bathrooms'] * 15000)
    + ((2024 - df['year_built']) * -500)
)

df['price'] = prices * df['location'].map(location_multiplier) * df['amenities'].map(amenities_multiplier)
df['price'] += np.random.normal(0, 20000, n_samples)

X = df.drop('price', axis=1)
y = df['price']

# Define preprocessing using column indices
categorical_features = [0, 2] # location, amenities
numerical_features = [1, 3, 4, 5] # size, year_built, num_bedrooms, num_bathrooms

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train the model using numpy array so that it expects a 2D array during prediction
model.fit(X.values, y.values)

model_path = os.path.join(os.path.dirname(__file__), 'real_estate_model.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(model, f)

print(f"Model successfully saved to {model_path}")

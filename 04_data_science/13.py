import os

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# Load the dataset
csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'customer_data.csv')
data = pd.read_csv(csv_path)

# K-Means needs numbers: cluster on the numeric columns (demographics + purchasing behaviour)
features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
data[features] = data[features].fillna(data[features].median())  # Fill missing values with the median

# Scale the features so that no column dominates the distance calculation
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data[features])

# Choosing the number of clusters: elbow method (SSE) and silhouette score (higher is better)
k_values = list(range(2, 7))
sse, silhouettes = [], []
for k in k_values:
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = kmeans.fit_predict(data_scaled)
    sse.append(kmeans.inertia_)
    silhouettes.append(silhouette_score(data_scaled, labels))
    print(f"k={k}: SSE={sse[-1]:.1f}, silhouette={silhouettes[-1]:.3f}")

plt.plot(k_values, sse, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('SSE')
plt.show()

# Use the k with the best silhouette score instead of a hard-coded guess
best_k = k_values[silhouettes.index(max(silhouettes))]
print(f"Best k by silhouette score: {best_k}")

# Applying K-Means Clustering
kmeans = KMeans(n_clusters=best_k, n_init=10, random_state=42)
data['Cluster'] = kmeans.fit_predict(data_scaled)

# Describe each cluster: size, average of each feature, and gender mix (a categorical demographic)
profile = data.groupby('Cluster')[features].mean().round(1)
profile['Customers'] = data.groupby('Cluster').size()
profile['Male share'] = data.groupby('Cluster')['Gender'].apply(lambda g: (g == 'Male').mean()).round(2)
print(profile)

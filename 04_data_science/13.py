import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

import os

# Load the dataset
csv_path = os.path.join(os.path.dirname(__file__), 'customer_data.csv')
data = pd.read_csv(csv_path)

# Simple preprocessing
data.fillna(data.mean(numeric_only=True), inplace=True) # Fill missing values with mean
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data[["Age", "Annual Income (k$)", "Spending Score (1-100)"]]) # Scale the features

# Choosing the number of clusters with the elbow method
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data_scaled)
    sse.append(kmeans.inertia_)
plt.plot(range(1, 11), sse)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('SSE')
plt.show()

# Applying K-Means Clustering
kmeans = KMeans(n_clusters=5, random_state=42) #Assuming 5 clusters based on the elbow method
clusters = kmeans.fit_predict(data_scaled)

# Add cluste information back to the original DataFrame
data['Cluster'] = clusters

# Display characteristics of each cluster
print(data.groupby('Cluster').mean(numeric_only=True))
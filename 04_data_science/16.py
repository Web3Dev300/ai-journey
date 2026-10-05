import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

import os

# Load dataset
csv_path = os.path.join(os.path.dirname(__file__), 'transactions.csv')
data = pd.read_csv(csv_path)
data['transaction_time'] = pd.to_datetime(data['transaction_time'])

# Feature Engineering
data['hour_of_day'] = data['transaction_time'].dt.hour

# Data normalization
scaler = StandardScaler()
data[['scaled_amount', 'scaled_hour_of_day']] = scaler.fit_transform(data[['amount', 'hour_of_day']])

# Anomaly Detection using Isolation Forest
model = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
data['anomaly'] = model.fit_predict(data[['scaled_amount', 'scaled_hour_of_day']])

# Filtering anomalies
anomalies = data[data['anomaly'] == -1]
print(anomalies)
print(f"Number of anomalies detected: {len(anomalies)}")
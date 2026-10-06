import os

import pandas as pd
from sklearn.ensemble import IsolationForest

REVIEW_CAPACITY = 20  # How many transactions the fraud team can check by hand

# Load dataset
csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'transactions.csv')
data = pd.read_csv(csv_path)
data['transaction_time'] = pd.to_datetime(data['transaction_time'])

# Feature Engineering
data['hour_of_day'] = data['transaction_time'].dt.hour
features = ['amount', 'hour_of_day']  # Isolation Forest is tree-based, so no scaling is needed

# Anomaly Detection using Isolation Forest (it never sees the 'is_fraud' column)
model = IsolationForest(n_estimators=200, random_state=42)
model.fit(data[features])

# score_samples: the lower the score, the more unusual the transaction
data['anomaly_score'] = model.score_samples(data[features])

# Send the most unusual transactions to the fraud team, most suspicious first
flagged = data.sort_values('anomaly_score').head(REVIEW_CAPACITY)
print(flagged[['transaction_id', 'transaction_time', 'amount', 'anomaly_score']].to_string(
    index=False, formatters={'anomaly_score': '{:.3f}'.format}))

# Evaluation: compare the flags with the fraud cases that were confirmed later
caught = flagged['is_fraud'].sum()
total_fraud = data['is_fraud'].sum()
print(f"\nFlagged for review: {len(flagged)} of {len(data)} transactions")
print(f"Precision: {caught / len(flagged):.0%} of the flagged transactions were real fraud")
print(f"Recall: {caught} of {total_fraud} real fraud cases were caught")

# Baseline: a simple rule (review the largest amounts). The model must beat this.
rule_caught = data.nlargest(REVIEW_CAPACITY, 'amount')['is_fraud'].sum()
print(f"Baseline rule (review the {REVIEW_CAPACITY} largest amounts): {rule_caught} of {total_fraud} caught")

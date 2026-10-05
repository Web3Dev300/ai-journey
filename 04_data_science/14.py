import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import os
# Load and prepare dataset
csv_path = os.path.join(os.path.dirname(__file__), 'customer_data.csv')
data = pd.read_csv(csv_path)
data['churn'] = data['churn'].map({'Yes': 1, 'No': 0}) # Encode target variable
data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 0}) # Encode Gender to numeric
data.fillna(0, inplace=True) # Fill missing values with 0

# Feature engineering
data['service_duration'] = data['last_service_date'] - data['first_service_date']

# Split dataset into features and target variable
X = data.drop('churn', axis = 1)
y = data['churn']
X_train,X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train logistic regression model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate model performance
print(classification_report(y_test, y_pred))


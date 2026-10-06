import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

def generate_synthetic_data(num_records=10000):
    """Generates synthetic hourly sensor data and failure events."""
    np.random.seed(42)
    
    # Create a time series index
    date_rng = pd.date_range(start='1/1/2023', periods=num_records, freq='h')
    df = pd.DataFrame(date_rng, columns=['timestamp'])
    
    # Simulate sensor data (normal operation)
    df['vibration'] = np.random.normal(loc=10, scale=2, size=num_records)
    df['temperature'] = np.random.normal(loc=70, scale=5, size=num_records)
    
    # Introduce failures
    df['failure'] = 0
    failure_indices = np.random.choice(num_records, size=50, replace=False)
    
    for idx in failure_indices:
        if idx > 24:
            df.loc[idx, 'failure'] = 1
            # Add a degradation signature leading up to the failure (last 24 hours)
            df.loc[idx-24:idx, 'vibration'] += np.linspace(0, 8, 25)
            df.loc[idx-24:idx, 'temperature'] += np.linspace(0, 15, 25)
            
    return df

def engineer_features(df):
    """Creates rolling window features and the predictive label."""
    # 1. Feature Engineering: Rolling Statistics (3h and 12h windows)
    df['vib_mean_3h'] = df['vibration'].rolling(window=3).mean()
    df['vib_std_12h'] = df['vibration'].rolling(window=12).std()
    
    df['temp_mean_3h'] = df['temperature'].rolling(window=3).mean()
    df['temp_max_12h'] = df['temperature'].rolling(window=12).max()
    
    # Drop rows with NaN values created by rolling windows
    df = df.dropna()
    
    # 2. Label Generation: Predict if a failure will happen in the *next 24 hours*
    # We shift the failure column backwards. 
    # If any of the next 24 hours has a failure, label the current hour as 1.
    df['failure_in_next_24h'] = df['failure'].rolling(window=24, min_periods=1).max().shift(-24)
    
    # Drop the very last rows where we don't have 24 hours of future visibility
    df = df.dropna()
    
    return df

def main():
    print("1. Generating Data...")
    raw_data = generate_synthetic_data()
    
    print("2. Engineering Features and Labels...")
    processed_data = engineer_features(raw_data.copy())
    
    # 3. Model Training Preparation
    # Drop the timestamp and the actual 'failure' event (we can't know it in real-time)
    X = processed_data.drop(columns=['timestamp', 'failure', 'failure_in_next_24h'])
    y = processed_data['failure_in_next_24h']
    
    # Note: For time series, we ideally use chronological splitting. 
    # For simplicity in this snippet, we use train_test_split without shuffling.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    print(f"Training set size: {X_train.shape[0]} records")
    
    # 4. Train the Model
    print("3. Training Random Forest Model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    
    # 5. Evaluation
    print("\n4. Evaluation on Test Set:")
    predictions = model.predict(X_test)
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))
    
    # Display Feature Importance
    feature_importances = pd.Series(model.feature_importances_, index=X.columns)
    print("\nTop Predictive Features:")
    print(feature_importances.sort_values(ascending=False).head(3))

if __name__ == "__main__":
    main()


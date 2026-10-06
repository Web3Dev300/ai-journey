import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, classification_report
from sklearn.model_selection import train_test_split


def generate_churn_data(n_customers=5000):
    """Generates synthetic subscription data: one row per customer, as seen on a snapshot date."""
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        'tenure_months': rng.integers(1, 61, n_customers),
        'monthly_spend': rng.normal(50, 15, n_customers).clip(10, 120).round(2),
        'support_tickets_90d': rng.poisson(1.0, n_customers),
        'days_since_last_login': rng.exponential(10, n_customers).round().clip(0, 90),
        'is_monthly_contract': rng.integers(0, 2, n_customers),
    })

    # New, inactive, unhappy customers on monthly contracts are more likely to leave
    risk = (-2.6
            - 0.05 * df['tenure_months']
            + 0.60 * df['support_tickets_90d']
            + 0.08 * df['days_since_last_login']
            + 1.00 * df['is_monthly_contract'])
    churn_probability = 1 / (1 + np.exp(-risk))

    # Target: did the customer cancel in the 30 days AFTER the snapshot date?
    df['churned_next_30d'] = (rng.random(n_customers) < churn_probability).astype(int)
    return df


def main():
    print("1. Generating Data...")
    data = generate_churn_data()
    X = data.drop(columns='churned_next_30d')
    y = data['churned_next_30d']
    print(f"Customers: {len(data)}, churn rate: {y.mean():.1%}")

    # Split BEFORE doing anything else; stratify keeps the same churn rate in both parts
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Baseline: always predict "stays". High accuracy, but it finds no churners at all.
    baseline = DummyClassifier(strategy='most_frequent').fit(X_train, y_train)
    print(f"\n2. Baseline accuracy (always predict 'stays'): {baseline.score(X_test, y_test):.2f}")

    # Handle imbalance with class weights: mistakes on the rare class (churners) cost more
    print("\n3. Training Random Forest Model...")
    model = RandomForestClassifier(n_estimators=200, min_samples_leaf=5, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)

    # Evaluate with precision and recall on the churn class, not accuracy
    churn_risk = model.predict_proba(X_test)[:, 1]
    print("\n4. Evaluation on Test Set:")
    print(classification_report(y_test, model.predict(X_test), target_names=['stays', 'churns']))
    print(f"PR-AUC: {average_precision_score(y_test, churn_risk):.2f} (a random model scores about {y_test.mean():.2f})")

    # Business view: if the retention team can only call 10% of customers, who should it call?
    riskiest = y_test[churn_risk >= np.quantile(churn_risk, 0.9)]
    print(f"Churn rate among the riskiest 10%: {riskiest.mean():.0%} (all customers: {y_test.mean():.0%})")

    feature_importances = pd.Series(model.feature_importances_, index=X.columns)
    print("\nTop Predictive Features:")
    print(feature_importances.sort_values(ascending=False).head(3).round(2).to_string())


if __name__ == "__main__":
    main()

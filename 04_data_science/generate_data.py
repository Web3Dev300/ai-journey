import csv
import os
import random
from datetime import datetime, timedelta

random.seed(42)

N_NORMAL = 970       # Everyday purchases: daytime, mostly small amounts
N_LARGE_LEGIT = 20   # Genuine big purchases (laptop, holiday): daytime, large amounts
N_FRAUD = 10         # Fraud: large amounts in the middle of the night

base_time = datetime(2023, 10, 1)
transactions = []


def add_transaction(hour, amount, is_fraud):
    dt = base_time + timedelta(days=random.randint(0, 30), hours=hour, minutes=random.randint(0, 59))
    transactions.append({
        'transaction_time': dt.strftime('%Y-%m-%d %H:%M:%S'),
        'amount': round(amount, 2),
        'is_fraud': is_fraud,
    })


for _ in range(N_NORMAL):
    hour = max(0, min(23, int(random.gauss(15, 4))))
    amount = max(5, min(2000, random.lognormvariate(4.0, 1.0)))
    add_transaction(hour, amount, 0)

for _ in range(N_LARGE_LEGIT):
    add_transaction(random.randint(9, 20), random.uniform(1500, 6000), 0)

for _ in range(N_FRAUD):
    add_transaction(random.choice([0, 1, 2, 3, 4]), random.uniform(800, 5000), 1)

# Shuffle first, then number the rows, so the ID reveals nothing about fraud
random.shuffle(transactions)
for i, transaction in enumerate(transactions, start=1):
    transaction['transaction_id'] = f"TXN{i:05d}"

# 'is_fraud' is the answer key. In real life it arrives weeks later (confirmed chargebacks).
# 16.py never trains on it; it only uses it to measure how good its flags are.
csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'transactions.csv')
with open(csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['transaction_id', 'transaction_time', 'amount', 'is_fraud'])
    writer.writeheader()
    writer.writerows(transactions)

print(f"{csv_path} generated: {len(transactions)} transactions, {N_FRAUD} fraudulent.")

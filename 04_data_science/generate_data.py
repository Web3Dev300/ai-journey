import csv
import random
from datetime import datetime, timedelta

random.seed(42)

# Normal transactions: hours between 8 and 22, amounts mostly low
n_normal = 990
# Anomalous transactions: hours between 1 and 4, amounts high
n_anomalous = 10

base_time = datetime(2023, 10, 1)

transactions = []

for i in range(n_normal):
    # random day in 30 days
    days = random.randint(0, 30)
    # random hour for normal
    hour = int(random.gauss(15, 4))
    hour = max(0, min(23, hour))
    minute = random.randint(0, 59)
    dt = base_time + timedelta(days=days, hours=hour, minutes=minute)
    
    # Amount (log normal like distribution using random.lognormvariate or just some exponential/gamma)
    amount = random.lognormvariate(4.0, 1.0)
    amount = max(5, min(2000, amount))
    
    transactions.append({
        'transaction_id': f"TXN{str(len(transactions)+1).zfill(5)}",
        'transaction_time': dt.strftime('%Y-%m-%d %H:%M:%S'),
        'amount': round(amount, 2),
        'is_anomaly_label': 0  # not strictly needed by 16.py but good for reference
    })

for i in range(n_anomalous):
    days = random.randint(0, 30)
    hour = random.choice([1, 2, 3, 4])
    minute = random.randint(0, 59)
    dt = base_time + timedelta(days=days, hours=hour, minutes=minute)
    
    # anomalous amounts
    amount = random.uniform(10000, 50000)
    
    transactions.append({
        'transaction_id': f"TXN{str(len(transactions)+1).zfill(5)}",
        'transaction_time': dt.strftime('%Y-%m-%d %H:%M:%S'),
        'amount': round(amount, 2),
        'is_anomaly_label': 1
    })

random.shuffle(transactions)

# We drop is_anomaly_label when saving because 16.py doesn't use it, but wait, it doesn't hurt.
# Actually, let's keep it simple and match exactly what's expected plus id
csv_path = '/run/media/shubh/New Volume/ai-journey/04_data_science/transactions.csv'
with open(csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['transaction_id', 'transaction_time', 'amount'])
    writer.writeheader()
    for t in transactions:
        writer.writerow({
            'transaction_id': t['transaction_id'],
            'transaction_time': t['transaction_time'],
            'amount': t['amount']
        })

print(f"{csv_path} generated.")

import re

with open("DATA_SCIENCE_ANSWERS.md", "r") as f:
    content = f.read()

# Extract everything between "## 1. Handling Missing Data (30% missing)" and "---" (the first divider)
pattern = r"(## 1\. Handling Missing Data \(30% missing\).*?)(?=\n---\n)"
match = re.search(pattern, content, flags=re.DOTALL)

if match:
    old_q1 = match.group(1)
    
    new_q1 = """## 1. Handling Missing Data (30% missing)
**Question:** Imagine you're given a dataset where 30% of the data for a key predictive variable is missing. The variable is crucial for your predictive model. How would you handle this situation to ensure the integrity and performance of your model?

**Answer:** 
When 30% of a crucial variable is missing, simply deleting the rows isn't ideal because you lose a significant portion of your dataset. Combining standard practices with advanced techniques, my step-by-step approach would be:
1. **Identify & Analyze the Pattern:** First, I'd locate the gaps (e.g., using `data.isnull().sum()`) and analyze *why* it's missing. Is it missing completely at random, or is there a pattern? 
2. **Choose & Implement Imputation:** While simple methods like Mean/Median (using `SimpleImputer`) are fast, substituting 30% of a crucial variable with an average can severely distort its distribution. Therefore, I prefer model-based imputation like **KNN (K-Nearest Neighbors)** or **MICE**, which use other variables to make highly accurate estimates.
3. **Add a Missing Indicator:** I would create a new binary column (e.g., `is_missing`) to flag the imputed rows, as the *fact* that data was missing can sometimes be predictive.
4. **Evaluate the Impact (Crucial):** Finally, I would use visualizations (like histograms and box plots) to compare the variable's distribution before and after imputation. If the shape is heavily warped, the strategy must be reconsidered.

**Visual Diagram:**
```mermaid
flowchart TD
    A[Identify Missing Data] --> B[Analyze Pattern]
    B --> C{Choose Imputation}
    C -->|Simple (High Risk for 30%)| D[Mean / Median]
    C -->|Advanced (Recommended)| E[KNN / MICE]
    E --> F[Add 'is_missing' Indicator]
    F --> G[Visualize Impact: Histograms/Box Plots]
```

**Example:**
Imagine a housing dataset where 30% of "Square Footage" is missing. Instead of guessing the average (which would artificially cluster 30% of houses exactly in the middle), we use KNN to look at the house's "Number of Bedrooms" to estimate the size. Then, we plot a histogram to ensure our new square footage curve looks natural.

**Python Code Example:**
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer

# Sample data
df = pd.DataFrame({
    'Bedrooms': [3, 4, 2, 4, 3, 5, 2, 3, 4, 3],
    'SqFt': [1500, 2000, np.nan, 2100, np.nan, 3000, 1100, np.nan, 2200, 1600]
})

# 1. Add missing indicator
df['SqFt_missing'] = df['SqFt'].isnull().astype(int)

# 2. KNN Imputation
imputer = KNNImputer(n_neighbors=2)
imputed_data = imputer.fit_transform(df[['Bedrooms', 'SqFt']])
df['SqFt_Imputed'] = imputed_data[:, 1]

# 3. Evaluate Impact (Visualization concept)
# In practice: plt.hist(df['SqFt'].dropna(), alpha=0.5, label='Original')
#              plt.hist(df['SqFt_Imputed'], alpha=0.5, label='Imputed')
print(df[['Bedrooms', 'SqFt', 'SqFt_Imputed', 'SqFt_missing']])
```

**🔥 Pro Tip / Common Pitfall (Data Leakage):** Never fit an imputer (like KNN or Mean) on your entire dataset before splitting it. You must split your data into train/test first, and only `.fit()` the imputer on the training data. Applying it to the whole dataset leaks information from the test set into your training process, artificially inflating your model's performance."""
    
    content = content.replace(old_q1, new_q1)
    with open("DATA_SCIENCE_ANSWERS.md", "w") as f:
        f.write(content)
    print("Successfully merged YouTube concepts into Q1!")
else:
    print("Failed to find Q1 in the document.")

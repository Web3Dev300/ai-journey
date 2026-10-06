import re

with open("DATA_SCIENCE_ANSWERS.md", "r") as f:
    content = f.read()

pattern = r"(## 1\. Handling Missing Data \(30% missing\).*?)(?=\n---\n)"
match = re.search(pattern, content, flags=re.DOTALL)

if match:
    old_q1 = match.group(1)
    
    new_q1 = """## 1. Handling Missing Data (30% missing)
**Question:** Imagine you're given a dataset where 30% of the data for a key predictive variable is missing. The variable is crucial for your predictive model. How would you handle this situation to ensure the integrity and performance of your model?

**Answer:** 
Deleting 30% of your data leads to massive information loss. To ensure model integrity, I would follow these steps:

1. **Identify the Gaps:** Use `data.isnull().sum()` to pinpoint exactly where the missing values are.
2. **Analyze the Pattern:** Determine if the data is missing randomly or if there is an underlying pattern.
3. **Choose an Imputation Strategy:** 
   * **Basic:** Using `SimpleImputer` to fill with the mean/median is fast, but replacing 30% of a dataset with an average can severely distort the data.
   * **Advanced (Recommended):** I would use a model-based approach like **KNN (K-Nearest Neighbors)** to predict and fill the missing values based on relationships with other columns.
4. **Evaluate the Impact:** This is the most crucial step. I would use **histograms or box plots** to compare the variable's distribution before and after imputation to ensure the data's natural shape wasn't destroyed.

**Visual Diagram:**
```mermaid
flowchart TD
    A[Identify Missing Values] --> B[Analyze Pattern]
    B --> C{Imputation Method}
    C -->|Basic| D[Mean/Median]
    C -->|Advanced| E[KNN/MICE]
    D --> F[Visualize Distribution]
    E --> F
    F --> G[Check for Distortions]
```

**Example:**
If 30% of house sizes are missing, filling them all with the "average size" creates an unnatural spike in the middle of a histogram. Instead, using KNN to estimate size based on "number of bedrooms" keeps the data distribution looking realistic.

**Python Code Example:**
```python
import pandas as pd
from sklearn.impute import KNNImputer

# Assume 'df' is our dataframe
# 1. Identify missing data
# print(df.isnull().sum())

# 2. Impute missing values using KNN
imputer = KNNImputer(n_neighbors=5)
df['feature_imputed'] = imputer.fit_transform(df[['feature', 'other_feature']])[:, 0]

# 3. Add an indicator column
df['was_missing'] = df['feature'].isnull().astype(int)
```

**🔥 Pro Tip / Common Pitfall (Data Leakage):** 
Never fit an imputer on your entire dataset before splitting it. You must split your data into train/test first, and only `.fit()` the imputer on the training data to prevent future information from leaking into your model."""
    
    content = content.replace(old_q1, new_q1)
    with open("DATA_SCIENCE_ANSWERS.md", "w") as f:
        f.write(content)
    print("Successfully cleaned Q1!")
else:
    print("Failed to find Q1 in the document.")

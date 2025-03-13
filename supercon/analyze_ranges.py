import pandas as pd
import json

# Read the training data
df = pd.read_csv('train.csv')

# Calculate statistics for each feature
feature_stats = {}
for col in df.columns[:-1]:  # Exclude the target variable
    feature_stats[col] = {
        'min': float(df[col].min()),
        'max': float(df[col].max()),
        'mean': float(df[col].mean()),
        'std': float(df[col].std())
    }

# Save the statistics to a JSON file
with open('feature_stats.json', 'w') as f:
    json.dump(feature_stats, f, indent=2)

print("Feature statistics have been saved to feature_stats.json") 
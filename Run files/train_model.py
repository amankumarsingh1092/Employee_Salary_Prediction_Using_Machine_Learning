
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib

# Load dataset
data = pd.read_csv("C:/Users/AMAN KUMAR  SINGH/Downloads/adult/adult 3.csv")

# Rename column if needed
if 'salary' in data.columns:
    data.rename(columns={"salary": "income"}, inplace=True)

# Add 'experience' if not present
if 'experience' not in data.columns:
    data['experience'] = 5

# Drop missing values
data.dropna(inplace=True)

# Define features
features = ['age', 'education', 'occupation', 'hours-per-week', 'experience']
target = 'income'

X = data[features]
y = data[target]

# Save feature names
joblib.dump(features, "C:/Users/AMAN KUMAR  SINGH/Downloads/adult/feature_columns.pkl")

# Preprocessing
numeric_features = ['age', 'hours-per-week', 'experience']
categorical_features = ['education', 'occupation']

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# Pipeline
clf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Train
clf.fit(X, y)

# Save model
joblib.dump(clf, "C:/Users/AMAN KUMAR  SINGH/Downloads/adult/income_classifier.pkl")

print(" Model trained and saved as 'income_classifier.pkl'")

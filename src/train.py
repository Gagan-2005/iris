# train.py
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import joblib
import os

# Load dataset
iris = load_iris(as_frame=True)
df = iris.frame
df["class"] = df["target"].map(dict(enumerate(iris.target_names)))

# Features and labels
X = df[iris.feature_names]
y = df["class"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Train KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Save models
os.makedirs("model", exist_ok=True)
joblib.dump(rf, "model/rf_model.pkl")
joblib.dump(knn, "model/knn_model.pkl")

# Print accuracies
print(f"[Random Forest] Training Accuracy: {rf.score(X_train, y_train):.4f}")
print(f"[Random Forest] Testing Accuracy: {rf.score(X_test, y_test):.4f}")
print(f"[KNN] Training Accuracy: {knn.score(X_train, y_train):.4f}")
print(f"[KNN] Testing Accuracy: {knn.score(X_test, y_test):.4f}")

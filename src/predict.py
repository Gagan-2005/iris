# predict.py
import joblib
import numpy as np
import os

# Choose model: 'rf' or 'knn'
model_choice = 'knn'  # Change to 'rf' if you want Random Forest

model_path = f"model/{model_choice}_model.pkl"
model = joblib.load(model_path)

# Sample input
sample = np.array([[5.1, 3.5, 1.4, 0.2]])

# Predict
prediction = model.predict(sample)
print(f"Predicted class using {model_choice.upper()} model:", prediction[0])

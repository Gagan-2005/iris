# predict.py
import joblib
import numpy as np

# Load model
model = joblib.load("iris_model.pkl")

# Example input: [sepal_length, sepal_width, petal_length, petal_width]
sample = np.array([[5.1, 3.5, 1.4, 0.2]])

# Predict
prediction = model.predict(sample)
print("Predicted class:", prediction[0])

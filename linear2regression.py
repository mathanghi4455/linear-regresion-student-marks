
import numpy as np
from sklearn.linear_model import LinearRegression

# Dataset (hours vs marks)
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([35, 50, 65, 70, 85])

# Create model
model = LinearRegression()

# Train model
model.fit(X, y)

# Predict
predicted = model.predict([[6]])

# Output
print("Predicted marks for 6 hours:", predicted[0])
# Step 1: Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Step 2: Creating a simple dataset
# Let's say we have data of house sizes (in sq.ft) and their corresponding prices (in $)
house_size = np.array([500, 1000, 1500, 2000, 2500, 3000, 3500, 4000])
house_price = np.array([150000, 300000, 450000, 600000, 750000, 900000, 1050000, 1200000])

# Step 3: Reshape the data for scikit-learn (it expects 2D arrays for features)
X = house_size.reshape(-1, 1)  # Reshape into a column (2D array)
y = house_price  # No need to reshape target variable

# Step 4: Splitting the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Creating the Linear Regression model and training it
model = LinearRegression()
model.fit(X_train, y_train)

# Step 6: Making predictions
y_pred = model.predict(X_test)

# Step 7: Visualizing the results
plt.scatter(X_test, y_test, color='blue', label='Actual Prices')
plt.plot(X_test, y_pred, color='red', label='Predicted Line')
plt.xlabel('House Size (sq.ft)')
plt.ylabel('House Price ($)')
plt.legend()
plt.show()

# Optional: Check the accuracy of the model
accuracy = model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

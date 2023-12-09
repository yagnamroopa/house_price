# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.datasets import fetch_california_housing

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# Load the California housing dataset
california_housing = fetch_california_housing()
X = california_housing.data
y = california_housing.target
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Train a Linear Regression Model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
linear_predictions = linear_model.predict(X_test)
# Evaluate the Linear Regression model
linear_mse = mean_squared_error(y_test, linear_predictions)
print("Linear Regression:")
print(f"Mean Squared Error: {linear_mse}")
# Train a Neural Network using TensorFlow
model = Sequential([
    Dense(16, input_shape=(X_train.shape[1],), activation='relu'),
    Dense(8, activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')
# Evaluate the Neural Network model
nn_predictions = model.predict(X_test)
nn_mse = mean_squared_error(y_test, nn_predictions)
print("\nNeural Network:")
print(f"Mean Squared Error: {nn_mse}")

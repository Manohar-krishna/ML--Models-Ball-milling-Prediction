import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# Define the data from the table
data = {
    'ball_size': [10, 10, 10, 10, 10, 10, 5, 5, 5],
    'rpm': [600, 600, 600, 400, 400, 400, 400, 400, 400],
    'time': [24, 48, 72, 24, 48, 72, 24, 48, 72],
    'sphericity': [0.79464743, 0.81355519, 0.82061402, 0.76501041, 0.79116524, 0.72118172, 0.84585709, 0.87625333, 0.90514226]
}
# Convert dictionary values to numpy arrays
X = np.column_stack((data['ball_size'], data['rpm'], data['time']))
y = np.array(data['sphericity'])
# Scale the features
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=42
)
# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)
# Make predictions on the test set
y_pred = model.predict(X_test)
# Calculate evaluation metrics
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
# Convert y_test and y_pred back to original scale for additional metrics
y_test_orig = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
y_pred_orig = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()
# Calculate additional metrics in original scale
rmse_orig = np.sqrt(mean_squared_error(y_test_orig, y_pred_orig))
# Print model coefficients
print("\nLinear Regression Model Coefficients:")
for i, feature in enumerate(['ball_size', 'rpm', 'time']):
    print(f"{feature}: {model.coef_[i]:.4f}")
print(f"Intercept: {model.intercept_:.4f}")
# Print model performance metrics
print("\nModel Performance:")
print(f"R² Score: {r2:.4f}")
print(f"RMSE (Standardized): {rmse:.4f}")
print(f"RMSE (Original Scale): {rmse_orig:.4f}")
# User input for prediction
print("\n--- Linear Regression Sphericity Prediction ---")
print("Enter the parameter values to predict sphericity:")
ball_size = float(input("Ball Size: "))
rpm = float(input("RPM: "))
time_val = float(input("Time: "))
# Prepare input data
input_data = np.array([[ball_size, rpm, time_val]])
# Scale the input data
input_scaled = scaler_X.transform(input_data)
# Make prediction
pred_scaled = model.predict(input_scaled)
# Inverse transform to get the original scale
prediction = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()[0]
# Display prediction
print(f"\nPredicted Sphericity: {prediction:.4f}")
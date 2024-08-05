import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import joblib

# Step 1: Load the dataset
url = "https://raw.githubusercontent.com/4GeeksAcademy/alternative-time-series-project/main/sales.csv"
sales_data = pd.read_csv(url)

# Convert the 'date' column to datetime
sales_data['date'] = pd.to_datetime(sales_data['date'])

# Set the 'date' column as the index
sales_data.set_index('date', inplace=True)

# Plot the time series
plt.figure(figsize=(12, 6))
plt.plot(sales_data['sales'], label='Sales')
plt.title('Sales Time Series')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.show()

# Step 2: Analyze the time series

# Check the frequency of the dataset
print(f"Frequency of the dataset: {sales_data.index.freq}")

# Perform ADF test to check for stationarity
result = adfuller(sales_data['sales'])
print('ADF Statistic:', result[0])
print('p-value:', result[1])

# Make the series stationary if necessary
if result[1] > 0.05:
    sales_data['sales_diff'] = sales_data['sales'].diff().dropna()
else:
    sales_data['sales_diff'] = sales_data['sales']

# Plot the differenced series
plt.figure(figsize=(12, 6))
plt.plot(sales_data['sales_diff'], label='Differenced Sales')
plt.title('Differenced Sales Time Series')
plt.xlabel('Date')
plt.ylabel('Sales Diff')
plt.legend()
plt.show()

# Step 3: Train an ARIMA model

# Split the data into training and test sets
train_size = int(len(sales_data) * 0.8)
train, test = sales_data['sales_diff'][:train_size], sales_data['sales_diff'][train_size:]

# Find the best parameterization of ARIMA
model = auto_arima(train.dropna(), seasonal=False, trace=True)
print(model.summary())

# Train the ARIMA model with the best parameters
arima_model = ARIMA(train.dropna(), order=model.order)
arima_model_fit = arima_model.fit()
print(arima_model_fit.summary())

# Step 4: Predict with the test set

# Make predictions
predictions = arima_model_fit.forecast(steps=len(test))
predictions = pd.Series(predictions, index=test.index)

# Plot predictions vs actual sales
plt.figure(figsize=(12, 6))
plt.plot(train, label='Train')
plt.plot(test, label='Test')
plt.plot(predictions, label='Predicted')
plt.legend()
plt.title('ARIMA Model Predictions vs Actual Sales')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.show()

# Measure performance using Mean Squared Error
mse = mean_squared_error(test, predictions)
print('Mean Squared Error:', mse)

# Step 5: Save the model

# Save the ARIMA model
joblib.dump(arima_model_fit, 'arima_model.pkl')

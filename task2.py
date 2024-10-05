# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# Fetching the data
stock = 'AAPL'  # You can replace 'AAPL' with any other stock symbol
df = yf.download(stock, start='2010-01-01', end='2023-10-01')

# Preparing the data
data = df['Close'].values  # We will use the 'Close' prices
data = data.reshape(-1, 1)

# Feature scaling using MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data)

# Creating training data
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

# Defining the time step window
def create_dataset(data, time_step=60):
    X, y = [], []
    for i in range(len(data)-time_step-1):
        X.append(data[i:(i+time_step), 0])
        y.append(data[i+time_step, 0])
    return np.array(X), np.array(y)

time_step = 60
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# Reshaping input for LSTM: (samples, time steps, features)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Building the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# Compiling the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Training the model
model.fit(X_train, y_train, batch_size=64, epochs=10)

# Predicting stock prices using the model
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Inverse scaling the predictions back to original scale
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

# Inverse scaling the y_train and y_test back to original scale
y_train = scaler.inverse_transform([y_train])
y_test = scaler.inverse_transform([y_test])

# Plotting the results
plt.figure(figsize=(12,6))
plt.plot(df.index[:train_size + time_step + 1], scaler.inverse_transform(scaled_data[:train_size + time_step + 1]), label='Actual Stock Price')
plt.plot(df.index[train_size + time_step + 1:], test_predict, label='Predicted Stock Price', color='red')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

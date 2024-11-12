
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

import matplotlib.pyplot as plt
# mbl
symbol = 'SCB'
df = pd.read_csv(f"{symbol}.csv")
df.sort_values(by=['Date'],ascending=False)
df = df.set_index('Date')
scaler = MinMaxScaler()
df['Open'] = df['Open'].astype(float)
df_scaled = scaler.fit_transform(df[['Open']])


def create_dataset(df_scaled,look_back=1):
    x_train=[]
    y_train=[]
    for i in range(len(df_scaled)-look_back):
        x_train.append(df_scaled[i:i + look_back])
        y_train.append(df_scaled[i + look_back])
    return x_train, y_train

sequence_length = 10
X, y = create_dataset(df_scaled, sequence_length)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]


model = Sequential()
model.add(LSTM(50, activation='sigmoid', return_sequences=True, input_shape=(sequence_length, 1)))
model.add(LSTM(50, activation='tanh'))

model.add(Dense(1))
model.compile(optimizer=Adam(learning_rate=0.01), loss='mse')

X_train = np.array(X_train).reshape((len(X_train), sequence_length, 1))
X_test = np.array(X_test).reshape((len(X_test), sequence_length, 1))

y_train = np.array(y_train).reshape((len(y_train), 1))
y_test = np.array(y_test).reshape((len(y_test), 1))
history = model.fit(X_train, y_train, epochs=150, batch_size=32, validation_data=(X_test, y_test), verbose=2)

train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)


# Inverse transform the predictions to the original scale
train_predictions = scaler.inverse_transform(train_predictions)
y_train_original = scaler.inverse_transform(y_train.reshape(-1, 1))

test_predictions = scaler.inverse_transform(test_predictions)
y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1))
model.save(f"{symbol}_model.h5")
plt.figure(figsize=(10, 6))

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# Plot the results
plt.figure(figsize=(12, 6))

# # Plot training data
# plt.plot(df.index[sequence_length:sequence_length+train_size], y_train_original, label='Actual Train Data')
# plt.plot(df.index[sequence_length:sequence_length+train_size], train_predictions, label='Train Predictions')
# # Plot testing data
# test_index = df.index[sequence_length + train_size:sequence_length + train_size + len(test_predictions)]
# plt.plot(test_index, y_test_original, label='Actual Test Data')
# plt.plot(test_index, test_predictions, label='Test Predictions')
#
# plt.legend()
# plt.show()


df = df.sort_index(ascending=False)

# Scale the most recent data
latest_data = df.head(sequence_length)
latest_scaled = scaler.transform(latest_data[['Open']])

# Reshape the scaled data
latest_scaled = np.array(latest_scaled).reshape((1, sequence_length, 1))

# Predict tomorrow's price
tomorrow_prediction = model.predict(latest_scaled)
tomorrow_prediction = scaler.inverse_transform(tomorrow_prediction)

print("Tomorrow's predicted price:", tomorrow_prediction[0])

# Define the number of days to consider for forecasting
n_days_to_consider = 200

# Use the first 100 days of data for forecasting
latest_data = df_scaled[:n_days_to_consider]

# Initialize temp_input with the latest data
temp_input = list(latest_data.reshape(-1))

lst_output = []

# Generating the forecast for the next 10 days
for i in range(10):
    if len(temp_input) > n_days_to_consider:
        x_input = np.array(temp_input[1:]).reshape(1, -1)
        x_input = x_input.reshape((1, n_days_to_consider, 1))
        yhat = model.predict(x_input, verbose=0)
        temp_input.extend(yhat[0].tolist())
        temp_input = temp_input[1:]
        lst_output.extend(yhat.tolist())
    else:
        x_input = np.array(temp_input).reshape((1, len(temp_input), 1))
        yhat = model.predict(x_input, verbose=0)
        temp_input.extend(yhat[0].tolist())
        lst_output.extend(yhat.tolist())

# Inverse transform the forecasted values
forecast_inv = scaler.inverse_transform(np.array(lst_output).reshape(-1, 1))

# Generate arrays for plotting
latest_dates = pd.date_range(start=df.index[0], periods=n_days_to_consider)  # Dates for the latest data
forecast_dates = pd.date_range(start=latest_dates[-1] + pd.Timedelta(days=1), periods=10)  # Dates for the forecast

# Plotting
plt.plot(df.index, df['Open'], label='Original Data')
plt.plot(latest_dates, scaler.inverse_transform(latest_data), label='Latest Data')
plt.plot(forecast_dates, forecast_inv, label='Forecasted Prices')
plt.xlabel('Date')
plt.ylabel('Open Price')
plt.title('Forecast for the Next 10 Days')
plt.legend()
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.show()


# Plotting
plt.plot(df.index, df['Open'], label='Original Data')
plt.plot(latest_dates, scaler.inverse_transform(latest_data), label='Latest Data')
plt.plot(forecast_dates, forecast_inv, label='Forecasted Prices')
plt.xlabel('Date')
plt.ylabel('Open Price')
plt.title('Forecast for the Next 10 Days')
plt.legend()
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability

# Expand the x-axis limits
plt.xlim(df.index[0], forecast_dates[-1])

plt.show()


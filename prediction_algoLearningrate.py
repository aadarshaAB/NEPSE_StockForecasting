import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import math
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error
# Function to calculate lookback window size dynamically
def calculate_lookback(dataset_length, window_percentage):
    return math.ceil(dataset_length * window_percentage)

# mbl
symbol = 'MBL'
df = pd.read_csv(f"{symbol}.csv")
data_open=df.filter(['Open'])
dataset=data_open.values
training_data_len=math.ceil(len(dataset)*.8)
test_dates=df.index[training_data_len:]

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)
print(scaled_data)

# Define lookback percentage
lookback_percentage = 0.2  # Adjust as needed

# Calculate the lookback window dynamically
initial_window_size = calculate_lookback(len(scaled_data), lookback_percentage)

#training data set
train_data=scaled_data[0:training_data_len,:]
#split data
x_train=[]
y_train=[]
for i in range(initial_window_size, len(train_data)):
    x_train.append(train_data[i-initial_window_size:i, 0])
    y_train.append(train_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

#convert the x_train and y_train numpy arrays
x_train,y_train=np.array(x_train),np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0], initial_window_size, 1))

#create the testing dataset
test_data = scaled_data[training_data_len - initial_window_size:, :]
x_test = []
y_test = dataset[training_data_len:, :]
for i in range(initial_window_size, len(test_data)):
    x_test.append(test_data[i-initial_window_size:i, 0])

x_test=np.array(x_test)
x_test=np.reshape(x_test, (x_test.shape[0],initial_window_size,1))


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.LSTM(100, return_sequences=True, input_shape=(initial_window_size, 1)))
model.add(tf.keras.layers.LSTM(100, return_sequences=False))
model.add(tf.keras.layers.Dense(25))
model.add(tf.keras.layers.Dense(1))

#train model
model.compile(optimizer=Adam(learning_rate=0.01), loss='mean_squared_error', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=32, epochs=50)

y_train_original=y_train.reshape(-1,1)
y_train_original=scaler.inverse_transform(y_train_original)

train_predictions=model.predict(x_train)
train_predictions=scaler.inverse_transform(train_predictions)

train_rmse=np.sqrt(mean_squared_error(y_train_original,train_predictions))
print("RMSE for Train data:",train_rmse)

predictions=model.predict(x_test)
predictions=scaler.inverse_transform(predictions)

model.save(f"{symbol}_model.h5")

# Select the last 'initial_window_size' days of data
last_80_days = dataset[-initial_window_size:]

# Scale the data
last_80_days_scaled = scaler.transform(last_80_days)

# Reshape the data for model prediction
X_test = []
X_test.append(last_80_days_scaled)
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Predict the next 10 days' stock prices
predicted_prices = []

for i in range(10):
    predicted_price = model.predict(X_test)
    predicted_prices.append(predicted_price[0][0])
    # Update X_test for the next prediction
    X_test = np.append(X_test[:, 1:, :], np.expand_dims(predicted_price, axis=1), axis=1)

# Invert the scaling
predicted_prices = np.array(predicted_prices).reshape(-1, 1)
predicted_prices = scaler.inverse_transform(predicted_prices)

y_test_orginal=y_test.reshape(-1,1)
test_rmse=np.sqrt(mean_squared_error(y_test,predictions))
print("RMSE for Train data:",test_rmse)


train=dataset
plt.figure(figsize=(16,8))
plt.title('Stock Market Prediction')
plt.plot(test_dates, y_test, label='Actual')
plt.plot(test_dates, predictions, label='Predicted')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.show()

# Print the predicted prices
print("Predicted Prices for Next 10 Days:", predicted_prices)

# Plot the predicted prices
future_dates=pd.date_range(start=test_dates[-1],periods=1)
plt.figure(figsize=(10, 6))
plt.plot(predicted_prices, label='Predicted Prices', color='red')
plt.title('Predicted Prices for Next 10 Days')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

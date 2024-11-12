import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import math

# import mpld3
# from mpld3 import plugins
# mbl
symbol = 'PRVU'
df = pd.read_csv(f"{symbol}.csv")
df=df.set_index('Date')
data_open=df.filter(['Open'])
dataset=data_open.values
training_data_len=math.ceil(len(dataset)*.8)
train_dates=df.index[10:training_data_len]
test_dates=df.index[training_data_len:]

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

#training data set
train_data=scaled_data[0:training_data_len,:]
#split data
x_train=[]
y_train=[]
initial_window_size=10
for i in range(initial_window_size, len(train_data)):
    x_train.append(train_data[i-initial_window_size:i, 0])
    y_train.append(train_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

#convert the x_train and y_train numpy arrays
x_train,y_train=np.array(x_train),np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0], initial_window_size, 1))

#create the testing dataset
test_data = scaled_data[training_data_len - 10:, :]
x_test = []
y_test = dataset[training_data_len:, :]
for i in range(initial_window_size, len(test_data)):
    x_test.append(test_data[i-initial_window_size:i, 0])

x_test=np.array(x_test)
# x_test=np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))
x_test=np.reshape(x_test, (x_test.shape[0],initial_window_size,1))


# model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.LSTM(100, return_sequences=True, input_shape=(initial_window_size, 1)))
# model.add(tf.keras.layers.LSTM(100, return_sequences=False))
# model.add(tf.keras.layers.Dense(25))
# model.add(tf.keras.layers.Dense(1))


# #train model
# model.compile(optimizer=Adam(learning_rate=0.01), loss='mean_squared_error', metrics=['accuracy'])
# model.fit(x_train, y_train, batch_size=32, epochs=50)
model = tf.keras.models.load_model(f"{symbol}_model.h5")

train_predictions = model.predict(x_train)
train_predictions = scaler.inverse_transform(train_predictions)

y_train_original =  y_train.reshape(-1, 1)
y_train_original=scaler.inverse_transform(y_train_original)

test_predictions=model.predict(x_test)
test_predictions=scaler.inverse_transform(test_predictions)

y_test_original = y_test.reshape(-1, 1)


# model.save(f"{symbol}_model.h5")


window_size =80

# Select the last 80 days of data
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


latest_dates = df.index[-window_size:]# Dates for the latest data

latest_data = df['Open'].tail(window_size)


x_labels = [f"Day {i+1}" for i in range(len(predicted_prices))]

#for web site
fig1, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(train_dates,y_train_original, label='Actual traint', color='blue')
ax1.plot(train_dates,train_predictions, label='Train Prediction', color='orange')
ax1.plot(test_dates,y_test_original,label='Actual testdata',color ='green')
ax1.plot(test_dates,test_predictions,label='test Prediction',color='red')
ax1.set_title('Figure 1')
ax1.set_xlabel('X-axis')
ax1.set_ylabel('Y-axis')
plt.xticks(rotation=45, ha='right')
ax1.legend()
# plugins.connect(fig1, plugins.PointLabelTooltip(ax1))
interactive_plot1 = mpld3.fig_to_html(fig1)
plt.show()


fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.plot(latest_dates,latest_data, label='latest 200 days data')
ax2.plot(x_labels,predicted_prices, label='latest 200 days data')

ax2.set_title('Recent data  and Prediction for 10 days')
ax2.set_xlabel('Date')
ax2.set_ylabel('Open Price')
plt.xticks(rotation=90, ha='right')
ax2.legend()
# plugins.connect(fig2, plugins.PointLabelTooltip(ax2))
interactive_plot2 = mpld3.fig_to_html(fig2)
plt.show()



# #window no of days and prediction
fig3, ax3 = plt.subplots(figsize=(18, 6))
ax3.plot(x_labels,predicted_prices, label='Predicted Prices', color='red')

ax3.set_title('Predicted Prices for Next 10 Days')
ax3.set_xlabel('Prediction days')
ax3.set_ylabel('Price')
plt.xticks(rotation=90, ha='right')
ax1.legend()
plugins.connect(fig3, plugins.PointLabelTooltip(ax3))
interactive_plot3 = mpld3.fig_to_html(fig3)
plt.show()


# Print the predicted prices
print("Predicted Prices for Next 10 Days:", predicted_prices)

# Define thresholds for price difference
threshold_up = 0.05  # Threshold for price increase
threshold_down = -0.05  # Threshold for price decrease

# Compute the price differences between actual and predicted prices
price_difference = y_test_original - test_predictions

# Classify predictions based on thresholds
price_increase_predicted = (predicted_prices > threshold_up).astype(int)
price_decrease_predicted = (predicted_prices < threshold_down).astype(int)

price_increase_actual = (y_test_original > threshold_up).astype(int)
price_decrease_actual = (y_test_original < threshold_down).astype(int)

# Calculate true positives, false positives, true negatives, false negatives
TP = np.sum(price_increase_predicted * price_increase_actual)
FP = np.sum(price_increase_predicted * (1 - price_increase_actual))
TN = np.sum((1 - price_increase_predicted) * (1 - price_increase_actual))
FN = np.sum((1 - price_increase_predicted) * price_increase_actual)

# Calculate accuracy, precision, recall, and F1-score
accuracy = (TP + TN) / (TP + FP + TN + FN)
precision = TP / (TP + FP + 1e-9)  # Adding 1e-9 to avoid division by zero
recall = TP / (TP + FN + 1e-9)  # Adding 1e-9 to avoid division by zero
f1_score = 2 * (precision * recall) / (precision + recall + 1e-9)  # Adding 1e-9 to avoid division by zero

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1_score)




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import mean_average_precision
from keras.optimizers import Adam
import math
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.metrics import mean_squared_error
plt.style.use('fivethirtyeight')

# Import dataset
df = pd.read_csv(r'data\Apt110_2016.csv')
df.Date = pd.to_datetime(df['Date'])

# Setting the date as index will make our time series plots much more understandable.
df.set_index('Date', inplace=True)

# Normalising data
# diff = df['Load'].max() - df['Load'].min()
# df['Load'] = df['Load'] - df['Load'].min()

# Firstly, we will define a new dataset equal to the existing one, but omitting the last 10 records,
# later we will use the model to predict such values.
new_df = df['Load'].iloc[:-10]

# We define the length of the training set as 95% of the total records inf new dataframe
train_len = math.ceil(len(new_df)*0.99)

# Based on window length training and validation data is generated.
window = 5

# Training dataset generation
train_data = new_df[0:train_len]
X_train = []
Y_train = []
for i in range(window, len(train_data)):
    X_train.append(train_data[i - window:i])
    Y_train.append(train_data[i])

X_train, Y_train = np.array(X_train), np.array(Y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Validation dataset generation
test_data = new_df[train_len - window:]
X_val = []
Y_val = []
for i in range(window, len(test_data)):
    X_val.append(test_data[i - window:i])
    Y_val.append(test_data[i])

X_val, Y_val = np.array(X_val), np.array(Y_val)
X_val = np.reshape(X_val, (X_val.shape[0], X_val.shape[1], 1))

# LSTM MODEL
model = Sequential()
model.add(LSTM(50, return_sequences=True, activation='relu', input_shape=(X_train.shape[1], 1), recurrent_dropout=0.2))
model.add(LSTM(50, return_sequences=False, activation='relu'))
model.add(Dense(100))
model.add(Dropout(0.2))
model.add(Dense(75))
model.add(Dropout(0.2))
model.add(Dense(25))
model.add(Dense(1))
opt1 = Adam(learning_rate=0.005, beta_1=0.9, beta_2=0.999)
model.compile(loss='mean_squared_error', optimizer=opt1)
model.fit(X_train, Y_train, epochs=25, batch_size=5000, verbose=1)

lstm_train_pred = model.predict(X_train)
lstm_valid_pred = model.predict(X_val)
r1 = (np.round(np.sqrt(mean_squared_error(Y_train, lstm_train_pred)), 2))
r2 = (np.round(np.sqrt(mean_squared_error(Y_val, lstm_valid_pred)), 2))
print('Training error:', r1)
print('Validation error:', r2)

# MAPE calculation
def MAPE(actual, predicted):
    s=0
    for i in range(len(actual)):
        s = s + (predicted[i]-actual[i])/actual[i]
    return s/len(actual)*100

m_ap_train = MAPE(Y_train, lstm_train_pred)
m_ap_val = MAPE(Y_val, lstm_valid_pred)
print('MAPE_train:', m_ap_train)
print('MAPE_valid:', m_ap_val)

# Plot the validation output and prediction
valid = pd.DataFrame(new_df[train_len:])
valid['Predictions'] = lstm_valid_pred
plt.figure(figsize=(30, 6))
plt.plot(valid[['Load', 'Predictions']])
plt.legend(['Validation', 'Predictions'])
plt.show()

import numpy as np
import pandas as pd
from keras.models import Sequential  
from keras.layers.core import Dense, Activation  
from keras.layers.recurrent import LSTM

def _load_data(data, n_prev=10):  
    """
    data should be pd.DataFrame()
    """
    docX, docY = [], []
    for i in range(len(data)-n_prev):
        docX.append(data.iloc[i:i+n_prev].as_matrix())
        docY.append(data.iloc[i+n_prev].as_matrix())
    alsX = np.array(docX)
    alsY = np.array(docY)
    return alsX, alsY

def train_test_split(df, test_size=0.1):  
    """
    This just splits data to training and testing parts
    """
    ntrn = round(len(df) * (1 - test_size))

    n_prex = 0
    X_train, y_train = _load_data(df.iloc[0:ntrn])
    X_test, y_test = _load_data(df.iloc[ntrn:])

    return (X_train, y_train), (X_test, y_test)

df = pd.read_pickle('dataframe.pkl')

(X_train, y_train), (X_test, y_test) = train_test_split(df)  # retrieve data

y_train = df["open_pct"][0:20]

in_out_neurons = 101
hidden_neurons = 300

data_dim = 101
timesteps = 100
num_classes = 10
batch_size = 481

model = Sequential()
model.add(LSTM(hidden_neurons, return_sequences=False,
               input_shape=(None, in_out_neurons)))
model.add(Dense(in_out_neurons, input_dim=hidden_neurons))  
model.add(Activation("linear"))  
model.compile(loss='mse', optimizer='rmsprop')

# try second model
model = Sequential()
model.add(LSTM(32, return_sequences=True, input_shape=(None, 101)))
model.add(LSTM(32, return_sequences=True))
model.add(LSTM(32))
model.add(Dense(1, activation = 'linear'))
      
model.compile(loss='mean_squared_error',
              optimizer='rmsprop',
              metrics=['accuracy'])


model.summary()

model.fit(X_train, y_train, epochs=20, validation_split=0.05)

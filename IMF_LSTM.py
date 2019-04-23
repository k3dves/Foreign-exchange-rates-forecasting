import pandas as pd
import matplotlib.pyplot as plt
from PyEMD import EMD
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

def imf_trainer(imf, testseries):
    #imf=scaler.fit_transform(imf)
    #error
    # scaler = MinMaxScaler(feature_range = (0, 1))
    features_set = []
    labels = []
    #imf = sc.fit_transform(imf)
    np.shape(imf)
    for i in range(15, len(imf)):
        features_set.append(imf[i-15:i])
        labels.append(imf[i])

    features_set, labels = np.array(features_set), np.array(labels)
    features_set = np.reshape(features_set, (features_set.shape[0], features_set.shape[1], 1))

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(features_set.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units = 1))
    model.compile(optimizer = 'adam', loss = 'mean_squared_error')

    model.fit(features_set, labels, epochs = 10, batch_size = 80)
    #aAtestseries=scaler.transform(testseries
    # testseries = scaler.transform(testseries)
    test_features=[]
    for i in range(15,len(testseries)):
        test_features.append(testseries[i-15:i,0])
    # test_features = sc.transform(test_features)
    test_features = np.array(test_features)
    test_features = np.reshape(test_features, (test_features.shape[0], test_features.shape[1], 1))

    predictions = model.predict(test_features)
    # predictions = sc.inverse_transform(predictions)
    return predictions

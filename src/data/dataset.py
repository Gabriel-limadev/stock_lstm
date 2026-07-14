import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler

def split_data(df, train_size=0.8):
    '''
    Split the data into training and testing sets.
    '''

    size = int(len(df) * train_size)

    train = df.iloc[:size]
    test = df.iloc[size:]

    return train, test

def scale_data(train, test, features):
    '''
    Scale the data using MinMaxScaler.
    '''
    
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train[features])
    test_scaled = scaler.transform(test[features])

    return (
        train_scaled,
        test_scaled,
        scaler
    )

def create_sequences(data, window_size, target_index):
    '''
    Create sequences of data for LSTM input.
    '''
    
    X = []
    y = []

    for i in range(window_size, len(data)):

        X.append(
            data[
                i-window_size:i
            ]
        )

        y.append(
            data[i, target_index]
        )

    return np.array(X), np.array(y)
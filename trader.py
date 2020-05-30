# initialize (daily):
# create and train new models (one for high and low) - this will keep models up to date and reset scaling functions - this will probably also need a different way to load data?

# run (throughout the day starting at )
# need a clock around the whole thing that repeats forever but only does the thing inside if a minute has passed
# loads data (method)
# import and run model from modelBuilding to get predictions
# math magic about what I want to buy and sell based on my positions (need to be stored in CSV? - probably just pandas table is good)
# trade (method that sends commands and also stores results in some data thing)
# print out how long this iteration took (hopefully less than a minute)

import math
# import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt

from modelBuilding import scaler
from modelBuilding import modelBuilder

PERIOD = 180
OFFSET = 15
CLOSE_COLUMN = 3
TRAIN_RATIO = 0.15
EPOCHS = 1
DATA_FILE = "aapl.csv"

def train_and_test_IPP(data_path="aapl.csv"):

    model = modelBuilder(180, 15, 3, .1, 1, data_path)
    model.test()
    print(model.error)


# Dynamic Recursive Prediction - use predictions from other columns to predict future values
def train_and_test_DRP(period=180, offset=15, column=3, train_ratio=0.8, epochs=1, data_path="aapl.csv"):

    # TODO: change 10 to a variable
    model_list = [None] * 10
    all_predictions = [None] * 10
    all_predictions.clear()
    curr_predictions = all_predictions.copy()
    # all_predictions =

    for i in range(len(model_list)):
        model_list[i] = modelBuilder(p=period, y=1, c=i, t=train_ratio, e=epochs, d=data_path)

    # train model for each column
    # TODO: define num trials
    for i in range(10):
        predictions = [0] * 10
        for j in range(len(model_list)):
        # for i in range(len(model_list)):
            # model_list[i].test()
            predictions[j] = (model_list[j].make_predictions_DRP(np.asarray(curr_predictions)))[0]
            # print(model_list[i].error)
        print("Predictions: {}".format(predictions))
        all_predictions.append(predictions)
        curr_predictions.append(predictions)
        if (len(curr_predictions) > period):
            curr_predictions.pop(0)

    print("All Predictions: {}".format(all_predictions))

    # TODO: use each model to predict one instance of each variable
    # use this to predict future instances

def graph_prediction(raw_data, train_len, y_test, y_pred):

    training = raw_data.iloc[:, 0:train_len]
    valid = raw_data[['close', 'Volume']][train_len:(len(y_test) + train_len)]
    valid.columns = ['close', 'Predictions']

    valid['Predictions'] = y_pred

    plt.figure(figsize=(100, 25))
    plt.title('Model')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.plot(training['close'])
    plt.plot(valid[['close', 'Predictions']])
    plt.legend(['Train', 'Val', 'Predictions'])
    plt.show()

def main():
    train_and_test_DRP(period=PERIOD, offset=OFFSET, column=CLOSE_COLUMN, train_ratio=TRAIN_RATIO, epochs=EPOCHS, data_path=DATA_FILE)

if __name__ == "__main__":
    main()

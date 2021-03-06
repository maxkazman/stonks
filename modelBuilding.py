#NOTES
#Need to make this a CLASSSSSS that creates new models
#Need to have another function that trains the models
#This is because for some reason having the model generate multiple outputs (highs and lows) isn't working and it probably is better to have a separate one for each anyways
#I need my own scaling function!!!!!!! - one that is set in "create" function and used in "train" function

import math
# import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
import joblib
import os.path

class scaler:
    minMax = np.array([[0, 0]])
    unscaleColumn = 0
    def __init__(self, xData = np.asarray([0,1]), uColumn = 0):
        self.unscaleColumn = uColumn
        if(len(xData.shape) == 1):
            xData = xData.reshape([1, 2])
        for i in range(len(xData[0])):
            if (i == 0):
                self.minMax[i, 0] = min(xData[:, i])
                self.minMax[i, 1] = max(xData[:, i])
            else:
                self.minMax = np.append(self.minMax, np.asarray([[min(xData[:, i]), max(xData[:, i])]]), axis=0)
    def scale(self, x):
        for i in range(len(x)):
            for j in range(len(x[i])):
                if ((self.minMax[j, 1] - self.minMax[j, 0]) != 0):
                    x[i,j] = (x[i,j] - self.minMax[j, 0])/(self.minMax[j, 1] - self.minMax[j, 0])
        return x
    def unscale(self, y):
        for i in range(len(y)): #assumes one-dimensional
            y[i] = y[i] * (self.minMax[self.unscaleColumn, 1] - self.minMax[self.unscaleColumn, 0]) + self.minMax[self.unscaleColumn, 0]
        return y

class modelBuilder:

    PERIOD = 180 #how much time is used for prediction
    Y_OFFSET = 15 #how far in the future you are predicting based on current data (in min)
    CLOSE_COLUMN = 3 #column the close variable is in (0 indexed)
    TRAIN_RATIO = 0.8 #amount of data from the training set that is used to train
    EPOCHS = 1
    DATA_FILE = "aapl.csv"
    model = None
    scalerObject = None
    error = None

    def __init__(self, p, y, c, t, e, d, model_path="nothing"):
        self.PERIOD = p
        self.Y_OFFSET = y
        self.CLOSE_COLUMN = c
        self.TRAIN_RATIO = t
        self.EPOCHS = e
        self.DATA_FILE = d

        rawData = pd.read_csv(self.DATA_FILE, index_col=0)

        #this somehow turns it into a numpy array
        rawDataNum = rawData.values

        #create training data - TODO make it predict next 30 minutes every time
        trainLen = math.ceil(len(rawDataNum) * self.TRAIN_RATIO)
        self.scalerObject = scaler(rawDataNum, self.CLOSE_COLUMN)
        # print(rawDataNum.shape)
        scaled = self.scalerObject.scale(rawDataNum)

        train_data = scaled[0:trainLen, :]
        x = []
        y = []

        for i in range(self.PERIOD, trainLen - self.Y_OFFSET):
            x.append(train_data[i-self.PERIOD:i, :])
            y.append(train_data[i + self.Y_OFFSET, self.CLOSE_COLUMN])

        x, y = np.array(x), np.array(y)

        #making model
        self.model = Sequential()
        self.model.add(LSTM(50, return_sequences=True, input_shape=(x.shape[1], x.shape[2])))
        self.model.add(LSTM(50, return_sequences=False))
        self.model.add(Dense(25))
        self.model.add(Dense(1))

        #compile model
        self.model.compile(optimizer='adam', loss='mean_squared_error')

        if os.path.isfile(model_path):
            print("Loading model: {}".format(model_path))
            self.model = joblib.load(model_path)
        else:
            print("Training model: {}".format(model_path))
            self.model.fit(x, y, batch_size=1, epochs=self.EPOCHS)
            joblib.dump(self.model, model_path)

    def predict(self, xTest): #change to predict

        #scale input data:
        xTest = np.asarray(xTest)
        for i in range(len(xTest)):
            xTest[i] = self.scalerObject.scale(xTest[i])
        predictions = self.model.predict(np.asarray(xTest))
        predictions = self.scalerObject.unscale(predictions)
        return predictions

    #new function for test(self, no args) which outputs a quantification of accuracy
    def test(self):
        rawDataNum = pd.read_csv(self.DATA_FILE, index_col=0).values
        testStart = math.ceil(len(rawDataNum) * self.TRAIN_RATIO) + self.PERIOD
        if (testStart - len(rawDataNum) >= 0):
            print("there isn't enough data left to test - it probably errored")
        testData = rawDataNum[testStart - self.PERIOD:, :]
        xTest = []
        yTest = rawDataNum[testStart:(len(rawDataNum) - self.Y_OFFSET), self.CLOSE_COLUMN]
        for i in range(self.PERIOD, len(testData) - self.Y_OFFSET):
            xTest.append(testData[i-self.PERIOD:i, :])

        yPredict = self.predict(xTest).flatten()

        self.error = np.sqrt(np.mean((yPredict - yTest)**2))

    def make_predictions_DRP(self, prev_predictions):
        rawDataNum = pd.read_csv(self.DATA_FILE, index_col=0).values
        testStart = math.ceil(len(rawDataNum) * self.TRAIN_RATIO)

        if (testStart >= len(rawDataNum)):
            print("there isn't enough data left to test - it probably errored")
            return

        testData = rawDataNum[testStart:testStart+self.PERIOD, :]
        if (len(prev_predictions) != 0):
            testData = np.concatenate((testData, prev_predictions))
            for i in range(len(prev_predictions)):
                testData = np.delete(testData, 1, 0)

        xTest = []
        for i in range(1):
            xTest.append(testData[i:i+self.PERIOD, :])
        yPredict = self.predict(xTest).flatten()

        return yPredict

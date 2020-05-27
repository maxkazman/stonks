#initialize (daily):
#create and train new models (one for high and low) - this will keep models up to date and reset scaling functions - this will probably also need a different way to load data?

#run (throughout the day starting at )
#need a clock around the whole thing that repeats forever but only does the thing inside if a minute has passed
#loads data (method)
#import and run model from modelBuilding to get predictions
#math magic about what I want to buy and sell based on my positions (need to be stored in CSV? - probably just pandas table is good)
#trade (method that sends commands and also stores results in some data thing)
#print out how long this iteration took (hopefully less than a minute)

import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt


from modelBuilding import scaler
from modelBuilding import modelBuilder


rawData = pd.read_csv("aapl.csv", index_col=0)

rawDataNum = rawData.values

trainLen = math.ceil(len(rawData) * .8)

testData = rawDataNum[trainLen - 180: , :]
xTest = []
yTest = rawDataNum[trainLen:(len(rawDataNum) - 15), 3]
for i in range(180, len(testData) - 15):
    xTest.append(testData[i-180:i, :])


testModel = modelBuilder(180, 15, 3, .8, 1, "aapl.csv")

yPredict = testModel.test(xTest).flatten()

training = rawData.iloc[:, 0:trainLen]
valid = rawData[['close','Volume']][trainLen:(len(yTest) + trainLen)]
valid.columns = ['close', 'Predictions']

valid['Predictions'] = yPredict

plt.figure(figsize=(100,25))
plt.title('Model')
plt.xlabel('Time')
plt.ylabel('Price')
plt.plot(training['close'])
plt.plot(valid[['close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'])
plt.show()
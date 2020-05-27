#this is gonna update CSVs (or a database?????) all the time - this is gonna be really hard cause it has to receive the webhooks from that website all the time
#once I actually implement this I'm gonna need to redo the loaddata functions from trader.py - maybe just include the two load data functions in here (the daily one and the minutely one)


#make test data
        testData = scaled[trainLen - self.PERIOD: , :]
        xTest = []
        yTest = scaled[trainLen:(len(scaled) - self.Y_OFFSET), self.CLOSE_COLUMN]
        for i in range(self.PERIOD, len(testData) - self.Y_OFFSET):
            xTest.append(testData[i-self.PERIOD:i, :])

        xTest = np.array(xTest)

        #run model
        predictions = model.predict(xTest).flatten()

        #reshape into to be unscaled
        unscaled = scaled
        unscaled[(len(unscaled) - len(predictions)):, CLOSE_COLUMN] = predictions
        unscaled = scaler.inverse_transform(unscaled)
        unscaledPredictions = unscaled[(len(unscaled) - len(predictions)):, CLOSE_COLUMN]

        print(np.sqrt(np.mean((predictions - yTest)**2)))

        #plotting
        training = rawData.iloc[:, 0:trainLen]
        valid = rawData[['close','Volume']][trainLen:(len(scaled) - Y_OFFSET)]
        valid.columns = ['close', 'Predictions']
        valid['Predictions'] = unscaledPredictions

        plt.figure(figsize=(100,25))
        plt.title('Model')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.plot(training['close'])
        plt.plot(valid[['close', 'Predictions']])
        plt.legend(['Train', 'Val', 'Predictions'])
        plt.show()
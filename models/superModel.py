# Rocco Haro & brandon sayers
# the supermodel class abstracts renewable models for one particular town

import models.stackedLSTM as modelBuilder_LSTM
import pandas as pd
import random
import models.workingNN as NN
import numpy as np
import csv

class renewableModel:
    def __init__(self, _id, dataFileTarget):
        self.id = _id
        self.NN = None
        self.LSTM_Models = []
        self.countFeats = 0
        self.dataFileTarget = dataFileTarget
        self.dataFrame = self.loadData()
        self.config()

    def loadData(self):
        # Pull all data from CSV file and
        # push into a dataframe for portability.

        df = pd.read_csv(self.dataFileTarget, index_col=0, skiprows=[1])
        df.index = pd.to_datetime(df.index)

        return df

    def getNumOfFeats(self):
        # TODO TODO TODO TODO
        # return the number of features in the data
        return self.countFeats

    def getSuperTestData(self, lookBackSize, ySize):
        # Grab a batch-sized batch of data

        arr_lookBack = []
        arr_futureFeature = []
        sizeof_dataframe = self.dataFrame.shape[0]
        sizeof_dataToPull = ySize + lookBackSize
        # print(sizeof_dataframe)

        # self.dataFrame location to pull data from
        start = int(sizeof_dataframe * np.random.rand())
        while (sizeof_dataframe < start + sizeof_dataToPull):
            start = int(sizeof_dataframe * np.random.rand())

        end_lookBack = start + lookBackSize
        end_actualY = start + sizeof_dataToPull

        for column in self.dataFrame:
            if column != "power_output":
                temp_arr = np.asarray(list(self.dataFrame[column][start:end_lookBack]))
                temp_arr = temp_arr.reshape(1,lookBackSize, 1)
                arr_lookBack.append(temp_arr)

                temp_future = np.asarray(list(self.dataFrame[column][end_lookBack:end_actualY]))
                arr_futureFeature.append(temp_future)


        arr_actualY = np.asarray(list(self.dataFrame["power_output"][end_lookBack:end_actualY]))
        arr_actualY = arr_actualY.reshape(ySize, 1)

        return arr_lookBack, arr_futureFeature, arr_actualY

    def masterTest(self):
        # batchSize:           number of past prediction to consider
        # ySize:               number of future predictions to consider
        # lookBackDataFeature: list of each LSTM's timestep data
        # futureFeature:       list of what each LSTM's future values should be
        # actual_Y:            list of power_output for each timestep
        features_forecasts = []
        batchSize = self.LSTM_Models[0].n_steps
        ySize = self.LSTM_Models[0].n_outputs
        lookBackDataFeature, futureFeature, actual_Y = self.getSuperTestData(batchSize, ySize)

        numOfFeats = self.getNumOfFeats()

        for i in range(numOfFeats):
            # TODO get unique look back for each feature from the same timesteps
            
            # lookBackData = killme[i]
            lookBackData = lookBackDataFeature[i]

            # TODO
            # NOTE: You can modify the codde in the forecastGiven
            # and add in the actual_Y.csv so that you can draw a graph
            # of the accuracy for each LSTM model.
            # investigate the code to figure out how to get it to work...
            # OR you can just do a straight up comparison in excel... but i woulnd't recommend that lol

            forecastForFeat_i = self.LSTM_Models[i].forecastGiven(lookBackData)
            features_forecasts.append(forecastForFeat_i)
            # for t in timeSteps:
                # curr_forecast = formatFeatureSet(feat_forecast, timestep)

        print("features_forecasts:", features_forecasts)
        forecasted_Power = []
        # 1 - .20 % NN, .5 loss LStm
        # 2 - 0.90 % NN, 0.01 loss LSTM
        # 3 - 0.95 % NN, 0.001 loss LSTM
        num = 3

        with open('superModel_Results_'+str(num), 'w') as csvFile:
            wr = csv.writer(csvFile, delimiter=",")
            for timestep in range(5):
                currFeatsInTimestep = []
                for feature in features_forecasts:
                    currFeatsInTimestep.append(feature[0][timestep])
                currFeatsInTimestep.append(feature[0][0])
                wr.writerow([["currFeatsInTimestep_"+str(timestep)], currFeatsInTimestep])
                print("Feats in timestep: " + str(timestep), currFeatsInTimestep)
                print("currFeatsInTimestep :", currFeatsInTimestep)
                curr_classification = self.NN.classifySetOf(currFeatsInTimestep)

                wr.writerow([["curr_classification_"+str(timestep)], curr_classification])
                forecasted_Power.append(curr_classification)

        print("forecasted_Power: ", forecasted_Power)

        self.NN.closeSession()
        return 0

    def train(self):
        # thread each model for training
        # continue training until NN > 95%
        NN_targetAcc = 0.1
        #try:
        self.NN.train(NN_targetAcc)
        #except:
        # self.NN.closeSession()
        # and loss over all feature models are satisfactory

        for i in range(self.getNumOfFeats()):
            self.LSTM_Models[i].train(target_loss = 1)

            # single model testing, not super model testing as that is done in masterTest
            # self.LSTM_Models[0].test()

        self.masterTest()

    def config(self):
        # initialize the NN
            # selecting network parameters?
        trainingDataPath = self.dataFileTarget # 12 is the most recent data with richer features (EMA)
        self.NN = NN.neuralNetwork(self.id, dataFileTarget=trainingDataPath) # configure options for NN ==  dataFileTarget="", LOG_DIR="LSTM_LOG/log_tb/temp", batchSize=144, hiddenSize=256, displaySteps=20):
        x = dict()
        x['temp'] = "temperature"
        x['hum'] = " asdf "

        # initialize the LSTMS
            # couont how many features there are
        count = 0
        for column in self.dataFrame:
            if column != "power_output": # TODO maybe don't include moving averages
                self.countFeats+=1
                curr_lstm = modelBuilder_LSTM.StackedLSTM(dataFrame=self.dataFrame[column], modelName=column)
                curr_lstm.networkParams(column) # can pass in custom configurations Note: necessary to call this function
                self.LSTM_Models.append(curr_lstm)
                #count+=1

       # curr_lstm = modelBuilder_LSTM.StackedLSTM(dataFileTarget='models/varyingData/moving/temperature', modelName="temperature")

        # for each F in len(features):
            # create lstm model for each feature

        # start training the models
        self.train()

    def printID(self):
        print("ID: ", self.id)

class superModel:
    """ This class represents a single town, where the town may contain more than
        one renewable source of energy. Each source has a unique Neural Net dedicated
        to classifying environmental features.
    """
    def __init__(self, numOfRenewables):
        self.renewableModels = []
        for i in range(numOfRenewables):
            self.renewableModels.append(renewableModel(i, "prod_Data/training_Data12.csv"))

        self.renewableModels[0].printID()


if __name__ == "__main__":
    numOfRenewables = 1
    SM = superModel(numOfRenewables)
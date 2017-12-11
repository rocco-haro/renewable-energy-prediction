# Rocco Haro & brandon sayers
# the supermodel class abstracts renewable models for one particular town

import models.stackedLSTM as modelBuilder_LSTM
import pandas as pd
import random
import models.workingNN as NN
import numpy as np
import csv
import math
import matplotlib.pyplot as plt
import numpy as np

class renewableModel:
    def __init__(self, _id, dataFileTarget):
        self.id = _id
        self.NN = None
        self.LSTM_Models = []
        self.countFeats = 0
        self.dataFileTarget = dataFileTarget
        self.renewableModel_Test_accuracy = 0
        self.reTrainLSTM = False
        self.dataFrame = self.loadData()

        # TODO
        # Have this decrease each time its called
        self.testNum            = "11"      # Increase this each time
        self.highNoiseTarget    = .001
        self.medNoiseTarget     = .0001
        self.lowNoiseTarget     = .00001
        self.highNoiseFeatures  = ["events", "gust_speed", "power_EMA60", "power_EMA90", "conditions", "wind_dir", "power_out_prev"]
        self.medNoiseFeatures   = ["power_MA10", "power_MA25", "dew_point", "visibility", "wind_speed", "temp", "humidity"]
        self.lowNoiseFeatures   = ["pressure", "precip", "power_EMA30", "power_MA50"]


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

        maxBatchSize =  -100000 #self.LSTM_Models[0].n_steps
        maxYSize = -100000 # self.LSTM_Models[0].n_outputs

        for i in range(self.getNumOfFeats()):
            if (self.LSTM_Models[i].n_steps > maxBatchSize):
                maxBatchSize = self.LSTM_Models[i].n_steps
            if (self.LSTM_Models[i].n_outputs > maxYSize):
                maxYSize = self.LSTM_Models[i].n_outputs

        lookBackDataFeature, futureFeature, actual_Y = self.getSuperTestData(maxBatchSize, maxYSize)


        numTests = 10
        masterTest_Accuracy_Avg = 0
        for k in range(numTests):
            numOfFeats = self.getNumOfFeats()
            features_forecasts = []
            lookBackDataFeature, futureFeature, actual_Y = self.getSuperTestData(maxBatchSize, maxYSize)

            for i in range(numOfFeats):
                steps = self.LSTM_Models[i].n_steps
                print("loobackdata: ", lookBackDataFeature[i] )
                # l, _, _ =  self.getSuperTestData(self.LSTM_Models[i].n_steps, self.LSTM_Models[i].n_outputs)  #lookBackDataFeature[i]
                lookBackData = np.array(lookBackDataFeature[i][0][maxBatchSize-steps:]) # to feed in the correct amount of values .. we have varying lookback for the LSTM_Models
                lookBackData = lookBackData.reshape(1, steps, 1)
                print(lookBackDataFeature[i][0])
                print(lookBackData)
                # TODO
                # NOTE: You can modify the codde in the forecastGiven
                # and add in the actual_Y.csv so that you can draw a graph
                # of the accuracy for each LSTM model.
                # investigate the code to figure out how to get it to work...
                # OR you can just do a straight up comparison in excel... but i woulnd't recommend that lol
                forecastForFeat_i = self.LSTM_Models[i].forecastGiven(lookBackData)
                features_forecasts.append(forecastForFeat_i)

            print("features_forecasts:", features_forecasts)

            forecasted_Power = []
            testResults = []
            num_timeSteps = maxYSize
            difference = []
            graphTheTest = True

            with open('resUlts/superModel_Results_'+str(self.testNum), 'w') as csvFile:
                wr = csv.writer(csvFile, delimiter=",")
                renewableModel_Test_accuracy_MA = self.renewableModel_Test_accuracy
                # while renewableModel_Test_accuracy_MA < 0.50

                for timestep in range(num_timeSteps):
                    currFeatsInTimestep = []
                    for feature in features_forecasts:
                        currFeatsInTimestep.append(feature[0][timestep])
                    currFeatsInTimestep.append(feature[0][0])
                    wr.writerow([["currFeatsInTimestep_"+str(timestep)], currFeatsInTimestep])
                    print("Feats in timestep: " + str(timestep), currFeatsInTimestep)
                    print("currFeatsInTimestep :", currFeatsInTimestep)


                    curr_classification = self.NN.classifySetOf(currFeatsInTimestep)
                    act_Y = actual_Y[timestep][0]
                    pred_Y = curr_classification[0]
                    eclud_distance = math.sqrt((math.fabs((act_Y-pred_Y)**2 -  timestep)))
                    difference.append(eclud_distance)

                    wr.writerow([["curr_classification_"+str(timestep) + "_testNumer: " +""str(k)], curr_classification])
                    forecasted_Power.append(curr_classification)

                wr.writerow([["Actual y: "], actual_Y])

                if (graphTheTest):
                    #plt.subplot(numTests, 1, k)
                    time = [x for x in range(num_timeSteps)]
                    actY = np.squeeze(actual_Y)
                    actY = actY.tolist()
                    plt.plot(time,actY, color='green', linestyle=':' )
                    forecasts = np.squeeze(forecasted_Power).tolist()
                    plt.plot(time,forecasts, color='red' )
                    plt.show()

                masterTest_Accuracy_Avg+= math.fsum(difference)

            print("forecasted_Power: ", forecasted_Power)

        errorThreshold = 13.48 # error froom seq off by 1 , 2 , 3 , 4 ... N respective to time
        masterTest_Accuracy_Avg/=numTests
        if (errorThreshold <= masterTest_Accuracy_Avg):
            self.reTrainLSTM = True
            #self.train(1)
        else:
            time = [x for x in range(num_timeSteps)]
            actY = np.squeeze(actual_Y)
            actY = actY.tolist()
            plt.plot(time,actY, color='green', linestyle=':')
            forecasts = np.squeeze(forecasted_Power).tolist()
            plt.plot(time,forecasts, color='red')
            plt.show()

        self.NN.closeSession()
        return 0

    def train(self, state):
        # thread each model for training
        # continue training until NN > 95%
        if (state == 0):
            NN_targetAcc = 0.97
            #try:
            self.NN.train(NN_targetAcc)

        i = 0;
        for column in self.dataFrame:
            if column in self.highNoiseFeatures:
                self.LSTM_Models[i].train(target_loss=self.highNoiseTarget)
                i += 1
            elif column in self.medNoiseFeatures:
                self.LSTM_Models[i].train(target_loss=self.medNoiseTarget)
                i += 1
            elif column in self.lowNoiseFeatures:
                self.LSTM_Models[i].train(target_loss=self.lowNoiseTarget)
                i += 1

        self.masterTest()

    def config(self):
        # initialize the NN
            # selecting network parameters?
        trainingDataPath = self.dataFileTarget # 12 is the most recent data with richer features (EMA)
        self.NN = NN.neuralNetwork(self.id, dataFileTarget=trainingDataPath) # configure options for NN ==  dataFileTarget="", LOG_DIR="LSTM_LOG/log_tb/temp", batchSize=144, hiddenSize=256, displaySteps=20):

        # initialize the LSTMS
            # couont how many features there are
        for column in self.dataFrame:
            if column != "power_output": # TODO maybe don't include moving averages
                self.countFeats+=1
                curr_lstm = modelBuilder_LSTM.StackedLSTM(dataFrame=self.dataFrame[column], modelName=("/" + column+self.testNum))
                if column in self.highNoiseFeatures:
                    curr_lstm.networkParams(column, n_steps=20, n_layers=4) # can pass in custom configurations Note: necessary to call this function
                    self.LSTM_Models.append(curr_lstm)
                elif column in self.medNoiseFeatures:
                    curr_lstm.networkParams(column, n_steps=40, n_layers=4) # can pass in custom configurations Note: necessary to call this function
                    self.LSTM_Models.append(curr_lstm)
                elif column in self.lowNoiseFeatures:
                    curr_lstm.networkParams(column, n_steps=18, n_layers=3) # can pass in custom configurations Note: necessary to call this function
                    self.LSTM_Models.append(curr_lstm)

        # start training the models
        self.train(0)

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
            self.renewableModels.append(renewableModel(i, "prod_Data/training_Data.csv"))
        self.renewableModels[0].printID()

if __name__ == "__main__":
    numOfRenewables = 1
    SM = superModel(numOfRenewables)

    # w/ lstm configuration of : def networkParams(self,ID, n_input = 1,n_steps = 11, n_hidden= 2, n_outputs = 5 , n_layers = 2, loading=False  ):
            # 1 - .20 % NN, .5 loss LStm
            # 2 - 0.90 % NN, 0.01 loss LSTM
            # w/ lstm configuration of :     def networkParams(self,ID, n_input = 1,n_steps = 11, n_hidden=20, n_outputs = 5 , n_layers = 5, loading=False  ):
            # 3 - 0.95 % NN, 0.001 loss LSTM
            # 4 - 0.97 % NN, 0.0001 loss lstm

            # 5 - 0.95  % NN, 0.05 lss lstm decrements by 0.01 if testing does not meet requirements
            # 6 - 0.965 % NN, 0.003 loss lstm decrements by 0.01 if testing does not meet requirements
            #     n_steps=18, n_layers=18
            # 7 - 0.97  % NN, .003 loss lstm decrements by 0.01 if testing does not meet requirements
            #     n_steps=12,n_layers=4
            # 8   0.97  % NN, .002 loss lstm decrements by 0.01 if testing does not meet requirements
            #     n_steps=24,n_layers=4
            # 9   0.97  % NN, .001 loss lstm decrements by 0.01 if testing does not meet requirements
            #     n_steps=36,n_layers=4
            # 10
            #     Noisy events | gust_speed
            #     0.97  % NN, .001 loss lstm decrements by 0.01 if testing does not meet requirements
            #     n_steps=40,n_layers=4
            #     Smooth
            #     0.97  % NN, .0001 loss lstm decrements by 0.01 if testing does not meet requirements
            #     n_steps=18,n_layers=4
            # 11
            # self.highNoiseTarget    = .001            n_steps=20, n_layers=4
            # self.medNoiseTarget     = .0001           n_steps=40, n_layers=4
            # self.lowNoiseTarget     = .00001          n_steps=18, n_layers=3
            # self.highNoiseFeatures  = ["events", "gust_speed", "power_EMA60", "power_EMA90", "conditions", "wind_dir", "power_out_prev"]
            # self.medNoiseFeatures   = ["power_MA10", "power_MA25", "dew_point", "visibility", "wind_speed", "temp", "humidity"]
            # self.lowNoiseFeatures   = ["pressure", "precip", "power_EMA30", "power_MA50"]

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
<<<<<<< HEAD
=======
        self.renewableModel_Test_accuracy = 0
        self.reTrainLSTM = False
>>>>>>> 64b0f9d6c6c5d44e053e1240768c375c4cff5691
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
<<<<<<< HEAD
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
=======
        killme = dict()
        killme[0] = [[[0.1441566904],
                    [0.1475406982],
                    [0.1543119368],
                    [0.1543119368],
                    [0.1553984474],
                    [0.1614521889],
                    [0.1679046177],
                    [0.1652883809],
                    [0.1645498985],
                    [0.1810413307],
                    [0.1816691056]]]

        killme[1] = [[
                    [0.654695],
            [0.67611],
            [0.6888036111],
            [0.6888036111],
            [0.6880211111],
            [0.6907802778],
            [0.6825419444],
            [0.6458975],
            [0.6214372222],
            [0.6912244444],
            [0.6823580556],

                    ]]

        killme[2] = [[
            [0.2420514493],
            [0.2831937198],
            [0.2554714976],
            [0.2554714976],
            [0.2524198068],
            [0.2441910628],
            [0.1898086473],
            [0.117946715],
            [0.1046561594],
            [0.2210125362],
            [0.3033555556]]]


        killme[3] = [[
            [0.2941176471],
            [0.2941176471],
            [0.2941176471],
            [0.0588235294],
            [0.1176470588],
            [0.2941176471],
            [0.1176470588],
            [0.2941176471],
            [0.1176470588],
            [0.1176470588],
            [0.1176470588]]
                    ]


        killme[4] = [[
            [-0.5518518519],
            [-0.5333333333],
            [-0.5333333333],
            [-0.5333333333],
            [-0.5333333333],
            [-0.5333333333],
            [-0.5333333333],
            [-0.6],
            [-0.6],
            [-0.6],
            [-0.6]
                    ]]

        killme[5] = [[
                [0.1428571429],
                [0.1428571429],
                [0.1428571429],
                [0.1428571429],
                [0.2857142857],
                [0.1428571429],
                [0.2857142857],
                [0.1428571429],
                [0.2857142857],
                [0.2857142857],
                [0.2857142857]
                ]
                        ]
        killme[6] = [[
            [0.6032786885],
            [0.5475409836],
            [0.6032786885],
            [0.5475409836],
            [0],
            [0.5655737705],
            [0],
            [0.4901639344],
            [0],
            [0],
            [0]
                    ]]
        killme[7] = [[
            [0.5],
            [0.52],
            [0.52],
            [0.52],
            [0.52],
            [0.52],
            [0.52],
            [0.48],
            [0.48],
            [0.48],
            [0.48]]
                    ]

        killme[8] = [[
                [0],
                [0],
                [0],
                [0],
                [0],
                [0],
                [0],
                [0],
                [0],
                [0],
                [0]]
        ]

        killme[9] = [[
            [0.9638671875],
            [0.9658203125],
            [0.9658203125],
            [0.9654947917],
            [0.9654947917],
            [0.9654947917],
            [0.9654947917],
            [0.9651692708],
            [0.9651692708],
            [0.9651692708],
            [0.9651692708]]
        ]

        killme[10] = [[
            [1],
            [0.7],
            [0.6],
            [0.6],
            [0.6],
            [1],
            [1],
            [1],
            [1],
            [1],
            [1]]
                    ]

        killme[11] = [[
            [0.5348837209],
            [0.4418604651],
            [0.488372093],
            [0.4418604651],
            [0.4418604651],
            [0.3953488372],
            [0.488372093],
            [0.488372093],
            [0.4418604651],
            [0.3488372093],
            [0.4418604651]]
        ]

        killme[12] = [[
            [0.5348837209],
            [0.4418604651],
            [0.5279069767],
            [0.5069767442],
            [0.4813953488],
            [0.4511627907],
            [0.4418604651],
            [0.4395348837],
            [0.4302325581],
            [0.4348837209],
            [0.4348837209]]
        ]

        killme[13] = [[
            [0.5348837209],
            [0.4418604651],
            [0.5279069767],
            [0.5069767442],
            [0.4813953488],
            [0.4511627907],
            [0.4418604651],
            [0.4395348837],
            [0.4302325581],
            [0.4576744186],
            [0.4437209302]]


        ]

        killme[14] = [[
            [0.5348837209],
            [0.4418604651],
            [0.5279069767],
            [0.5069767442],
            [0.4813953488],
            [0.4511627907],
            [0.4418604651],
            [0.4395348837],
            [0.4302325581],
            [0.4576744186],
            [0.4437209302]]
        ]

        killme[15] = [[
            [0.6697674419],
            [0.5888372093],
            [0.5006511628],
            [0.4932837209],
            [0.4624297674],
            [0.450088186],
            [0.4172445767],
            [0.4599210865],
            [0.4769916904],
            [0.4559129552],
            [0.3916675077]]
        ]

        killme[16] = [[
            [0.6558139535],
            [0.5469767442],
            [0.452372093],
            [0.484772093],
            [0.4461516279],
            [0.4422895814],
            [0.4000429116],
            [0.4795391749],
            [0.4874888012],
            [0.4464232987],
            [0.3585958182]]
        ]

        killme[17] = [[
            [0.6837209302],
            [0.6390697674],
            [0.5799069767],
            [0.5524465116],
            [0.5192706977],
            [0.4960476279],
            [0.4658379907],
            [0.4725982214],
            [0.4773303829],
            [0.4666894075],
            [0.4313337481]]
            ]

        numTests = 1
        masterTest_Accuracy_Avg = 0
        for k in range(numTests):
            numOfFeats = self.getNumOfFeats()
            for i in range(numOfFeats):
                # TODO get unique look back for each feature from the same timesteps
                # lookBackData = [[[-0.34587266],
                #     [-0.34637179],
                #     [-0.34822873],
                #     [-0.35029343],
                #     [-0.3497669 ],
                #     [-0.35025731],
                #     [-0.35231279],
                #     [-0.35329935],
                #     [-0.35478794],
                #     [-0.35399758]]]
                lookBackData = killme[i]
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

            # w/ lstm configuration of : def networkParams(self,ID, n_input = 1,n_steps = 11, n_hidden= 2, n_outputs = 5 , n_layers = 2, loading=False  ):
            # 1 - .20 % NN, .5 loss LStm
            # 2 - 0.90 % NN, 0.01 loss LSTM

            # w/ lstm configuration of :     def networkParams(self,ID, n_input = 1,n_steps = 11, n_hidden=20, n_outputs = 5 , n_layers = 5, loading=False  ):

            # 3 - 0.95 % NN, 0.001 loss LSTM
            # 4 - 0.97 % NN, 0.0001 loss lstm

            # 5 - 0.95 % NN, 0.05 lss lstm decrements by 0.01 if testing does not meet requirements
            num = "testing"

            testResults = []
            actual_Y = [[17], [21], [19], [19], [16]]
            num_timeSteps = 5
            difference = []
            graphTheTest = False

            with open('resUlts/superModel_Results_'+str(num), 'w') as csvFile:
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
                    eclud_distance = math.sqrt((math.fabs(act_Y-pred_Y)**2 -  timestep) )
                    difference.append(eclud_distance)

                    wr.writerow([["curr_classification_"+str(timestep)], curr_classification])
                    forecasted_Power.append(curr_classification)

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
            self.train(1)
        else:
            time = [x for x in range(num_timeSteps)]
            actY = np.squeeze(actual_Y)
            actY = actY.tolist()
            plt.plot(time,actY, color='green', linestyle=':' )
            forecasts = np.squeeze(forecasted_Power).tolist()
            plt.plot(time,forecasts, color='red' )
            plt.show()

>>>>>>> 64b0f9d6c6c5d44e053e1240768c375c4cff5691

        self.NN.closeSession()
        return 0

    def train(self, state):
        # thread each model for training
        # continue training until NN > 95%
<<<<<<< HEAD
        NN_targetAcc = 0.1
        #try:
        self.NN.train(NN_targetAcc)
        #except:
        # self.NN.closeSession()
        # and loss over all feature models are satisfactory

        for i in range(self.getNumOfFeats()):
            self.LSTM_Models[i].train(target_loss = 1)
=======
        if (state == 0):
            NN_targetAcc = 0.95
            #try:
            self.NN.train(NN_targetAcc)

            # TODO
            # Have this decrease each time its called
        targetLoss = 0.05 - 0.01


        for i in range(self.getNumOfFeats()):
            self.LSTM_Models[i].train(target_loss = targetLoss)
>>>>>>> 64b0f9d6c6c5d44e053e1240768c375c4cff5691

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
            self.renewableModels.append(renewableModel(i, "prod_Data/training_Data12.csv"))

        self.renewableModels[0].printID()


if __name__ == "__main__":
    numOfRenewables = 1
    SM = superModel(numOfRenewables)
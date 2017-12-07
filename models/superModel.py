# Rocco Haro & brandon sayers
# the supermodel class abstracts renewable models for one particular town

import models.stackedLSTM as modelBuilder_LSTM
import pandas as pd
import models.NN as NN

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

    def masterTest(self):
        # for n number of test:
            # y, testInput = getSample()
            # for each feature in batch:
                #feat_forecast.append(sess.run(self.LSTM_Models[feature], stripFeatureSequence(feature/idx, testInput)))
        features_forecasts = []
        numOfFeats = self.getNumOfFeats()
        for i in range(numOfFeats):
            # TODO get unique look back for each feature from the same timesteps
            lookBackData = [[[-0.34587266],
                [-0.34637179],
                [-0.34822873],
                [-0.35029343],
                [-0.3497669 ],
                [-0.35025731],
                [-0.35231279],
                [-0.35329935],
                [-0.35478794],
                [-0.35399758],
                [-0.34904793],
                [-0.34782878],
                [-0.34592822],
                [-0.34093009],
                [-0.34066085],
                [-0.34244692],
                [-0.34422623],
                [-0.34754532],
                [-0.35311227],
                [-0.3589083 ]]]
            forecastForFeat_i = self.LSTM_Models[i].forecastGiven(lookBackData)
            features_forecasts.append(forecastForFeat_i)
            # for t in timeSteps:
                # curr_forecast = formatFeatureSet(feat_forecast, timestep)
        print("features_forecasts:", features_forecasts)
        forecasted_Power = []



          # answer should be 13 for the above

        #try:
        while (False):
        # for each timestep in timesteps;
            #curr_forecast =  getFeaturesFor[timestep]
            curr_forecast = [ 1.0, 0.27882305,  0.69449111, 0.25765821 , 0.11764706,  0.10740741, 0.28571429 , 0.0,0.82,        0.0,          0.9593099 ,  1.0,
              0.27906977,  0.24418605,  0.24093023 , 0.26   ,     0.24132051 , 0.25139744,
              0.23851726]
            # curr_forecast are all of the features that the lstms have forecasted

            print("curr forecast: ", curr_forecast)
            curr_classification = self.NN.classifySetOf(curr_forecast)
            forecasted_Power.append(curr_classification)
            print("curr class: ", curr_classification)
            x=input()
        # power_forecast.append(self.NN, curr_forecast)
    # display(power_forecast, y)

        # except:
        #     print("closing session.")
        #     self.NN.closeSession()

        self.NN.closeSession()
        return 0

    def train(self):
        # thread each model for training
        # continue training until NN > 95%
        NN_targetAcc = 0.70
        self.NN.train(NN_targetAcc)
        # and loss over all feature models are satisfactory


        if (True):
            for i in range(self.getNumOfFeats()):
                self.LSTM_Models[i].train(target_loss = 0.5)

            # single model testing, not super model testing as that is done in masterTest
        #    self.LSTM_Models[0].test()

        self.masterTest()

    def config(self):
        # initialize the NN
            # selecting network parameters?
        trainingDataPath = self.dataFileTarget # 12 is the most recent data with richer features (EMA)
        self.NN = NN.classicalNeuralNetwork(self.id, trainingDataPath) # configure options for NN ==  dataFileTarget="", LOG_DIR="LSTM_LOG/log_tb/temp", batchSize=144, hiddenSize=256, displaySteps=20):
        x = dict()
        x['temp'] = "temperature"
        x['hum'] = " asdf "

        # initialize the LSTMS
            # couont how many features there are

        for column in self.dataFrame:

            if column != "power_output": # TODO maybe don't include moving averages
                self.countFeats+=1
                curr_lstm = modelBuilder_LSTM.StackedLSTM(dataFrame=self.dataFrame[column], modelName=column)
                curr_lstm.networkParams(column) # can pass in custom configurations Note: necessary to call this function
                self.LSTM_Models.append(curr_lstm)

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

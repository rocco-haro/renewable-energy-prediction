# Rocco Haro
import models.stackedLSTM as modelBuilder_LSTM
import pandas as pd
class renewableModel:
    def __init__(self, _id, dataFileTarget):
        self.id = _id
        self.NN = None
        self.LSTM_Models = []

        self.dataFileTarget = dataFileTarget
        self.dataFrame = self.loadData()
        self.config()

    def loadData(self):
        # Pull all data from CSV file and
        # push into a dataframe for portability.

        df = pd.read_csv(self.dataFileTarget, index_col=0, skiprows=[1])
        df.index = pd.to_datetime(df.index)

        return df


    def masterTest(self):
        # for n number of test:
            # y, testInput = getSample()
            # for each feature in batch:
                #feat_forecast.append(sess.run(self.LSTM_Models[feature], stripFeatureSequence(feature/idx, testInput)))
            # for t in timeSteps:
                # curr_forecast = formatFeatureSet(feat_forecast, timestep)
                # power_forecast.append(self.NN, curr_forecast)
            # display(power_forecast, y)

        return 0

    def train(self):
        # thread each model for training
        # continue training until NN > 95%
        # and loss over all feature models are satisfactory
        self.LSTM_Models[1].train()

        # single model testing, not super model testing as that is done in masterTest
        self.LSTM_Models[1].test()
        self.masterTest()

    def config(self):
        # initialize the NN
            # selecting network parameters?

        # initialize the LSTMS
            # couont how many features there are

        for column in self.dataFrame:
            if column != "power_output":
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
            self.renewableModels.append(renewableModel(i, "models/varyingData/moving/newDataWithTemporalsTEST2.csv"))

        self.renewableModels[0].printID()


if __name__ == "__main__":
    numOfRenewables = 1
    SM = superModel(numOfRenewables)

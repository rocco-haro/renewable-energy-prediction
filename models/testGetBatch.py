import numpy as np
import random
import pandas as pd


def getOneBatch(lookBackSize, ySize, df):
    # Grab a batch-sized batch of data

    arr_lookBack = []
    arr_futureFeatures = []
    sizeof_dataframe = df.shape[0]
    sizeof_dataToPull = ySize + lookBackSize
    # print(sizeof_dataframe)

    # df location to pull data from
    start = int(sizeof_dataframe * np.random.rand())
    while (sizeof_dataframe < start + sizeof_dataToPull):
        start = int(sizeof_dataframe * np.random.rand())

    end_lookBack = start + lookBackSize
    end_actualY = start + sizeof_dataToPull

    for column in df:
        if column != "power_output":
            temp_arr = np.asarray(list(df[column][start:end_lookBack]))
            temp_arr = temp_arr.reshape(1,lookBackSize, 1)
            arr_lookBack.append(temp_arr)

            temp_future = np.asarray(list(df[column][end_lookBack:end_actualY]))
            arr_futureFeatures.append(temp_future)


    arr_actualY = np.asarray(list(df["power_output"][end_lookBack:end_actualY]))
    arr_actualY = arr_actualY.reshape(ySize, 1)


    return arr_lookBack, arr_futureFeatures, arr_actualY

def get_data(lookBackSize, ySize, df):
    """ Format batch of test data"""
    lookBackData, futureFeatures, actual_Y = getOneBatch(lookBackSize, ySize, df)
    print("lookBackData:" + str(np.shape(lookBackData)))
    print(lookBackData)
    print("futureFeatures:" + str(np.shape(futureFeatures)))
    print(futureFeatures)
    print("actual_Y:" + str(np.shape(actual_Y)))
    print(actual_Y)



    # Prepend the column of 1s for bias
    # N, M  = data.shape
    #all_X = np.ones((N, M + 1))
    #all_X[:, 1:] = data

    # Convert into one-hot vectors
    #num_labels = 49
    ############## num_labels = Number of unique power prediction numbers

    # One liner trick!
    #all_Y = np.eye(num_labels)[target]
    return lookBackData, futureFeatures, actual_Y


def loadData():
    # Pull all data from CSV file and
    # push into a dataframe for portability.

    df = pd.read_csv("prod_Data/training_Data12.csv", index_col=0, skiprows=[1])
    df.index = pd.to_datetime(df.index)
    return df

df = loadData()

lookBackData, futureFeatures, actual_Y = get_data(11, 5, df)

killme = [[[0.1441566904],
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

print(np.shape(killme))
print(df.shape)
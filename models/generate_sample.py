import numpy as np
from typing import Optional, Tuple

def getSingleSeriesDataOfSize(maxDataPoints, tag, fileTarget):
    def getStartingPt(numOfPs):
        starting = int(numOfPs * np.random.rand())
        #print("numOfPs: ", numOfPs)
        #print("starting: ", starting)
        while (numOfPs - starting < maxDataPoints):
            starting = int(numOfPs * np.random.rand())
        return starting

    arr = []
    counter = 0
    #print("path: ", tag)
    #print("Return number of :" +str(maxDataPoints))
    filePath = fileTarget+tag
    with open(filePath, 'r') as readF: # TODO removed the tag
        for line in readF:
            if counter == 0:
                numOfPts = int(line)
                counterStart = getStartingPt(numOfPts)
                counter = 0
            #    print("Max data pts: ", numOfPts)
            #    print("Counter start: ", counterStart)
            else:

                if counter >= counterStart and (counter - counterStart) < maxDataPoints:
                #    print("counter: " + str(counter))
                    arr.append(float(line))
                elif (counter - counterStart) >= maxDataPoints:
                    break
            counter+=1

                #break
    #print("Size: ", str(len(arr)))
    if len(arr) != maxDataPoints:
        arr = getSingleSeriesDataOfSize(maxDataPoints, tag, fileTarget)
    print(type(arr))
    print(arr)
    print(len(arr))
    return arr

def generate_sample(fileTarget="", training: Optional[bool] = None, batch_size: int = 1,
                    predict: int = 50, samples: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates data samples.

    :param f: The frequency to use for all time series or None to randomize.
    :param t0: The time offset to use for all time series or None to randomize.
    :param batch_size: The number of time series to generate.
    :param predict: The number of future samples to generate.
    :param samples: The number of past (and current) samples to generate.
    :return: Tuple that contains the past times and values as well as the future times and values. In all outputs,
             each row represents one time series of the batch.
    """
    Fs = 100

    T = np.empty((batch_size, samples))
    Y = np.empty((batch_size, samples))
    FT = np.empty((batch_size, predict))
    FY = np.empty((batch_size, predict))

    _t0 = training
    for i in range(batch_size):
        t = np.arange(0, samples + predict)
        if _t0 is True:
        #     t0 = np.random.rand() * 2 * np.pi
        # else:
        #     t0 = _t0 + i/float(batch_size)
        #
        # freq = f
        # if freq is None:
        #     freq = np.random.rand() * 3.5 + 0.5

            y = np.array(getSingleSeriesDataOfSize(predict+samples, "_TRAIN", fileTarget)) # np.sin(2 * np.pi * freq * (t + t0))
        #    print(np.shape(y))
        #    z = input()
            T[i, :] = t[0:samples]
            Y[i, :] = y[0:samples]

            FT[i, :] = t[samples:samples + predict]
            FY[i, :] = y[samples:samples + predict]
        else:
            y = np.array(getSingleSeriesDataOfSize(predict+samples, "_TEST", fileTarget)) # np.sin(2 * np.pi * freq * (t + t0))

            T[i, :] = t[0:samples]
            Y[i, :] = y[0:samples]

            FT[i, :] = t[samples:samples + predict]
            FY[i, :] = y[samples:samples + predict]

    # print("T: ", T)
    # print("Y: ", Y)
    # print("FT: ", FT)
    # print("FY: ", FY)
    return T, Y, FT, FY


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    # noinspection PyUnresolvedReferences
    import seaborn as sns

    t, y, t_next, y_next = generate_sample("varyingData/moving/temperature", True)#, batch_size=1, samples=300, predict=200)

    n_tests = t.shape[0]
    for i in range(0, n_tests):
        plt.subplot(n_tests, 1, i+1)
        plt.plot(t[i, :], y[i, :])
        plt.plot(np.append(t[i, -1], t_next[i, :]), np.append(y[i, -1], y_next[i, :]), color='red', linestyle=':')

    plt.xlabel('time [t]')
    plt.ylabel('signal')
    plt.show()

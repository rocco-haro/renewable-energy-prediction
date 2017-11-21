# Implementation of a simple MLP network with one hidden layer. Tested on the iris data set.
# Requires: numpy, sklearn>=0.18.1, tensorflow>=1.0

# NOTE: In order to make the code simple, we rewrite x * W_1 + b_1 = x' * W_1'
# where x' = [x | 1] and W_1' is the matrix W_1 appended with a new row with elements b_1's.
# Similarly, for h * W_2 + b_2

# code from https://gist.github.com/vinhkhuc/e53a70f9e5c3f55852b0

import tensorflow as tf
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42
tf.set_random_seed(RANDOM_SEED)


def init_weights(shape):
    """ Weight initialization """
    weights = tf.random_normal(shape, stddev=0.1)
    return tf.Variable(weights)

def forwardprop(X, w_1, w_2):
    """
    Forward-propagation.
    IMPORTANT: yhat is not softmax since TensorFlow's softmax_cross_entropy_with_logits() does that internally.
    """
    h    = tf.nn.sigmoid(tf.matmul(X, w_1))  # The \sigma function
    yhat = tf.matmul(h, w_2)  # The \varphi function
    return yhat

def getDataBatchSizeOf(maxDataPoints, tag):
    def getStartingPt(numOfPs):
        starting = int(numOfPs * np.random.rand())
        #print("numOfPs: ", numOfPs)
        #print("starting: ", starting)
        while (numOfPs - starting < maxDataPoints):


            # TODO fix this back so that its random
            starting = int(numOfPs *  np.random.rand())
        return 1 #starting

    arrX = []
    arrY = []
    counter = 0
    #print("path: ", tag)
    #print("Return number of :" +str(maxDataPoints))
    with open('varyingData/state/state_powerGeneratedWindTurbine_'+tag, 'r') as readF:
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
                    datas = line.split(',')
                    arrX.append([float(x) for x in datas[1:]])
                    arrY.append(int(datas[0]))
                elif (counter - counterStart) >= maxDataPoints:
                    break
            counter+=1

                #break
    #print("Size: ", str(len(arr)))
    # if len(arrX) != maxDataPoints:
    #     arrX, arrY = getSingleSeriesDataOfSize(maxDataPoints, tag)
    return np.array(arrX), np.array(list(arrY))
def get_iris_data():
    """ Read the iris data set and split them into training and test sets """
    iris   = datasets.load_iris()
    data   = iris["data"]
    target = iris["target"]
    data, target = getDataBatchSizeOf(288, 'TRAIN')
    print("data: ", data)
    print("target: ", target)
    #z = input()
    # Prepend the column of 1s for bias
    N, M  = data.shape
    print("N: ", N)
    print("M: ", M)
    all_X = np.ones((N, M + 1))
    all_X[:, 1:] = data

    # Convert into one-hot vectors
    num_labels = 49 # len(np.unique(target)) + 5 # TODO the number of labels NEEDS TO MATCH up to the maximum value returned...
    print("Num labels: ",num_labels)
    print("target shape: ", target.shape)
    #print("[target] :", [target])
    all_Y = np.eye(num_labels)[target]  # One liner trick!
    print("all_Y ", all_Y)
    z=input()
    return train_test_split(all_X, all_Y, test_size=0.5, random_state=RANDOM_SEED)

def main():
    train_X, test_X, train_y, test_y = get_iris_data()

    # Layer's sizes
    x_size = train_X.shape[1]   # Number of input nodes: 4 features and 1 bias
    h_size = 256                # Number of hidden nodes
    y_size = train_y.shape[1]   # Number of outcomes (3 iris flowers)

    # Symbols
    X = tf.placeholder("float", shape=[None, x_size])
    y = tf.placeholder("float", shape=[None, y_size])

    # Weight initializations
    w_1 = init_weights((x_size, h_size))
    w_2 = init_weights((h_size, y_size))

    # Forward propagation
    yhat    = forwardprop(X, w_1, w_2)
    predict = tf.argmax(yhat, axis=1)

    # Backward propagation
    cost    = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=yhat))
    updates = tf.train.GradientDescentOptimizer(0.07).minimize(cost)

    # Run SGD
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    test_accuracy = 0
    epoch = 0
    while (test_accuracy < 0.75):
        # Train with each example
        for i in range(len(train_X)):
            sess.run(updates, feed_dict={X: train_X[i: i + 1], y: train_y[i: i + 1]})

        train_accuracy = np.mean(np.argmax(train_y, axis=1) ==
                                 sess.run(predict, feed_dict={X: train_X, y: train_y}))
        test_accuracy  = np.mean(np.argmax(test_y, axis=1) ==
                                 sess.run(predict, feed_dict={X: test_X, y: test_y}))

        print("Epoch = %d, train accuracy = %.2f%%, test accuracy = %.2f%%"
              % (epoch + 1, 100. * train_accuracy, 100. * test_accuracy))
        epoch+=1
    sess.close()

if __name__ == '__main__':
	main()

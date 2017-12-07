# Implementation of a simple MLP network with one hidden layer. Tested on the iris data set.
# Requires: numpy, sklearn>=0.18.1, tensorflow>=1.0

# NOTE: In order to make the code simple, we rewrite x * W_1 + b_1 = x' * W_1'
# where x' = [x | 1] and W_1' is the matrix W_1 appended with a new row with elements b_1's.
# Similarly, for h * W_2 + b_2

# code from https://gist.github.com/vinhkhuc/e53a70f9e5c3f55852b0

import tensorflow as tf
import numpy as np
import pandas as pd
import random
from pathlib import Path
from sklearn import datasets
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42
tf.set_random_seed(RANDOM_SEED)

class neuralNetwork:
    def __init__(self, dataFileTarget="", LOG_DIR="LSTM_LOG/log_tb/temp", batchSize=144, hiddenSize=256, displaySteps=20):
        self.dataFileTarget = dataFileTarget
        self.LOG_DIR = LOG_DIR
        self.batchSize = batchSize
        self.hiddenSize = hiddenSize
        self.displaySteps = displaySteps

        self.df = self.loadFile()
        
        self.train()

    def loadFile(self):
        # File path to the input data
        trainLink = Path.cwd().parent.parent.joinpath(str('prod_Data/' + self.dataFileTarget))

        # Load all data into dataframe
        df = pd.read_csv(trainLink, index_col=0, skiprows=[1])
        df.index = pd.to_datetime(df.index)

        return df

    def init_weights(self, shape):
        """ Weight initialization """
        weights = tf.random_normal(shape, stddev=0.1)
        return tf.Variable(weights)

    def forwardprop(self, X, w_1, w_2):
        """
        Forward-propagation.
        IMPORTANT: yhat is not softmax since TensorFlow's softmax_cross_entropy_with_logits() does that internally.
        """
        h    = tf.nn.sigmoid(tf.matmul(X, w_1))  # Sigmoid function [0, 1]
        yhat = tf.matmul(h, w_2)  # The \varphi function
        return yhat

    def getOneBatch(self):
        arrX = []
        arrY = []

        # Randomly select a day
        start = random.randrange(0, self.batchSize)
        while(start > self.df.shape[0] - self.batchSize):   # Out of bounds exception
            start = random.randrange(0, self.batchSize)
        end = start + self.batchSize

        # Pull the day's data
        arrX = self.df.ix[start:end,'temp':].values
        arrY = self.df['power_output'][start:end].astype(int)

        return np.array(arrX), np.array(arrY)

    def get_data(self):
        """  """
        data, target = self.getOneBatch()

        # Prepend the column of 1s for bias
        N, M  = data.shape
        all_X = np.ones((N, M + 1))
        all_X[:, 1:] = data

        # Convert into one-hot vectors
        num_labels = 49 
        ############## num_labels = Number of unique power prediction numbers
      
        # One liner trick!
        all_Y = np.eye(num_labels)[target]  
        return train_test_split(all_X, all_Y, test_size=0.2, random_state=RANDOM_SEED)

    def train(self):
        train_X, test_X, train_y, test_y = self.get_data()

        # Layer's sizes
        x_size = train_X.shape[1]   # Number of input nodes
        h_size = self.hiddenSize    # Number of hidden nodes
        y_size = train_y.shape[1]   # Number of outcomes

        # Symbols
        X = tf.placeholder("float", shape=[None, x_size])
        y = tf.placeholder("float", shape=[None, y_size])

        # Weight initializations
        w_1 = self.init_weights((x_size, h_size))
        w_2 = self.init_weights((h_size, y_size))

        # Forward propagation
        yhat    = self.forwardprop(X, w_1, w_2)
        predict = tf.argmax(yhat, axis=1)

        # Backward propagation
        cost    = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=yhat))
        updates = tf.train.GradientDescentOptimizer(0.07).minimize(cost)

        # Run SGD
        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)
        test_accuracy = 0
        ema_train_acc = 0
        ema_test_acc = 0
        epoch = 0

        # Run Tensorboard
        writer = tf.summary.FileWriter(self.LOG_DIR, sess.graph)

        while (ema_test_acc < 0.99):
            train_X, test_X, train_y, test_y = self.get_data()
            # Train with each example
            for i in range(len(train_X)):
                sess.run(updates, feed_dict={X: train_X[i: i + 1], y: train_y[i: i + 1]})

            train_accuracy = np.mean(np.argmax(train_y, axis=1) ==
                                     sess.run(predict, feed_dict={X: train_X, y: train_y}))
            test_accuracy  = np.mean(np.argmax(test_y, axis=1) ==
                                     sess.run(predict, feed_dict={X: test_X, y: test_y}))
            
            ema_train_acc = ( (10*train_accuracy) + (ema_train_acc * 90) ) / 100
            ema_test_acc = ( (10*test_accuracy) + (ema_test_acc * 90) ) / 100

            test_sum = tf.Summary(value=[tf.Summary.Value(tag="test_accuracy", simple_value=test_accuracy),])
            writer.add_summary(test_sum, epoch)
            #train_summ = tf.Summary(value=[tf.Summary.Value(tag="train_accuracy", simple_value=train_accuracy),])
            #writer.add_summary(train_summ, epoch)
            writer.flush()

            print("Epoch = %d, train accuracy = %.2f%%, test accuracy = %.2f%% | %.2f%%"
                  % (epoch + 1, 100. * train_accuracy, 100. * test_accuracy, 100. * ema_test_acc))
            epoch+=1
        sess.close()

neuralNetwork('training_Data.csv')
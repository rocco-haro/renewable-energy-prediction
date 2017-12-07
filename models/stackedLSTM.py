"""
A Recurrent Neural Network (GRU) implementation example using TensorFlow library.
Inspired by https://github.com/aymericdamien/TensorFlow-Examples/ and http://mourafiq.com/2016/05/15/predicting-sequences-using-rnn-in-tensorflow.html
"""

from __future__ import print_function
#from models.generate_sample import generate_sample

import numpy as np
import matplotlib.pyplot as plt
# noinspection PyUnresolvedReferences
import seaborn as sns
import os

import tensorflow as tf
from tensorflow.contrib import rnn
from typing import Optional, Tuple

class StackedLSTM:
    def __init__(self, dataFrame=None, modelName="_Unnamed!", learning_rate=0.00025, training_iters=1000000, training_iter_step_down_every=250000, batch_size=40, display_step=100):
        """ dataFileTarget="",modelName, learning_rate=0.005,training_iters = 1000000,training_iter_step_down_every = 250000, batch_size = 10 , display_step = 100
        """
        #self.dataFileTarget = dataFileTarget
        self.dataFrame = dataFrame
        self.modelName = modelName
        self.learning_rate = learning_rate
        self.training_iters = training_iters
        self.training_iter_step_down_every = training_iter_step_down_every
        self.batch_size = batch_size
        self.display_step = display_step

        self.NetworkParametersSet = False

    def networkParams(self,ID, n_input = 1,n_steps = 20, n_hidden= 2, n_outputs = 5 , n_layers = 2, loading=False  ):
        # Network Parameters
        self.ID = ID
        self.n_input = n_input # input is sin(x), a scalar
        self.n_steps = n_steps  # historical time steps look back
        self.n_hidden = n_hidden  # hidden layer num of features
        self.n_outputs = n_outputs  # output is a series of sin(x+...)
        self.n_layers = n_layers  # number of stacked GRU layers

        # tf Graph input
        self.lr = tf.placeholder(tf.float32, [])
        self.x = tf.placeholder(tf.float32, [None, n_steps, n_input], name="x")
        self.y = tf.placeholder(tf.float32, [None, n_outputs])

        # Define weights
        self.weights = {
            'out': tf.Variable(tf.truncated_normal([self.n_hidden, self.n_outputs], stddev=1.0))
        }
        self.biases = {
            'out': tf.Variable(tf.truncated_normal([self.n_outputs], stddev=0.1))
        }
        if (not loading):
            # Define the GRU cells
            with tf.variable_scope(str(self.ID)):
                gru_cells = [rnn.GRUCell(n_hidden) for _ in range(n_layers)]
                self.stacked_lstm = rnn.MultiRNNCell(gru_cells)
                self.outputs, self.states = tf.nn.dynamic_rnn(self.stacked_lstm, inputs=self.x, dtype=tf.float32, time_major=False)

                h = tf.transpose(self.outputs, [1, 0, 2])
                self.pred = tf.nn.bias_add(tf.matmul(h[-1], self.weights['out']), self.biases['out'], name="pred")

                # Define loss (Euclidean distance) and optimizer
                individual_losses = tf.reduce_sum(tf.squared_difference(self.pred, self.y), reduction_indices=1)
                self.loss = tf.reduce_mean(individual_losses)
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

                self.NetworkParametersSet = True

    def getDataOfSize(self, numRows):
        #getSingleSeriesDataOfSize(predict+samples, "_TRAIN", fileTarget)

        # Number of rows
        sizeof_dataframe = self.dataFrame.shape[0]
        # print(sizeof_dataframe)

        # Starting and ending row numbers
        start = int(sizeof_dataframe * np.random.rand())
        while (sizeof_dataframe < start + numRows):
            start = int(sizeof_dataframe * np.random.rand())
        end = start + numRows

        # Pull the day's data
        arr = list(self.dataFrame[start:end])
        # /nt(len(arr))
        return arr

    def generateSubset(self, training: Optional[bool] = None, batch_size: int = 1, predict: int = 50, samples: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

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

                y = np.array(self.getDataOfSize(predict+samples)) # np.sin(2 * np.pi * freq * (t + t0))
            #    print(np.shape(y))
            #    z = input()
                T[i, :] = t[0:samples]
                Y[i, :] = y[0:samples]

                FT[i, :] = t[samples:samples + predict]
                FY[i, :] = y[samples:samples + predict]
            else:
                y = np.array(self.getDataOfSize(predict+samples)) # np.sin(2 * np.pi * freq * (t + t0))

                T[i, :] = t[0:samples]
                Y[i, :] = y[0:samples]

                FT[i, :] = t[samples:samples + predict]
                FY[i, :] = y[samples:samples + predict]

        # print("T: ", T)
        # print("Y: ", Y)
        # print("FT: ", FT)
        # print("FY: ", FY)
        return T, Y, FT, FY


    def train(self, target_loss=0.005):
        if (self.NetworkParametersSet and self.dataFrame is not None):
            # Initializing the variables
            init = tf.global_variables_initializer()

            # add ops to save and restore all variables
            saver = tf.train.Saver()
            # Launch the graph
            with tf.Session() as sess:
                sess.run(init)
                step = 1

                training_loss_value = float('+Inf')
                testing_loss_value = float('+Inf')
                # Keep training until reach max iterations
                while (step * self.batch_size < self.training_iters) and (testing_loss_value > target_loss):
                    current_learning_rate = self.learning_rate
                    current_learning_rate *= 0.1 ** ((step * self.batch_size) // self.training_iter_step_down_every)

                    _, batch_x, __, batch_y = self.generateSubset(training=True, batch_size=self.batch_size, samples=self.n_steps, predict=self.n_outputs)
                    batch_x = batch_x.reshape((self.batch_size, self.n_steps, self.n_input))
                    batch_y = batch_y.reshape((self.batch_size,self. n_outputs))

                    # Run optimization op (backprop)
                    sess.run(self.optimizer, feed_dict={self.x: batch_x, self.y: batch_y, self.lr: current_learning_rate})
                    if step % self.display_step == 0:
                        # Calculate batch loss
                        training_loss_value = sess.run(self.loss, feed_dict={self.x: batch_x, self.y: batch_y})

                        # Run on test data
                        _, batch_x, __, batch_y = self.generateSubset(training=False, batch_size=self.batch_size, samples=self.n_steps, predict=self.n_outputs)
                        batch_x = batch_x.reshape((self.batch_size, self.n_steps, self.n_input))
                        batch_y = batch_y.reshape((self.batch_size,self. n_outputs))
                        testing_loss_value = sess.run(self.loss, feed_dict={self.x: batch_x, self.y: batch_y})

                        print("Iter " + str(step * self.batch_size) + ", Training Loss= " +
                              "{:.6f} Testing loss= {:.6f}".format(training_loss_value, testing_loss_value))
                    step += 1
                print("Optimization Finished!")
                targetSavePath = "models/savedModels/"+self.modelName #+"/"+self.modelName # need the underscore
                if (not os.path.isdir(targetSavePath)):
                    os.mkdir(targetSavePath)
                save_path = saver.save(sess, targetSavePath+"/"+self.modelName)

                print("Saving to : " + save_path)
        else:
            print("*** stackedLSTM says: Network Parameters are not set or no dataFile target given.")

    def restoreModel(self, targetModel):
        # Test the prediction

        with tf.Session() as sess:
            saver = tf.train.import_meta_graph('savedModels/'+targetModel+'/'+targetModel+'.meta')
            saver.restore(sess, tf.train.latest_checkpoint('savedModels/'+ targetModel+'/./'))
            graph = tf.get_default_graph()
            self.pred = graph.get_tensor_by_name("pred:0")
            _x = graph.get_tensor_by_name("x:0")
            # self.n_steps = graph.get_tensor_by_name("n_steps:0")
            # self.n_outputs = graph.get_tensor_by_name("n_outputs:0")
            # self.n_input = graph.get_tensor_by_name("n_input:0")
            # self.x = graph.get_tensor_by_name("x:0")
            n_tests = 3
            for i in range(1, n_tests + 1):
                plt.subplot(n_tests, 1, i)
                t, y, next_t, expected_y = generate_sample(self.dataFileTarget,training=False, samples=self.n_steps, predict=self.n_outputs)
                test_input = y.reshape((1, self.n_steps, self.n_input))
                print("test: ", test_input)
                prediction = sess.run(self.pred, feed_dict={_x: test_input})
                # remove the batch size dimensions
                t = t.squeeze()
                y = y.squeeze()
                next_t = next_t.squeeze()
                prediction = prediction.squeeze()

                plt.plot(t, y, color='black')
                plt.plot(np.append(t[-1], next_t), np.append(y[-1], expected_y), color='green', linestyle=':')
                plt.plot(np.append(t[-1], next_t), np.append(y[-1], prediction), color='red')
                plt.ylim([0,0.6])
                plt.xlabel('time [t]')
                plt.ylabel(self.dataFileTarget)

            plt.show()

    def forecastGiven(self, lookBackData):
        init = tf.global_variables_initializer()
        n_tests = 1
        with tf.Session() as sess:
            sess.run(init)
            for i in range(1, n_tests + 1):
                plt.subplot(n_tests, 1, i)
                t, y, next_t, expected_y = generate_sample(self.dataFileTarget, training=False, samples=self.n_steps, predict=self.n_outputs)

                test_input = y.reshape((1, self.n_steps, self.n_input))
            #    print("test_input: ", test_input)
                prediction = sess.run(self.pred, feed_dict={self.x: lookBackData})
            #    print("prediction: ", prediction)
                return prediction
                # remove the batch size dimensions
                t = t.squeeze()
                y = y.squeeze()
                next_t = next_t.squeeze()
                prediction = prediction.squeeze()

                plt.plot(t, y, color='black')
                plt.plot(np.append(t[-1], next_t), np.append(y[-1], expected_y), color='green', linestyle=':')
                plt.plot(np.append(t[-1], next_t), np.append(y[-1], prediction), color='red')
                plt.ylim([-1,1])
                plt.xlabel('time [t]')
                plt.ylabel('temp')

            plt.show()

    def test(self, n_tests=3):
        # Test the prediction
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)
            for i in range(1, n_tests + 1):
                plt.subplot(n_tests, 1, i)
                t, y, next_t, expected_y = generate_sample(self.dataFileTarget, training=False, samples=self.n_steps, predict=self.n_outputs)

                test_input = y.reshape((1, self.n_steps, self.n_input))
                prediction = sess.run(self.pred, feed_dict={self.x: test_input})

                # remove the batch size dimensions
                t = t.squeeze()
                y = y.squeeze()
                next_t = next_t.squeeze()
                prediction = prediction.squeeze()

                plt.plot(t, y, color='black')
                plt.plot(np.append(t[-1], next_t), np.append(y[-1], expected_y), color='green', linestyle=':')
                plt.plot(np.append(t[-1], next_t), np.append(y[-1], prediction), color='red')
                plt.ylim([-1,1])
                plt.xlabel('time [t]')
                plt.ylabel('temp')

            plt.show()

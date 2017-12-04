"""
A Recurrent Neural Network (GRU) implementation example using TensorFlow library.
Inspired by https://github.com/aymericdamien/TensorFlow-Examples/ and http://mourafiq.com/2016/05/15/predicting-sequences-using-rnn-in-tensorflow.html
"""

from __future__ import print_function

from generate_sample import generate_sample

import numpy as np
import matplotlib.pyplot as plt
# noinspection PyUnresolvedReferences
import seaborn as sns

import tensorflow as tf
from tensorflow.contrib import rnn

class StackedLSTM:
    def __init__(self,dataFileTarget="",learning_rate=0.05,training_iters = 1000000,training_iter_step_down_every = 250000, batch_size = 10 , display_step = 100  ):
        self.dataFileTarget = dataFileTarget
        self.learning_rate = learning_rate
        self.training_iters = training_iters
        self.training_iter_step_down_every = training_iter_step_down_every
        self.batch_size = batch_size
        self.display_step = display_step

        self.NetworkParametersSet = False

    def networkParams(self, n_input = 1,n_steps = 20, n_hidden= 20, n_outputs = 2 , n_layers = 5, loading=False  ):
        # Network Parameters
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

    def train(self, target_loss=0.005):
        if (self.NetworkParametersSet and self.dataFileTarget != ""):
            # Initializing the variables
            init = tf.global_variables_initializer()

            # add ops to save and restore all variables
            saver = tf.train.Saver()
            # Launch the graph
            with tf.Session() as sess:
                sess.run(init)
                step = 1

                loss_value = float('+Inf')

                # Keep training until reach max iterations
                while (step * self.batch_size < self.training_iters) and (loss_value > target_loss):
                    current_learning_rate = self.learning_rate
                    current_learning_rate *= 0.1 ** ((step * self.batch_size) // self.training_iter_step_down_every)

                    _, batch_x, __, batch_y = generate_sample(self.dataFileTarget, f=None, t0=None, batch_size=self.batch_size, samples=self.n_steps,
                                                              predict=self.n_outputs)

                    batch_x = batch_x.reshape((self.batch_size, self.n_steps, self.n_input))
                    batch_y = batch_y.reshape((self.batch_size,self. n_outputs))

                    # Run optimization op (backprop)
                    sess.run(self.optimizer, feed_dict={self.x: batch_x, self.y: batch_y, self.lr: current_learning_rate})
                    if step % self.display_step == 0:
                        # Calculate batch loss
                        loss_value = sess.run(self.loss, feed_dict={self.x: batch_x, self.y: batch_y})
                        print("Iter " + str(step * self.batch_size) + ", Minibatch Loss= " +
                              "{:.6f}".format(loss_value))
                    step += 1
                print("Optimization Finished!")

                save_path = saver.save(sess, "savedModels/model")
                print("Saving to : " + save_path)
        else:
            print("*** stackedLSTM says: Network Parameters are not set or no dataFile target given.")

    def restoreModel(self, targetModel):
        # Test the prediction

        with tf.Session() as sess:
            saver = tf.train.import_meta_graph("savedModels/"+targetModel+'.meta')
            saver.restore(sess, tf.train.latest_checkpoint('savedModels/./'))
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
                t, y, next_t, expected_y = generate_sample(self.dataFileTarget, f=i, t0=True, samples=self.n_steps, predict=self.n_outputs)

                test_input = y.reshape((1, self.n_steps, self.n_input))
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

    def test(self, n_tests=3):
        # Test the prediction
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)
            for i in range(1, n_tests + 1):
                plt.subplot(n_tests, 1, i)
                t, y, next_t, expected_y = generate_sample(self.dataFileTarget, f=i, t0=True, samples=self.n_steps, predict=self.n_outputs)

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
                plt.ylim([0,0.6])
                plt.xlabel('time [t]')
                plt.ylabel(self.dataFileTarget)

            plt.show()

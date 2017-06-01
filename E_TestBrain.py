from __future__ import division

import gym
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt


import tensorflow.contrib.slim as slim



class E_Experience():
    def __init__(self, state0= None, action0= None, reward0= None, state1= None):
        self.state0 = state0
        self.action0 = action0
        self.reward0 = reward0
        self.state1 = state1

class E_Window():
    def __init__(self, size, minsize):
        self.v = []
        self.size = size
        self.minsize = minsize
        self.sum = 0

    def add(self, x):
        self.v.append(x)
        self.sum += x

        if len(self.v) > self.size:
            xold = self.v[0]
            self.sum -= xold
            del self.v[0]

    def get_average(self):
        if len(self.v) < self.minsize: return -1
        else: return self.sum / len(self.v)

    def reset(self):
        self.v = []
        self.sum = 0






class Q_Network():
    def __init__(self, num_inputs, num_actions):
        #These lines establish the feed-forward part of the network used to choose actions
        self.inputs = tf.placeholder(shape=[None,num_inputs],dtype=tf.float32)
        self.Temp = tf.placeholder(shape=None,dtype=tf.float32)
        self.dropout = tf.placeholder(shape=None,dtype=tf.float32)

        hidden = slim.fully_connected(self.inputs,64,activation_fn=tf.nn.relu,biases_initializer=None)
        hidden = slim.fully_connected(hidden,64,activation_fn=tf.nn.relu,biases_initializer=None)
        hidden = slim.fully_connected(hidden,64,activation_fn=tf.nn.relu,biases_initializer=None)
        hidden = slim.dropout(hidden,self.dropout)
        self.Q_out = slim.fully_connected(hidden,num_actions,activation_fn=None,biases_initializer=None)

        self.predict = tf.argmax(self.Q_out,1)

        #Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        self.Y = tf.placeholder(shape=[None],dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.Y,num_actions,dtype=tf.float32)

        self.loss = tf.reduce_sum(tf.abs(self.Q_out - self.actions_onehot), reduction_indices=1)

        # self.nextQ = tf.placeholder(shape=[None],dtype=tf.float32)
        # loss = tf.reduce_sum(tf.square(self.nextQ - self.Q))
        trainer = tf.train.AdamOptimizer(learning_rate=0.1)
        self.updateModel = trainer.minimize(self.loss)

class E_Brain():
    def __init__(self, num_states, num_actions):

        self.temporal_window = 1
        self.experience_size = 30000
        self.start_learn_threshold = 300
        self.gamma = 0.8


        self.learning_steps_total = 10000
        self.learning_steps_burnin = 3000
        self.epsilon_min = 0.05
        self.epsilon_test_time = 0.01

        self.random_action_distribution = []


        self.net_inputs = num_states * self.temporal_window + num_actions * self.temporal_window + num_states
        self.num_states = num_states
        self.num_actions = num_actions

        self.window_size = max([self.temporal_window, 2])
        self.state_window = [None] * self.window_size
        self.action_window = [None] * self.window_size
        self.reward_window = [None] * self.window_size
        self.net_window = [None] * self.window_size

        self.batch_size = 64

        tf.reset_default_graph()
        self.value_net = Q_Network(self.net_inputs, num_actions)

        #Initialize TF Variables
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)



        self.experience = []


        #Various houskeepgin variables
        self.age = 0
        self.forward_passes = 0
        self.epsilon = 1.0
        self.latest_reward = 0
        self.last_input_array = []
        self.average_reward_window = E_Window(1000, 10)
        self.average_loss_window = E_Window(1000, 10)
        self.learning = True

    def random_action(self):
        if len(self.random_action_distribution) == 0:
            return random.randint(0, self.num_actions-1)
        else:
            p = random.random()
            cumprob = 0.0
            for k in range(self.num_actions):
                cumprob += self.random_action_distribution[k]
                if p < cumprob:
                    return k


    def policy(self, s):
        #get predicted action index and value
        action_values, maxk = self.sess.run([self.value_net.Q_out, self.value_net.predict],feed_dict={self.value_net.inputs:[s], self.value_net.dropout:1.0})
        maxk = maxk[0]
        maxval = action_values[0][maxk]

        return maxk, maxval

    def getNetInput(self, xt):
        w = []
        w = np.concatenate((w, xt))

        n = self.window_size


        for k in range(self.temporal_window):

            w = np.concatenate((w , self.state_window[n-1-k]))
            actionlofk = [0.0] * self.num_actions
            actionlofk[self.action_window[n-1-k]] = 1.0 * self.num_states
            w = np.concatenate((w , actionlofk))
        return w

    def Forward(self, input_array):
        self.forward_passes += 1
        self.last_input_array = input_array



        if self.forward_passes > self.temporal_window:
            net_input = self.getNetInput(input_array)

            if(self.learning):
                self.epsilon = min(1.0, max(self.epsilon_min, 1.0-(self.age - self.learning_steps_burnin)/(self.learning_steps_total - self.learning_steps_burnin)))
            else:
                self.epsilon = self.epsilon_test_time

            rf = random.random()
            if rf < self.epsilon:
                action = self.random_action()
            else:
                action, val = self.policy(net_input)
        else:
            net_input = []
            action = self.random_action()


        del self.net_window[0]
        self.net_window.append(net_input)

        del self.state_window[0]
        self.state_window.append(input_array)

        del self.action_window[0]
        self.action_window.append(action)


        return action


    def Backward(self, reward):
        self.latest_reward = reward
        self.average_reward_window.add(reward)

        del self.reward_window[0]
        self.reward_window.append(reward)

        if not self.learning:
            return

        self.age += 1

        if self.forward_passes > self.temporal_window + 1:
            e = E_Experience()
            n = self.window_size

            e.state0 = self.net_window[n-2]
            e.action0 = self.action_window[n-2]
            e.reward0 = self.reward_window[n-2]
            e.state1 = self.net_window[n-1]

            if len(self.experience) < self.experience_size:
                self.experience.append(e)
            else:
                ri = random.randint(0, self.experience_size-1)
                self.experience[ri] = e

        if len(self.experience) > self.start_learn_threshold:
            avcost = 0.0

            for k in range(self.batch_size):
                re = random.randint(0, len(self.experience)-1)
                e = self.experience[re]
                x = np.reshape(e.state0, (1, self.net_inputs))
                maxact, maxval = self.policy(e.state1)
                r = e.reward0 + self.gamma * maxval

                y = [0.0] * self.num_actions
                y[e.action0] = r

                loss, _ = self.sess.run([self.value_net.loss, self.value_net.updateModel],feed_dict={self.value_net.inputs:x, self.value_net.dropout:1.0, self.value_net.Y:y})
                avcost += loss
            avcost = avcost / self.batch_size
            self.average_loss_window.add(avcost)

    def GetLog(self):
        log = "Epsilon : " + str(self.epsilon) + "    age : " + str(self.age) +  "  smooth -ish reward : " + str(self.average_reward_window.get_average()) + "   Loss : " + str(self.average_loss_window.get_average())


        return log

import tensorflow as tf
import numpy as np
import random
import gym
import math


class E_PolicyGradientBrain():
    def __init__(self, NUM_STATE, NUM_ACTION):
        self.NUM_STATE = NUM_STATE
        self.NUM_ACTION = NUM_ACTION

        #Experience Buffer
        self.states = []
        self.actions = []
        self.advantages = []
        # self.transitions = []
        self.update_vals = []


        #Initialize Graph and Session
        self.pl_calculated, self.pl_state, self.pl_actions, self.pl_advantages, self.pl_optimizer= self.policy_gradient()
        self.vl_calculated, self.vl_state, self.vl_newvals, self.vl_optimizer, self.vl_loss = self.value_gradient()

        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())

    def Reset(self):
        self.states = []
        self.actions = []
        self.advantages = []
        # self.transitions = []
        self.update_vals = []



    def softmax(x):
        #softmax
        e_x = np.exp(x - np.max(x))
        out = e_x / e_x.sum()
        return out

    def policy_gradient(self):
        with tf.variable_scope("policy"):

            params = tf.get_variable("policy_parameters",[self.NUM_STATE,self.NUM_ACTION])

            state = tf.placeholder("float",[None,self.NUM_STATE])
            actions = tf.placeholder("float",[None,self.NUM_ACTION])
            advantages = tf.placeholder("float",[None,1])



            linear = tf.matmul(state,params)
            probabilities = tf.nn.softmax(linear)

            good_probabilities = tf.reduce_sum(tf.multiply(probabilities, actions),reduction_indices=[1])
            eligibility = tf.log(good_probabilities) * advantages

            loss = -tf.reduce_sum(eligibility)
            optimizer = tf.train.AdamOptimizer(0.0001).minimize(loss)

            return probabilities, state, actions, advantages, optimizer

    def value_gradient(self):
        with tf.variable_scope("value"):
            state = tf.placeholder("float",[None,self.NUM_STATE])
            newvals = tf.placeholder("float",[None,1])

            w1 = tf.get_variable("w1",[self.NUM_STATE,50])
            b1 = tf.get_variable("b1",[50])
            h1 = tf.nn.relu(tf.matmul(state,w1) + b1)

            w2 = tf.get_variable("w2",[50,50])
            b2 = tf.get_variable("b2",[50])
            h2 = tf.nn.relu(tf.matmul(h1,w2) + b2)

            w3 = tf.get_variable("w3",[50,1])
            b3 = tf.get_variable("b3",[1])
            calculated = tf.matmul(h2,w3) + b3


            diffs = calculated - newvals

            loss = tf.nn.l2_loss(diffs)
            optimizer = tf.train.AdamOptimizer(0.0001).minimize(loss)

            return calculated, state, newvals, optimizer, loss


    def Forward(self, state, epsilon = 0.1):
        obs_vector = np.expand_dims(state, axis=0)

        probs = self.sess.run(self.pl_calculated,feed_dict={self.pl_state: obs_vector})

        # print(probs)
        idx = np.argmax(probs)

        if random.uniform(0,1) > epsilon:
            action = idx
        else:
            action = random.randint(0, self.NUM_ACTION-1)





        # record the transition
        self.states.append(state)
        actionblank = np.zeros(self.NUM_ACTION)
        actionblank[action] = 1
        self.actions.append(actionblank)


        return action

    def Backward(self, transitions):

        for index, trans in enumerate(transitions):
            obs, action, reward = trans

            # calculate discounted monte-carlo return
            future_reward = 0
            future_transitions = len(transitions) - index
            decrease = 1
            for index2 in range(future_transitions):
                future_reward += transitions[(index2) + index][2] * decrease
                decrease = decrease * 0.97
            obs_vector = np.expand_dims(obs, axis=0)
            currentval = self.sess.run(self.vl_calculated,feed_dict={self.vl_state: obs_vector})[0][0]

            # advantage: how much better was this action than normal
            self.advantages.append(future_reward - currentval)

            # update the value function towards new return
            self.update_vals.append(future_reward)

        # update value function
        update_vals_vector = np.expand_dims(self.update_vals, axis=1)


        # print(update_vals_vector.shape)
        # print(np.array(self.states).shape)

        self.sess.run(self.vl_optimizer, feed_dict={self.vl_state: self.states, self.vl_newvals: update_vals_vector})
        # real_vl_loss = sess.run(vl_loss, feed_dict={vl_state: states, vl_newvals: update_vals_vector})

        advantages_vector = np.expand_dims(self.advantages, axis=1)
        self.sess.run(self.pl_optimizer, feed_dict={self.pl_state: self.states, self.pl_advantages: advantages_vector, self.pl_actions: self.actions})

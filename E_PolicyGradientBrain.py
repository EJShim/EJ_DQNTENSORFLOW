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
        self.transitions = []
        self.update_vals = []


        #Initialize Graph and Session
        self.pl_calculated, self.pl_state, self.pl_actions, self.pl_advantages, self.pl_optimizer = self.policy_gradient()
        self.vl_calculated, self.vl_state, self.vl_newvals, self.vl_optimizer, self.vl_loss = self.value_gradient()

        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())



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
            optimizer = tf.train.AdamOptimizer(0.01).minimize(loss)

            return probabilities, state, actions, advantages, optimizer

    def value_gradient(self):
        with tf.variable_scope("value"):
            state = tf.placeholder("float",[None,self.NUM_STATE])
            newvals = tf.placeholder("float",[None,1])

            w1 = tf.get_variable("w1",[self.NUM_STATE,10])
            b1 = tf.get_variable("b1",[10])
            h1 = tf.nn.relu(tf.matmul(state,w1) + b1)

            w2 = tf.get_variable("w2",[10,1])
            b2 = tf.get_variable("b2",[1])
            calculated = tf.matmul(h1,w2) + b2

            diffs = calculated - newvals
            loss = tf.nn.l2_loss(diffs)
            optimizer = tf.train.AdamOptimizer(0.1).minimize(loss)

            return calculated, state, newvals, optimizer, loss

    def Forward(self, state):
        obs_vector = np.expand_dims(state, axis=0)
        probs = self.sess.run(self.pl_calculated,feed_dict={self.pl_state: obs_vector})

        if random.uniform(0,1) < probs[0][0]:
            action = 0
        else:
            action = 1


        # record the transition
        self.states.append(state)
        actionblank = np.zeros(2)
        actionblank[action] = 1
        self.actions.append(actionblank)


        return action

    def Backward(self, state0, action, reward):

        #Backward From Here
        self.transitions.append((state0, action, reward))

        for index, trans in enumerate(self.transitions):
            obs, action, reward = trans

            # calculate discounted monte-carlo return
            future_reward = 0
            future_transitions = len(self.transitions) - index
            decrease = 1
            for index2 in range(future_transitions):
                future_reward += self.transitions[(index2) + index][2] * decrease
                decrease = decrease * 0.97
            obs_vector = np.expand_dims(obs, axis=0)
            currentval = self.sess.run(self.vl_calculated,feed_dict={self.vl_state: obs_vector})[0][0]

            # advantage: how much better was this action than normal
            self.advantages.append(future_reward - currentval)

            # update the value function towards new return
            self.update_vals.append(future_reward)

        # update value function
        update_vals_vector = np.expand_dims(self.update_vals, axis=1)
        self.sess.run(self.vl_optimizer, feed_dict={self.vl_state: self.states, self.vl_newvals: update_vals_vector})
        # real_vl_loss = sess.run(vl_loss, feed_dict={vl_state: states, vl_newvals: update_vals_vector})

        advantages_vector = np.expand_dims(self.advantages, axis=1)
        self.sess.run(self.pl_optimizer, feed_dict={self.pl_state: self.states, self.pl_advantages: advantages_vector, self.pl_actions: self.actions})








#Main FUnction
env = gym.make('CartPole-v0')
brain = E_PolicyGradientBrain(4, 2)

num_epiosdes = 2000
max_step = 999


for i in range(num_epiosdes):

    state = env.reset()
    done = False

    rewardsAll = 0.0

    for j in range(max_step):
        action = brain.Forward(state)
        state1, reward, done, info = env.step(action)
        #Backward
        brain.Backward(state, action, reward)

        rewardsAll += reward

        if done : break


        state = state1


    print(rewardsAll)

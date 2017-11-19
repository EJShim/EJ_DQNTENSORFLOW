from __future__ import division

import gym
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt


import tensorflow.contrib.slim as slim

class experience_buffer():
    def __init__(self, buffer_size = 30000):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self,experience):
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:(len(experience)+len(self.buffer))-self.buffer_size] = []
        self.buffer.extend(experience)

    def sample(self,size):
        return np.reshape(np.array(random.sample(self.buffer,size)),[size,5])


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
        self.Q_dist = tf.nn.softmax(self.Q_out/self.Temp)


        self.nextQ = tf.placeholder(shape=[None, num_actions],dtype=tf.float32)
        self.loss = tf.losses.mean_squared_error(self.nextQ, self.Q_out)
        # self.loss = tf.reduce_sum(tf.square(self.nextQ - self.predict))
        trainer = tf.train.AdamOptimizer(learning_rate=0.005)
        self.updateModel = trainer.minimize(self.loss)




class E_Agent():
    def __init__(self, num_inputs, num_actions):
        # self.env = gym.make('CartPole-v0')

        self.num_actions = num_actions
        # Set learning parameters
        self.exploration = "e-greedy" #Exploration method. Choose between: greedy, random, e-greedy, boltzmann, bayesian.
        self.disFact = .99 #Discount factor.
        self.num_episodes = 10000 #Total number of episodes to train network for.
        self.tau = 1.0 #Amount to update target network at each step.
        self.batch_size = 10 #Size of training batch
        self.startE = 1 #Starting chance of random action
        self.endE = 0.1 #Final chance of random action
        self.epsilon = self.startE

        self.anneling_steps = 200000 #How many steps of training to reduce startE to endE.
        self.pre_train_steps = 500 #Number of steps used before training updates begin.
        self.current_steps = 0

        #Experience
        self.buffer = experience_buffer()
        self.n_actions = num_actions

        #Initialize Graph
        self.InitGraph(num_inputs, num_actions)

        self.loss = []

        #Run Training And Save Weights
        # self.RunTraining()
        # self.SaveWeights()

        # self.RestoreWeights()
        # self.Test()
        #
        # self.sess.close()

    def RandomAction(self):
        return random.randint(0, self.n_actions)


    def InitTargetGraph(self, weights):
        total_vars = len(weights)
        op_holder = []
        for idx,var in enumerate(weights[0:total_vars//2]):
            op_holder.append( weights[idx+total_vars//2].assign((var.value()*self.tau) + ((1-self.tau)*weights[idx+total_vars//2].value())) )
        # op_holder = weights
        return op_holder

    def UpdateTargetGraph(self):
        for op in self.targetGraph:
            self.sess.run(op)

    def InitGraph(self, num_inputs, num_actions):
        #Initialize TF Graph
        tf.reset_default_graph()
        self.q_net = Q_Network(num_inputs, num_actions)
        self.target_net = Q_Network(num_inputs, num_actions)

        #Initialize TF Variables
        init = tf.global_variables_initializer()

        self.sess = tf.Session()
        self.sess.run(init)

        #Initialize TF Trainable Variables
        self.targetGraph = self.InitTargetGraph(tf.trainable_variables())
        self.UpdateTargetGraph()



    def Forward(self, state):
        if self.exploration == "greedy":
            #Choose an action with the maximum expected value.
            action , allQ = self.sess.run([self.q_net.predict, self.q_net.Q_out],feed_dict={self.q_net.inputs:[state], self.q_net.dropout:1.0})
            action = action[0]

        if self.exploration == "random":
            #Choose an action randomly.
            action = self.RandomAction()

        if self.exploration == "e-greedy":
            #Choose an action by greedily (with e chance of random action) from the Q-network
            if np.random.rand(1) < self.epsilon or self.current_steps < self.pre_train_steps:
                # action = self.env.action_space.sample()
                action = self.RandomAction()
            else:
                action, allQ = self.sess.run([self.q_net.predict,self.q_net.Q_out],feed_dict={self.q_net.inputs:[state],self.q_net.dropout:1.0})
                action = action[0]


        if self.exploration == "boltzmann":
            #Choose an action probabilistically, with weights relative to the Q-values.
            Q_d,allQ = self.sess.run([self.q_net.Q_dist,self.q_net.Q_out],feed_dict={self.q_net.inputs:[state], self.q_net.Temp:self.epsilon, self.q_net.dropout:1.0})
            action = np.random.choice(Q_d[0],p=Q_d[0])
            action = np.argmax(Q_d[0] == action)

        if self.exploration == "bayesian":
            #Choose an action using a sample from a dropout approximation of a bayesian q-network.
            action,allQ = self.sess.run([self.q_net.predict,self.q_net.Q_out],feed_dict={self.q_net.inputs:[state],self.q_net.dropout:(1-self.epsilon)+0.1})
            action = action[0]

        return action

    def Backward(self, state, action, reward, s1, done):
        self.buffer.add(np.reshape(np.array([state,action,reward ,s1,done]),[1,5]))
        #Epsilon Drop Rate
        stepDrop = (self.startE - self.endE)/self.anneling_steps

        #Update Network using rewards
        if self.epsilon > self.endE and self.current_steps > self.pre_train_steps:
            self.epsilon -= stepDrop

        if self.current_steps > self.pre_train_steps and self.current_steps % 5 == 0:
            #We use Double-DQN training algorithm
            trainBatch = self.buffer.sample(self.batch_size)
            Q1 = self.sess.run(self.q_net.predict,feed_dict={self.q_net.inputs:np.vstack(trainBatch[:,3]), self.q_net.dropout:1.0})
            Q2 = self.sess.run(self.target_net.Q_out,feed_dict={self.target_net.inputs:np.vstack(trainBatch[:,3]), self.target_net.dropout:1.0})

            

            end_multiplier = -(trainBatch[:,4] - 1)
            doubleQ = Q2[range(self.batch_size),Q1]
            Q2[:,action] = trainBatch[:,2] + (self.disFact * doubleQ * end_multiplier)
            loss, _ = self.sess.run([self.q_net.loss, self.q_net.updateModel],feed_dict={self.q_net.inputs:np.vstack(trainBatch[:,0]), self.q_net.nextQ:Q2, self.q_net.dropout:1.0})
            self.UpdateTargetGraph()

            self.loss.append(loss)
            if len(self.loss) > 150:
                self.loss.pop(0)

            plt.clf()
            plt.plot(self.loss, 'r-')
            plt.pause(1e-45)

        self.current_steps += 1


    def SaveWeights(self):
        saver = tf.train.Saver()
        save_path = saver.save(self.sess, "./model/DQN.ckpt")
        print("Model Saved in File: %s" % save_path);

    def RestoreWeights(self):
        saver = tf.train.Saver()
        saver.restore(self.sess, "./model/DQN.ckpt")

        #Set Epsilon to endE
        self.epsilon = self.endE
        self.UpdateTargetGraph()

        print("Model Restored")

    def GetLog(self):
        return "Loss : " + str(self.loss) + "    epsilon : " + str(self.epsilon)

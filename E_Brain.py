from __future__ import division

import gym
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt


import tensorflow.contrib.slim as slim



class experience_buffer():
    def __init__(self, buffer_size = 10000):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self,experience):
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:(len(experience)+len(self.buffer))-self.buffer_size] = []
        self.buffer.extend(experience)

    def sample(self,size):
        return np.reshape(np.array(random.sample(self.buffer,size)),[size,5])


class Q_Network():
    def __init__(self):
        #These lines establish the feed-forward part of the network used to choose actions
        self.inputs = tf.placeholder(shape=[None,4],dtype=tf.float32)
        self.Temp = tf.placeholder(shape=None,dtype=tf.float32)
        self.keep_per = tf.placeholder(shape=None,dtype=tf.float32)

        hidden = slim.fully_connected(self.inputs,64,activation_fn=tf.nn.tanh,biases_initializer=None)
        hidden = slim.dropout(hidden,self.keep_per)
        self.Q_out = slim.fully_connected(hidden,2,activation_fn=None,biases_initializer=None)

        self.predict = tf.argmax(self.Q_out,1)
        self.Q_dist = tf.nn.softmax(self.Q_out/self.Temp)


        #Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        self.actions = tf.placeholder(shape=[None],dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions,2,dtype=tf.float32)

        self.Q = tf.reduce_sum(tf.multiply(self.Q_out, self.actions_onehot), reduction_indices=1)

        self.nextQ = tf.placeholder(shape=[None],dtype=tf.float32)
        loss = tf.reduce_sum(tf.square(self.nextQ - self.Q))
        trainer = tf.train.GradientDescentOptimizer(learning_rate=0.0005)
        self.updateModel = trainer.minimize(loss)




class E_Agent():
    def __init__(self):
        self.env = gym.make('CartPole-v0')



        # Set learning parameters
        self.exploration = "greedy" #Exploration method. Choose between: greedy, random, e-greedy, boltzmann, bayesian.
        self.disFact = .99 #Discount factor.
        self.num_episodes = 200000 #Total number of episodes to train network for.
        self.tau = 0.1 #Amount to update target network at each step.
        self.batch_size = 32 #Size of training batch
        self.startE = 1 #Starting chance of random action
        self.endE = 0.1 #Final chance of random action
        self.epsilon = self.startE

        self.anneling_steps = 200000 #How many steps of training to reduce startE to endE.
        self.pre_train_steps = 100 #Number of steps used before training updates begin.
        self.current_steps = 0

        #Experience
        self.buffer = experience_buffer()

        #Epsilon Drop Rate
        self.stepDrop = (self.startE - self.endE)/self.anneling_steps


        #Initialize TF Graph
        tf.reset_default_graph()
        self.q_net = Q_Network()
        self.target_net = Q_Network()


        #Initialize TF Variables
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

        #Initialize TF Trainable Variables
        self.targetGraph = self.InitTargetGraph(tf.trainable_variables())
        self.updateTarget()


        #create lists to contain total rewards and steps per episode
        jList = []
        jMeans = []
        rList = []
        rMeans = []

        i = -1
        while 1:
            state = self.env.reset()
            rAll = 0
            done = False
            j = 0
            i += 1

            while j < 999:
                j+=1

                action = self.Forward(state);

                #Get new state and reward from environment

                s1, reward ,done, _ = self.env.step(action)


                if self.epsilon < 0.3:
                    self.env.render()
                    print("episode : ", i, "epsilon : ", self.epsilon)

                #Add Experience
                self.buffer.add(np.reshape(np.array([state,action,reward ,s1,done]),[1,5]))
                self.Backward(reward);

                rAll += reward
                state = s1
                self.current_steps += 1
                if done == True:
                    break

            # jList.append(j)
            # rList.append(rAll)
            #
            # if i % 100 == 0 and i != 0:
            #     r_mean = np.mean(rList[-100:])
            #     j_mean = np.mean(jList[-100:])
            #     # if self.exploration == 'e-greedy':
            #     #     print("Mean Reward: " + str(r_mean) + " Steps: " + str(self.current_steps) + " epsilon: " + str(self.epsilon))
            #     # if self.exploration == 'boltzmann':
            #     #     print("Mean Reward: " + str(r_mean) + " Steps: " + str(self.current_steps) + " t: " + str(self.epsilon))
            #     # if self.exploration == 'bayesian':
            #     #     print("Mean Reward: " + str(r_mean) + " Steps: " + str(self.current_steps) + " p: " + str(self.epsilon))
            #     # if self.exploration == 'random' or self.exploration == 'greedy':
            #     #     print("Mean Reward: " + str(r_mean) + " Steps: " + str(self.current_steps))
            #     rMeans.append(r_mean)
            #     jMeans.append(j_mean)

        self.sess.close()
        # print("Percent of succesful episodes: " + str(sum(rList)/self.num_episodes) + "%")


        plt.plot(rMeans)
        plt.show()

    def Forward(self, state):
        if self.exploration == "greedy":
            #Choose an action with the maximum expected value.
            action , allQ = self.sess.run([self.q_net.predict, self.q_net.Q_out],feed_dict={self.q_net.inputs:[state], self.q_net.keep_per:1.0})
            action = action[0]

        if self.exploration == "random":
            #Choose an action randomly.
            action = self.env.action_space.sample()

        if self.exploration == "e-greedy":
            #Choose an action by greedily (with e chance of random action) from the Q-network
            if np.random.rand(1) < self.epsilon or self.current_steps < self.pre_train_steps:
                action = self.env.action_space.sample()
            else:
                print("Nework Prediction");
                action, allQ = self.sess.run([self.q_net.predict,self.q_net.Q_out],feed_dict={self.q_net.inputs:[state],self.q_net.keep_per:1.0})
                action = action[0]
        if self.exploration == "boltzmann":
            #Choose an action probabilistically, with weights relative to the Q-values.
            Q_d,allQ = self.sess.run([self.q_net.Q_dist,self.q_net.Q_out],feed_dict={self.q_net.inputs:[state], self.q_net.Temp:self.epsilon, self.q_net.keep_per:1.0})
            action = np.random.choice(Q_d[0],p=Q_d[0])
            action = np.argmax(Q_d[0] == action)

        if self.exploration == "bayesian":
            #Choose an action using a sample from a dropout approximation of a bayesian q-network.
            action,allQ = self.sess.run([self.q_net.predict,self.q_net.Q_out],feed_dict={self.q_net.inputs:[state],self.q_net.keep_per:(1-self.epsilon)+0.1})
            action = action[0]

        return action

    def Backward(self, reward):
        #Update Network using rewards
        if self.epsilon > self.endE and self.current_steps > self.pre_train_steps:
            self.epsilon -= self.stepDrop

        if self.current_steps > self.pre_train_steps and self.current_steps % 5 == 0:
            #We use Double-DQN training algorithm
            trainBatch = self.buffer.sample(self.batch_size)
            Q1 = self.sess.run(self.q_net.predict,feed_dict={self.q_net.inputs:np.vstack(trainBatch[:,3]), self.q_net.keep_per:1.0})
            Q2 = self.sess.run(self.target_net.Q_out,feed_dict={self.target_net.inputs:np.vstack(trainBatch[:,3]), self.target_net.keep_per:1.0})
            end_multiplier = -(trainBatch[:,4] - 1)
            doubleQ = Q2[range(self.batch_size),Q1]
            targetQ = trainBatch[:,2] + (self.disFact*doubleQ * end_multiplier)
            _ = self.sess.run(self.q_net.updateModel,feed_dict={self.q_net.inputs:np.vstack(trainBatch[:,0]), self.q_net.nextQ:targetQ, self.q_net.keep_per:1.0, self.q_net.actions:trainBatch[:,1]})
            self.updateTarget()

    def InitTargetGraph(self, weights):
        total_vars = len(weights)
        op_holder = []
        for idx,var in enumerate(weights[0:total_vars//2]):
            op_holder.append(weights[idx+total_vars//2].assign((var.value()*self.tau) + ((1-self.tau)*weights[idx+total_vars//2].value())))
        return op_holder

    def updateTarget(self):
        for op in self.targetGraph:
            self.sess.run(op)

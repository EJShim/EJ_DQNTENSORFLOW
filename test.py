import tensorflow as tf


DIM_STATE = 8
DIM_ACTIONS = 4


l_input = tf.placeholder(shape=[None,DIM_STATE], dtype=tf.float32)
l_hidden1 = tf.layers.dense(inputs=l_input, units=64, activation=tf.nn.relu, name='hidden1')
print(l_hidden1.shape)
l_hidden2 = tf.layers.dense(inputs=l_hidden1, units=64, activation=tf.nn.relu, name='hidden2')
print(l_hidden2.shape)
l_out = tf.layers.dense(inputs=l_hidden2, units=DIM_ACTIONS, name='out')
print(l_out.shape)
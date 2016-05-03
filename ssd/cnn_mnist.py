import os
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


import tensorflow as tf
sess = tf.InteractiveSession()

def weight_variable(shape, layer_counter):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial, name="w_%d"%(layer_counter))

def bias_variable(shape, layer_counter):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial, name="b_%d"%(layer_counter))

def conv2d(x, W, layer_counter):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME', name = "conv_%d"%(layer_counter))

def max_pool_2x2(x, layer_counter):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME', name="max_pool_%d"%(layer_counter))


layer_counter = 0
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

W_conv1 = weight_variable([5, 5, 1, 32], layer_counter)
b_conv1 = bias_variable([32], layer_counter)

x_image = tf.reshape(x, [-1,28,28,1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1, layer_counter) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1, layer_counter)

layer_counter = layer_counter + 1

W_conv2 = weight_variable([5, 5, 32, 64],layer_counter)
b_conv2 = bias_variable([64],layer_counter)

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2,layer_counter) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2,layer_counter)

layer_counter = layer_counter + 1

W_fc1 = weight_variable([7 * 7 * 64, 1024], layer_counter)
b_fc1 = bias_variable([1024], layer_counter)

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1, name="fc_%d"%(layer_counter)) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob, name="dropout_%d"%(layer_counter))

layer_counter = layer_counter + 1

W_fc2 = weight_variable([1024, 10], layer_counter)
b_fc2 = bias_variable([10], layer_counter)

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2, name="fc_%d"%(layer_counter)) + b_fc2)

cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv), name="cross_entropy")
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Add ops to save and restore all the variables.
saver = tf.train.Saver(tf.all_variables(), max_to_keep=None)

sess.run(tf.initialize_all_variables())
for i in range(20000):
  batch = mnist.train.next_batch(50)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
    save_path = saver.save(sess, os.getcwd() + "/models/model.ckpt", global_step=i)
    print("Model saved in file: %s" % save_path)
    
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

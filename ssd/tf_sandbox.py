""" TF sandbox for testing new stuff """

import math

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_boolean('fake_data', False, 'If true, uses fake data '
                    'for unit testing.')
flags.DEFINE_integer('max_steps', 1000, 'Number of steps to run trainer.')
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')

def logSoftMax(vector):
    maxLogit= tf.reduce_max(vector,reduction_indices=1) # [batch_size]
    lse = tf.log( tf.reduceSum(tf.exp( logits - maxLogit ) + maxLogit) )
    return vector - lse


def test_sandbox():
    # Create a tensor with dummy vaules
    # NumpyArrays
    inputA = np.random.rand(3,2)
    inputB = np.random.rand(2,4)

    print("Inputs:")
    print(inputA)
    print(inputB)
    
    def numpyTest():
        return np.dot(inputA,inputB)

    # Do whatever calculation I want to test (build the graph)
    sess = tf.InteractiveSession()

    in1 = tf.placeholder(tf.float32, [3,2], name='input1')
    in2 = tf.placeholder(tf.float32, [2,4], name='input2')

    #out1 = tf.placeholder(tf.float32, [3,4], name='output')

    with tf.name_scope('test-matmul'):
        out_tf = tf.matmul( in1, in2 )
        
    merged = tf.merge_all_summaries()
    writer = tf.train.SummaryWriter('logs', sess.graph_def)
    #tf.initialize_all_variables().run() # no idea what this does
    
    # Execute and print result

    feed = {in1: inputA, in2: inputB}
    result = sess.run(out_tf, feed_dict=feed)

    print("\n\ngolden result:")
    print(numpyTest())
    print("result:")
    print(result)
    
    #summary_str = result[0]
    #outputGraph = result[1]
    #writer.add_summary(summary_str)
    #print('output of graph: %s' % (outputGraph))
    
def test_tensorboard(_):
    # Import data
    mnist = input_data.read_data_sets('/tmp/data/', one_hot=True,
                                      fake_data=FLAGS.fake_data)

    sess = tf.InteractiveSession()

    # Create the model
    x = tf.placeholder(tf.float32, [None, 784], name='x-input')
    W = tf.Variable(tf.zeros([784, 10]), name='weights')
    b = tf.Variable(tf.zeros([10], name='bias'))

    # Use a name scope to organize nodes in the graph visualizer
    with tf.name_scope('Wx_b'):
        y = tf.nn.softmax(tf.matmul(x, W) + b)
        
    # Add summary ops to collect data
    _ = tf.histogram_summary('weights', W)
    _ = tf.histogram_summary('biases', b)
    _ = tf.histogram_summary('y', y)

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')
    # More name scopes will clean up the graph representation
    with tf.name_scope('xent'):
        cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
        _ = tf.scalar_summary('cross entropy', cross_entropy)
        
    with tf.name_scope('train'):
        train_step = tf.train.GradientDescentOptimizer(
            FLAGS.learning_rate).minimize(cross_entropy)
        
    with tf.name_scope('test'):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        _ = tf.scalar_summary('accuracy', accuracy)
        
    # Merge all the summaries and write them out to /tmp/mnist_logs
    merged = tf.merge_all_summaries()
    writer = tf.train.SummaryWriter('logs', sess.graph_def)
    tf.initialize_all_variables().run()

    # Train the model, and feed in test data and record summaries every 10 steps

    for i in range(FLAGS.max_steps):
        if i % 10 == 0:  # Record summary data and the accuracy
            if FLAGS.fake_data:
                batch_xs, batch_ys = mnist.train.next_batch(
                    100, fake_data=FLAGS.fake_data)
                feed = {x: batch_xs, y_: batch_ys}
            else:
                feed = {x: mnist.test.images, y_: mnist.test.labels}
            result = sess.run([merged, accuracy], feed_dict=feed)
            summary_str = result[0]
            acc = result[1]
            writer.add_summary(summary_str, i)
            print('Accuracy at step %s: %s' % (i, acc))
        else:
            batch_xs, batch_ys = mnist.train.next_batch(
                100, fake_data=FLAGS.fake_data)
            feed = {x: batch_xs, y_: batch_ys}
            sess.run(train_step, feed_dict=feed)


def main(_):
    test_sandbox()
    
if __name__ == '__main__':
    tf.app.run()

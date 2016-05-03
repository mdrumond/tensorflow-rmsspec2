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

def getTestMatrix():
    return np.random.randn(25, 20)

def runTestSession(feed,graph,golden_res):
    # Do whatever calculation I want to test (build the graph)
    sess = tf.InteractiveSession()

    merged = tf.merge_all_summaries()
    writer = tf.train.SummaryWriter('logs', sess.graph_def)
    result = sess.run(graph, feed_dict=feed)

    print("\n\ngolden result:")
    print(golden_res)
    print("result:")
    print(result)

    print("match? ",np.allclose(golden_res,result, rtol=1e-03, atol=1e-03,))

                     

def test_sharpOp():

    # NumpyArrays
    inputA = getTestMatrix()
    print("Inputs:")
    print(inputA)

    def numpyTest():
        U, s, V = np.linalg.svd(inputA, full_matrices=False)
        return np.dot( U, V ) * np.sum(s)

 
    tf_inA = tf.placeholder(tf.float32, inputA.shape, name='input1')
    tf_graph=sharpOp(tf_inA)
    feed = {tf_inA : inputA}
    runTestSession(feed,tf_graph,numpyTest())

 
    
def logSoftMax(vector):
    maxLogit= tf.reduce_max(vector,reduction_indices=1,keep_dims=True) # [batch_size]
    lse = tf.log( tf.reduce_sum(tf.exp( vector - maxLogit ), reduction_indices=1, keep_dims=True ) ) + maxLogit
    return vector - lse

def test_logSoftMax():
    # NumpyArrays
    inputA = getTestMatrix()

    print("Inputs:")
    print(inputA)

    def numpyTest():
        maxLogit = np.apply_along_axis(np.max,1,inputA) # returns [batch]
        print(maxLogit)
        expSubMax = np.exp(np.apply_along_axis(np.subtract,0,inputA,maxLogit)) # returns [batch,classes]
        print(expSubMax)
        lse =  np.log( np.sum(expSubMax, axis=1) ) + maxLogit # returns [batch]
        print(lse)
        return np.apply_along_axis(np.subtract,0,inputA,lse) # returns [batch,classes]

    tf_inA = tf.placeholder(tf.float32, [4,3], name='input1')
    tf_graph=logSoftMax(tf_inA)
    feed = {tf_inA : inputA}
    runTestSession(feed,tf_graph,numpyTest())
    

def test_NNLCriterion():
    # NumpyArrays
    inputA = np.array([[1.,2.,3.],[4.,5.,6.],[7.,8.,9.],[10.,11.,12.]])
    labels = np.array([2,1,0,1], dtype=np.int32)
    
    def numpyTest():
        numPyOut = np.empty(inputA.shape[0])
        for currLine in range(inputA.shape[0]):
            numPyOut[currLine] = - inputA[currLine][labels[currLine]]
            
        return numPyOut
    
    tf_inA = tf.placeholder(tf.float32, [4,3], name='input1')
    tf_labels = tf.placeholder(tf.int32,4,name='labels')

    def tf_graph(inA, labels):
        batch_size = tf.size(labels)
        labels = tf.expand_dims(labels, 1)
        indices = tf.expand_dims(tf.range(0, batch_size), 1)
        concated = tf.concat(1, [indices, labels])
        onehot_labels = tf.sparse_to_dense(
            concated, tf.pack([batch_size, 3]), 1.0, 0.0)
        return - tf.reduce_sum(tf.mul(inA, onehot_labels), reduction_indices=1)
    
    feed = {tf_inA : inputA, tf_labels : labels}
    runTestSession(feed,tf_graph(tf_inA,tf_labels),numpyTest())

        
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


    in1 = tf.placeholder(tf.float32, [3,2], name='input1')
    in2 = tf.placeholder(tf.float32, [2,4], name='input2')

    #out1 = tf.placeholder(tf.float32, [3,4], name='output')
    with tf.name_scope('test-matmul'):
        out_tf = tf.matmul( in1, in2 )
        
    #tf.initialize_all_variables().run() # no idea what this does
    
    # Execute and print result

    feed = {in1: inputA, in2: inputB}
    runTestSession(feed,out_tf,numpyTest())
    
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


def test_matrix_comp():
    def getTestMatrix(transp=False):
        return np.random.randn(3, 4) if transp else np.random.radn(4,3)

    def numpyTestSvd(test_in):
        U, s, V = np.linalg.svd(test_in, full_matrices=False)
        print("### SVD Test:")
        print("U")
        print(U)
        print("s")
        print(s)
        print("V")
        print(V)

    def numpyTestSvdS(test_in):
        U, s, V = np.linalg.svd(test_in, full_matrices=False)
        return s

    def numpyTestQr(test_in):
        q,r = np.linalg.qr(test_in,mode='complete')
        print("### QR Test")
        print("q")
        print(q)
        print("r")
        print(r)

        print("normal")
        a = getTestMatrix(True)
        print("a",a.shape,"\n",a)
        U, s, V = np.linalg.svd(a, full_matrices=False)
        print("U",U.shape,"\n",U)
        print("s",s.shape,"\n", s)
        print("V",V.shape,"\n",V)
    
        print("transp")
        a = getTestMatrix(True)
        print("a",a.shape,"\n",a)
        U, s, V = np.linalg.svd(a, full_matrices=False)
        print("U",U.shape,"\n",U)
        print("s",s.shape,"\n", s)
        print("V",V.shape,"\n",V)
    
def main(_):
    test_sharpOp()

if __name__ == '__main__':
    tf.app.run()

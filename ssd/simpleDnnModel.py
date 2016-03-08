
"""Builds the simple MNIST network

The network is composed by:
Input [28*28] -> Layer1 [300] -> Output [10]

The loss function used os logSoftMax"""

import math

import tensorflow as tf

# The MNIST dataset has 10 classes, representing the digits 0 through 9.
NUM_CLASSES = 10

# The MNIST images are always 28x28 pixels.
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

LAYER_SIZE = 300

def logSoftMax(vector):
    maxLogit= tf.reduce_max(vector,reduction_indices=1) # [batch_size]
    lse = tf.log( tf.reduceSum(tf.exp( logits - maxLogit ) + maxLogit) )
    return vector - lse



def inference(images):
    """
    Build the MNIST model
    """

    # Hidden 1
    with tf.name_scope('hidden1'):
        weights = tf.Variable(
            tf.truncated_normal([IMAGES_PIXELS, LAYER_SIZE],
                                stddev= 1.0 / math.sqrt(float(IMAGE_PIXELS))),
        name='weights')
        biases = tf.Variable(tf.zeros([LAYER_SIZE]),
                             name='biases')
        hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)

    # Output Layer - is this correct? does this layer have any weights?
    with tf.name_scope('softmax_linear'):
        weights = tf.Variable(
            tf.truncated_normal([LAYER_SIZE, NUM_CLASSES],
                                stddev=1.0 / math.sqrt(float(LAYER_SIZE))),
            name='weights')
        biases = tf.Variable(tf.zeros([NUM_CLASSES]),
                             name='biases')
        logits = logSoftMax(tf.matmul(hidden2, weights) + biases)
        return logits
    
    
def loss(logits, labels):
      """Calculates the loss from the logits and the labels.

      Args:
      logits: Logits tensor, float - [batch_size, NUM_CLASSES].
      labels: Labels tensor, int32 - [batch_size].

      Returns:
      loss: NLL criterion Loss tensor of type float.
  """
      return None

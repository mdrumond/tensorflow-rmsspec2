
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
# Takses vector[batch_size,layer_size]
def logSoftMax(vector):
    maxLogit= tf.reduce_max(vector,reduction_indices=1,keep_dims=True) # [batch_size]
    lse = tf.log( tf.reduce_sum(tf.exp( vector - maxLogit ), reduction_indices=1, keep_dims=True ) ) + maxLogit
    return vector - lse

# Takes inA[batch_size,categories], labels[batch_size] and outputs loss[batch_size]
def NLLCriterion(inA, labels):
    batch_size = tf.size(labels)
    labels = tf.expand_dims(labels, 1)
    indices = tf.expand_dims(tf.range(0, batch_size), 1)
    concated = tf.concat(1, [indices, labels])
    onehot_labels = tf.sparse_to_dense(
        concated, tf.pack([batch_size, 3]), 1.0, 0.0)
    return - tf.reduce_sum(tf.mul(inA, onehot_labels), reduction_indices=1)


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
      return NLLCriterion(logits,labels)


  def trainingSGD(loss, learning_rate):
      """Sets up the training Ops.
      
      Creates a summarizer to track the loss over time in TensorBoard.
      
      Creates an optimizer and applies the gradients to all trainable variables.

      The Op returned by this function is what must be passed to the
      `sess.run()` call to cause the model to train.

      Args:
        loss: Loss tensor, from loss().
        learning_rate: The learning rate to use for gradient descent.

      Returns:
        train_op: The Op for training.
  """
      # Add a scalar summary for the snapshot loss.
      tf.scalar_summary(loss.op.name, loss)
      # Create the gradient descent optimizer with the given learning rate.
      optimizer = tf.train.GradientDescentOptimizer(learning_rate)
      # Create a variable to track the global step.
      global_step = tf.Variable(0, name='global_step', trainable=False)
      # Use the optimizer to apply the gradients that minimize the loss
      # (and also increment the global step counter) as a single training step.
      train_op = optimizer.minimize(loss, global_step=global_step)
      return train_op



def evaluation(logits, labels):
    """Evaluate the quality of the logits at predicting the label.

    Args:
      logits: Logits tensor, float - [batch_size, NUM_CLASSES].
      labels: Labels tensor, int32 - [batch_size], with values in the
        range [0, NUM_CLASSES).

    Returns:
      A scalar int32 tensor with the number of examples (out of batch_size)
      that were predicted correctly.
    """
    # For a classifier model, we can use the in_top_k Op.
    # It returns a bool tensor with shape [batch_size] that is true for
    # the examples where the label's is was in the top k (here k=1)
    # of all logits for that example.
    correct = tf.nn.in_top_k(logits, labels, 1)
    # Return the number of true entries.
    return tf.reduce_sum(tf.cast(correct, tf.int32))


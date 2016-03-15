import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_boolean('fake_data', False, 'If true, uses fake data '
                     'for unit testing.')
flags.DEFINE_integer('max_steps', 1000, 'Number of steps to run trainer.')
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')

from 
def main(_):
    

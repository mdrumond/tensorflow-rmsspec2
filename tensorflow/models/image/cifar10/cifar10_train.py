# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A binary to train CIFAR-10 using a single GPU.

Accuracy:
cifar10_train.py achieves ~86% accuracy after 100K steps (256 epochs of
data) as judged by cifar10_eval.py.

Speed: With batch_size 128.

System        | Step Time (sec/batch)  |     Accuracy
------------------------------------------------------------------
1 Tesla K20m  | 0.35-0.60              | ~86% at 60K steps  (5 hours)
1 Tesla K40m  | 0.25-0.35              | ~86% at 100K steps (4 hours)

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time
import os
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.models.image.cifar10 import cifar10

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train-dir', '/data/cifar10/cifar10_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max-steps', 1000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log-device-placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('gpu-number', 0,
                            """GPU to use.""")

def print_logits(logits_value, label_value, loss_per_batch_value):
  out_str = ""
  for test, loss, label in zip(logits_value, loss_per_batch_value, label_value):
    out_str += "%d,%5.5f>>>\t" % (label, loss)
    for logit in test:
      out_str += "%5.5f\t" % logit
    out_str += "\n"

  return out_str

def train():
  """Train CIFAR-10 for a number of steps."""
  with tf.Graph().as_default():
    
    global_step = tf.Variable(0, trainable=False)

    # Get images and labels for CIFAR-10.
    #with tf.device('/gpu:%d' % FLAGS.gpu_number):
    images, labels = cifar10.distorted_inputs()

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = cifar10.inference(images)
    loss_per_batch = cifar10.loss_per_batch(logits, labels)
    # Calculate loss.
    loss = cifar10.loss(logits, labels)

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    train_op = cifar10.train(loss, global_step, FLAGS.gpu_number)

    # Create a saver.
    saver = tf.train.Saver(tf.all_variables(), max_to_keep=None)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.merge_all_summaries()

    # Build an initialization operation to run below.
    init = tf.initialize_all_variables()

    # Start running operations on the Graph.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    config.allow_soft_placement=True
    config.log_device_placement=FLAGS.log_device_placement
    sess = tf.Session(config=config)
    
    tf.train.write_graph(sess.graph_def, FLAGS.train_dir,
                         "cifar10_train.pb", False)
    sess.run(init)

    # Start the queue runners.
    tf.train.start_queue_runners(sess=sess)

    summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)

    train_start_time = time.time()
    for step in xrange(FLAGS.max_steps):
      start_time = time.time()
      _, loss_value, logits_value, loss_per_batch_value, labels_value = sess.run([train_op, loss, logits, loss_per_batch, labels])
      duration = time.time() - start_time
      #logits_str = print_logits(logits_value, labels_value, loss_per_batch_value)
      
      #with open(os.path.join(FLAGS.train_dir, 'logits_%d.log' % step),'w') as f:
      #  f.write("%s" % logits_str)

      assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

      if step % 10 == 0:
        num_examples_per_step = FLAGS.batch_size
        examples_per_sec = num_examples_per_step / duration
        sec_per_batch = float(duration)

        format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                      'sec/batch)')
        log_str  = (format_str % (datetime.now(), step, loss_value,
                                  examples_per_sec, sec_per_batch))
        print(log_str)
        with open(os.path.join(FLAGS.train_dir, 'train.log'),'a+') as f:
          f.write("%s\n" % log_str)

      if step % 500 == 0:
        summary_str = sess.run(summary_op)
        summary_writer.add_summary(summary_str, step)
        checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
        save_path = saver.save(sess, checkpoint_path, global_step=step)
    train_duration = time.time() - train_start_time

    log_str = ("Finishing. Training %d batches of %d images took %fs\n" %
               (FLAGS.max_steps, FLAGS.batch_size, float(train_duration)))
    print(log_str)
    with open(os.path.join(FLAGS.train_dir, 'train.log'),'a+') as f:
      f.write("%s" % log_str)

def main(argv=None):  # pylint: disable=unused-argument
  cifar10.maybe_download_and_extract()
  if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)
  train()


if __name__ == '__main__':
  tf.app.run()

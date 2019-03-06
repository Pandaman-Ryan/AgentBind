# Evaluation
# 
# Start Date: Apr. 16th, 2017
# Last Update: May 4th, 2017
# Contributor: An Zheng
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model import AgentBind
import deep_taylor_lrp as lrp

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data_dir', '/storage/pandaman/project/AgentBind-Release-Version/tmp/HNF4A+HepG2/seqs_one_hot_b/vis-samples/',
                            """Directory where to read input data""")
tf.app.flags.DEFINE_integer('seq_size', 1024, """The length of each sample.""")
tf.app.flags.DEFINE_integer('batch_size', 128, """The size of each batch.""")
tf.app.flags.DEFINE_integer('n_eval_samples', 700, """Number of input samples in the evaluation process.""")
tf.app.flags.DEFINE_integer('n_samples', 253, """Number of input samples in the evaluation process.""")
tf.app.flags.DEFINE_integer('n_classes', 1, """Number of categories in data.""")
tf.app.flags.DEFINE_integer('n_channels', 4, """Number of input channels.""")

tf.app.flags.DEFINE_string('eval_dir', '/storage/pandaman/project/AgentBind-Release-Version/tmp/JunD+GM12878/ckpt-test-phase_b/',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('checkpoint_dir', '/storage/pandaman/project/AgentBind-Release-Version/tmp/JunD+GM12878/model_b/',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_float('moving_average_decay', 0.999,
                            """Moving average decay""")


def test(sample_index):
    """Eval CNN for a number of steps."""
    with tf.Graph().as_default() as g:
        training = tf.placeholder(tf.bool, shape=[])
        # Data loading
        vpht = AgentBind(batch_size=FLAGS.batch_size,
                                seq_size=FLAGS.seq_size,
                                n_channels=FLAGS.n_channels,
                                n_classes=FLAGS.n_classes)
        test_dataset = vpht.load(data_dir=FLAGS.data_dir+"/%d/"%(sample_index),
                                n_samples=FLAGS.n_eval_samples,
                                n_epochs=None, shuffle=False,
                                n_info=6)
        iterator = test_dataset.make_one_shot_iterator()
        seqs, labels, info = iterator.get_next()

        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits = vpht.infer(seqs, training, trainable_conv_layers=False)
        input_weights = lrp.lrp(logits, 0, 1,
                                graph = g, conv_strides=[1, 1, 2, 1])


        # Restore the moving average version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(
                                FLAGS.moving_average_decay)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        with tf.Session(config=tf.ConfigProto(
            log_device_placement=False,
            gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.45)
            )) as sess:
            # init
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            # load the trained model
            ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                # Restores from checkpoint
                saver.restore(sess, ckpt.model_checkpoint_path)
                # Assuming model_checkpoint_path looks something like:
                #   /my-favorite-path/cifar10_train/model.ckpt-0,
                # extract global_step from it.
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            else:
                print('No checkpoint file found')
                return
            
            weights_value_list = []
            num_iter = int(math.ceil(FLAGS.n_eval_samples / FLAGS.batch_size))
            for step in range(num_iter): 
                try:
                    weights_value = sess.run([input_weights],
                                        feed_dict={training:False})
                    weights_value_list.append(weights_value)
                except tf.errors.OutOfRangeError:
                    break

    return

def main(argv=None):  # pylint: disable=unused-argument
    test(0)

if __name__ == '__main__':
    tf.app.run()

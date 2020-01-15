# Evaluation
# 
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time

import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model import AgentBind

FLAGS = tf.app.flags.FLAGS

# Parameters 
tf.app.flags.DEFINE_string('data_dir', None,"""Directory where to read input data""")
tf.app.flags.DEFINE_integer('seq_size', None, """The length of each sample.""")
tf.app.flags.DEFINE_integer('n_eval_samples', None, """Number of input samples in the evaluation process.""")
tf.app.flags.DEFINE_string('eval_dir', None, """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('checkpoint_dir', None, """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_string('result_dir', None, """Directory where to store all results.""")
tf.app.flags.DEFINE_string('TF_name', None, """The name of the TF of interest""")

# default parameters
tf.app.flags.DEFINE_integer('batch_size', 128, """The size of each batch.""")
tf.app.flags.DEFINE_integer('n_classes', 1, """Number of categories in data.""")
tf.app.flags.DEFINE_integer('n_channels', 4, """Number of input channels.""")
tf.app.flags.DEFINE_float('moving_average_decay', 0.999, """Moving average decay""")

def _print_summary(losses, logits=None, labels=None, info=None):
    loss_avg = sum(losses)/float(len(losses))
    num_of_features = (logits.shape)[1]
    feature_index = 0 # because there is only one class in this model
    lbl = labels[:, feature_index]
    lgt = logits[:, feature_index]
    intensity = (info[:, 0]).astype(float)
    pval = (info[:, 1]).astype(float)
    ###
    # AUC
    roc_value = roc_auc_score(lbl, lgt)
    pr_value = average_precision_score(lbl, lgt)
    summary_file = "%s/auc_summary.txt" %(FLAGS.result_dir)
    with open(summary_file,'a') as ofile:
        line_to_save = "%s\t%f\t%f\n" %(FLAGS.TF_name, roc_value, pr_value)
        ofile.write(line_to_save)

    ###
    # plot curves
    figure_dir = "%s/%s/" %(FLAGS.result_dir, FLAGS.TF_name)
    if not tf.gfile.Exists(figure_dir):
        tf.gfile.MakeDirs(figure_dir)

    # plot ROC
    fpr, tpr, thresholds = roc_curve(lbl, lgt)
    plt.plot(fpr, tpr)
    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate')
    plt.savefig('%s/roc-curve.png' %(figure_dir))
    plt.close()

    # plot PR curve
    precision, recall, _ = precision_recall_curve(lbl, lgt)
    plt.plot(precision, recall)
    plt.xlabel('precision')
    plt.ylabel('recall')
    plt.savefig('%s/pr-curve.png' %(figure_dir))
    plt.close()

    # raw data
    raw_data_file = "%s/lable-logit.txt" %(figure_dir)
    with open(raw_data_file, 'w') as ofile:
        for sample_index in range(len(lbl)):
            line_to_save = "%d\t%f\n" %(lbl[sample_index], lgt[sample_index])
            ofile.write(line_to_save)

    return

def test():
    """Eval CNN for a number of steps."""
    with tf.Graph().as_default() as g:
        training = tf.placeholder(tf.bool, shape=[])
        # Data loading
        agent = AgentBind(batch_size=FLAGS.batch_size,
                                seq_size=FLAGS.seq_size,
                                n_channels=FLAGS.n_channels,
                                n_classes=FLAGS.n_classes)
        test_dataset = agent.load(data_dir=FLAGS.data_dir,
                                n_samples=FLAGS.n_eval_samples,
                                n_epochs=1, shuffle=False,
                                n_info=2)
        iterator = test_dataset.make_one_shot_iterator()
        seqs, labels, info = iterator.get_next()

        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits = agent.infer(seqs, training, trainable_conv_layers=False)
        logits_sigmoid = tf.sigmoid(logits)
        loss = agent.loss(logits, labels)

        # Restore the moving average version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(
                                FLAGS.moving_average_decay)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        # run_eval(saver, logits, labels, loss)
        #with tf.Session(config=tf.ConfigProto(
        #    gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.49)
        #   )) as sess:
        with tf.Session(config=tf.ConfigProto(
            log_device_placement=False,
            )) as sess:
            #gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.45)
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

            # Start the run.
            logits_value_list = []
            labels_value_list = []
            info_value_list = []
            losses = []
            
            step = 0
            while True:
                step += 1
                try:
                    logits_value, labels_value, info_value, loss_value = sess.run(
                                                [logits_sigmoid, labels, info, loss],
                                                feed_dict={training:False})
                    losses.append(loss_value)
                    logits_value_list.append(logits_value)
                    labels_value_list.append(labels_value)
                    info_value_list.append(info_value)
                except tf.errors.OutOfRangeError:
                    break

            logits_value_total = np.concatenate(logits_value_list, axis = 0)
            labels_value_total = np.concatenate(labels_value_list, axis = 0)
            info_value_total = np.concatenate(info_value_list, axis = 0)
            _print_summary(losses, logits_value_total, labels_value_total, info_value_total)

    return

def main(argv=None):  # pylint: disable=unused-argument
    if tf.gfile.Exists(FLAGS.eval_dir):
        tf.gfile.DeleteRecursively(FLAGS.eval_dir)
    tf.gfile.MakeDirs(FLAGS.eval_dir)
    test()

if __name__ == '__main__':
    tf.app.run()
###END OF FILE###############################

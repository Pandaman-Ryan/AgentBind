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

#from skimage.transform import resize
from model import AgentBind

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data_dir', None, """Directory where to read input data""")
tf.app.flags.DEFINE_integer('seq_size', None, """The length of each sample.""")
tf.app.flags.DEFINE_integer('n_samples', None, """Number of input samples in the evaluation process.""")
tf.app.flags.DEFINE_string('eval_dir', None, """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('checkpoint_dir', None, """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_string('figure_dir', None, """Directory where to save visualized figures.""")

# parameters by default
tf.app.flags.DEFINE_integer('batch_size', 128, """The size of each batch.""")
tf.app.flags.DEFINE_integer('n_classes', 1, """Number of categories in data.""")
tf.app.flags.DEFINE_integer('n_channels', 4, """Number of input channels.""")
tf.app.flags.DEFINE_float('moving_average_decay', 0.999, """Moving average decay""")

tf.app.flags.DEFINE_string('weight_file', None, """Directory where to save all weights.""")

def _resize(cam_arr, input_len):
    resized_arr = [[] for nt_index in range(input_len)]
    for nt_index in range(cam_arr.shape[0]):
        cam_score = cam_arr[nt_index]
        for nt_index_offset in range(26): #(the first-layer conv filter length is 26)
            resized_arr[nt_index+nt_index_offset].append(cam_score)
    
    resized_cam = np.array([np.mean(score_arr) for score_arr in resized_arr])
    return resized_cam

def _calculate_importance_score(weights, cam_values, cam_gradients, seqs):
    data_to_vis_list = []
    for sample_index in range(len(weights)):
        weights_ref = weights[sample_index]
        seqs_ref = seqs[sample_index]
        cam_grads = cam_gradients[sample_index]
        cam_vals = cam_values[sample_index]

        filter_weights = np.mean(cam_grads, axis=0)
        cam = np.zeros(cam_vals.shape[0], dtype=np.float32)
        for i, w in enumerate(filter_weights):
            cam += w * cam_vals[:, i]
        cam = np.maximum(cam, 0)
        cam_resized = _resize(cam, len(weights_ref))

        saliency_map = []
        for pos_index in range(len(weights_ref)):
            weight_value = sum([weights_ref[pos_index][nuc_index]* seqs_ref[pos_index][nuc_index] \
                                for nuc_index in range(len(weights_ref[pos_index]))])
            saliency_map.append(weight_value)

        data_to_vis = [cam_resized[pos_index]*saliency_map[pos_index] for pos_index in range(len(cam_resized))]
        signal_sum = sum([abs(val) for val in data_to_vis])
        if signal_sum == 0:
            print ("A failed sample occurred!")
            continue
        data_to_vis = [elem/signal_sum for elem in data_to_vis]
        data_to_vis_list.append(data_to_vis)
    return data_to_vis_list

#def _print_summary(weights=None, cam_values=None, cam_gradients=None, logits=None, info=None, seqs=None):
def _print_summary(scores=None, logits=None, info=None):
    # TODO: now this function can only deal with one class
    for sample_index in range(scores.shape[0]):
        feature_index = 0 # because there is only one class in this model
        lgt = logits[sample_index, feature_index]

        #lgt_ref = lgt[0]
        chromID = str(info[sample_index, 0])
        seq_start = int(info[sample_index, 1])
        seq_end = int(info[sample_index, 2])

        data_to_vis = scores[sample_index]

        # save results
        with open('%s' %(FLAGS.weight_file), 'a') as ofile:
            line_to_print = "%s;%d;%d\n" %(chromID, seq_start, seq_end)
            ofile.write(line_to_print)

            line_to_print = "%s" %(data_to_vis[0])
            for pos in range(1, FLAGS.seq_size):
                line_to_print += ";%s" %(data_to_vis[pos])
            line_to_print += "\n"
            ofile.write(line_to_print)

        with open('%s/vis-sample-%d.txt' %(FLAGS.figure_dir, sample_index), 'w') as ofile:
            for pos in range(FLAGS.seq_size):
                line_to_print = "%d\t%s\n" %(pos, data_to_vis[pos])
                ofile.write(line_to_print)

            plt.figure(figsize=(11,11))
            plt.scatter([pos for pos in range(FLAGS.seq_size)], data_to_vis, marker=".")
            plt.title("signal vs positions")
            plt.xlabel('positions -- %s-%d-%d' %(chromID, seq_start, seq_end))
            plt.ylabel('weights')
            plt.savefig('%s/vis-sample-%d.png' %(FLAGS.figure_dir, sample_index))
            plt.close()
    return

def test():
    """Eval CNN for a number of steps."""
    with tf.Graph().as_default() as g:
        training = tf.placeholder(tf.bool, shape=[])
        n_eval_samples = FLAGS.n_samples + FLAGS.batch_size
        # Data loading
        agent = AgentBind(batch_size=FLAGS.batch_size,
                                seq_size=FLAGS.seq_size,
                                n_channels=FLAGS.n_channels,
                                n_classes=FLAGS.n_classes)
        test_dataset = agent.load(data_dir=FLAGS.data_dir+"/vis-samples/",
                                n_samples=n_eval_samples,
                                n_epochs=None, shuffle=False,
                                n_info=3)
        iterator = test_dataset.make_one_shot_iterator()
        seqs, labels, info = iterator.get_next()

        # Build a Graph that computes the logits predictions from the
        # inference model.
        #with g.gradient_override_map({'Relu': 'GuidedRelu'}):
        logits = agent.infer(seqs, training, trainable_conv_layers=False)
        weights = agent.analyze(logits, seqs)
        cam_layer = agent.retrieve_cam_layer()
        cam_gradients = agent.compute_cam_gradient(logits)

        # Restore the moving average version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(
                                FLAGS.moving_average_decay)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        with tf.Session(config=tf.ConfigProto(
            log_device_placement=False,
            gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
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

            # Start the run.
            #weights_value_list = []
            #cam_layer_value_list = []
            #cam_gradients_value_list = []
            #seqs_value_list = []
            data_to_vis_list = []
            logits_value_list = []
            info_value_list = []
            
            num_iter = int(math.floor(n_eval_samples / FLAGS.batch_size))
            for step in range(num_iter): 
                try:
                    weights_value, cam_layer_value, cam_gradients_value,\
                            logits_value, info_value, seqs_value = sess.run(
                                        [weights, cam_layer, cam_gradients, logits, info, seqs],
                                        feed_dict={training:False})
                    #weights_value_list.append(weights_value)
                    #cam_layer_value_list.append(cam_layer_value)
                    #cam_gradients_value_list.append(cam_gradients_value)
                    #seqs_value_list.append(seqs_value)
                    data_to_vis_value = _calculate_importance_score(
                                            weights_value, cam_layer_value, cam_gradients_value,
                                            seqs_value)
                    logits_value_list.append(logits_value)
                    info_value_list.append(info_value)
                    data_to_vis_list.append(data_to_vis_value)
                except tf.errors.OutOfRangeError:
                    break
            #weights_value_total = np.concatenate(weights_value_list, axis = 0)
            #cam_layer_value_total = np.concatenate(cam_layer_value_list, axis = 0)
            #cam_gradients_value_total = np.concatenate(cam_gradients_value_list, axis = 0)
            #seqs_value_total = np.concatenate(seqs_value_list, axis=0)
            data_to_vis_total = np.concatenate(data_to_vis_list, axis = 0)
            logits_value_total = np.concatenate(logits_value_list, axis = 0)
            info_value_total = np.concatenate(info_value_list, axis = 0)
            #_print_summary(weights_value_total, cam_layer_value_total,
            #            cam_gradients_value_total, logits_value_total,
            #            info_value_total, seqs_value_total)
            _print_summary(data_to_vis_total, logits_value_total, info_value_total)

    return

def main(argv=None):  # pylint: disable=unused-argument
    if tf.gfile.Exists(FLAGS.eval_dir):
        tf.gfile.DeleteRecursively(FLAGS.eval_dir)
    tf.gfile.MakeDirs(FLAGS.eval_dir)

    if tf.gfile.Exists(FLAGS.figure_dir):
        tf.gfile.DeleteRecursively(FLAGS.figure_dir)
    tf.gfile.MakeDirs(FLAGS.figure_dir)
 
    if tf.gfile.Exists(FLAGS.weight_file):
        tf.gfile.Remove(FLAGS.weight_file)
    
    test()

if __name__ == '__main__':
    tf.app.run()

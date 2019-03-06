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
import random
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
#from model_improved_vis_gradients import VampireHunter
#import deep_taylor_lrp as lrp

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data_dir', None, """Directory where to read input data""")
tf.app.flags.DEFINE_integer('seq_size', None, """The length of each sample.""")
tf.app.flags.DEFINE_integer('n_samples', None, """Number of input samples in the evaluation process.""")
tf.app.flags.DEFINE_string('eval_dir', None, """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('checkpoint_dir', None, """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_string('figure_dir', None, """Directory where to save visualized figures.""")
tf.app.flags.DEFINE_string('report_dir', None, """Directory where to save infomation of explainable ratios.""")

# parameters by default
tf.app.flags.DEFINE_integer('batch_size', 128, """The size of each batch.""")
tf.app.flags.DEFINE_integer('n_classes', 1, """Number of categories in data.""")
tf.app.flags.DEFINE_integer('n_channels', 4, """Number of input channels.""")
tf.app.flags.DEFINE_float('moving_average_decay', 0.999, """Moving average decay""")


def _print_summary(weights=None, logits=None, info=None, seqs=None, n_samples=None):
    '''
        Evalutions:
            1:  p-value: shuffle positions, averaged on all samples, and then calculate pval of sn-ratio
            2:  signal & noise distribution
            3:  signal to noise ratio
    '''
    n_control_experiments = 1000

    explainable_ratio_list = []
    signal_list_total = []
    noise_list_total = []
    sn_ratio_list = [] # signal/noise ratio
    sn_ratio_control_list = []

    # evaluate each sample
    #weights_integrated = None
    for sample_index in range(n_samples):
        lgt = logits[sample_index, 0] #feature_index = 0, because there is only one class in this model
        position_info = info[sample_index, 0]
        wgt = weights[sample_index]
        sq = seqs[sample_index]

        saliency_map = []
        for pos_index in range(len(wgt)):
            weight_value = sum([wgt[pos_index][nuc_index]* sq[pos_index][nuc_index] \
                            for nuc_index in range(len(wgt[pos_index]))])
            saliency_map.append(weight_value)

        #if weights_integrated == None:
        #    weights_integrated = saliency_map
        #else:
        #    weights_integrated = [saliency_map[elem_index] + weights_integrated[elem_index] \
        #                            for elem_index in range(len(saliency_map))]

        # record stats
        #if (sample_index%FLAGS.n_units) == (FLAGS.n_units-1):
        # normalization
        #signal_sum = sum([abs(val) for val in weights_integrated])
        #weights_integrated = [elem/signal_sum for elem in weights_integrated]
        #data_to_vis = weights_integrated
        data_to_vis = saliency_map 
        signal_sum = sum([abs(val) for val in data_to_vis])
        if signal_sum == 0:
            print ("A failed sample occurred!")
            continue
        data_to_vis = [elem/signal_sum for elem in data_to_vis]

        # calculate signals inside peaks
        explainable_ratio = 0
        signal_positions = []
        signal_positions_control = [[] for _ in range(n_control_experiments)]
        positions = position_info.split('&')
        for pos in positions:
            start, end = pos.split('-')
            start = int(start)
            end = int(end)
            length = end - start 

            for motif_pos in range(start,end):
                explainable_ratio += abs(data_to_vis[motif_pos])
                signal_positions.append(motif_pos)

            for exp_index in range(n_control_experiments):
                pos_start = random.choice([_ for _ in range(len(data_to_vis)-length+1)])
                signal_positions_control[exp_index] += [pos for pos in range(pos_start, pos_start+length)]

        # explainable ratio
        explainable_ratio_list.append(explainable_ratio)

        # signals and noises
        signal_list = []
        noise_list = []
        for pos in range(len(data_to_vis)):
            if pos in signal_positions:
                signal_list.append(abs(data_to_vis[pos]))
            else:
                noise_list.append(abs(data_to_vis[pos]))
        signal_avg = sum(signal_list)/len(signal_list)
        noise_avg = sum(noise_list)/len(noise_list)
        signal_noise_ratio = signal_avg/noise_avg
        sn_ratio_list.append(signal_noise_ratio)
        signal_list_total += [signal_avg] #signal_list_total += signal_list
        noise_list_total += noise_list

        # control experiments
        for exp_index in range(n_control_experiments):
            signal_list_control = []
            noise_list_control = []
            for pos in range(len(data_to_vis)):
                if pos in signal_positions_control[exp_index]:
                    signal_list_control.append(abs(data_to_vis[pos]))
                else:
                    noise_list_control.append(abs(data_to_vis[pos]))
            signal_avg_control = sum(signal_list_control)/len(signal_list_control)
            noise_avg_control = sum(noise_list_control)/len(noise_list_control)
            sn_ratio_control = signal_avg_control/noise_avg_control
            sn_ratio_control_list.append(sn_ratio_control)

        # save results
        with open('%s/vis-sample-%d.txt' %(FLAGS.figure_dir, sample_index), 'w') as ofile:
            for pos in range(FLAGS.seq_size):
                line_to_print = "%d\t%s\n" %(pos, data_to_vis[pos])
                ofile.write(line_to_print)

        # figure 1: plot signal distribution
        plt.figure(figsize=(11,11))
        plt.scatter([pos for pos in range(FLAGS.seq_size)], data_to_vis, marker=".")
        plt.title("signal vs positions")
        plt.xlabel('positions, signal at %s'\
                        %(position_info))
        plt.ylabel('weights')
        plt.savefig('%s/vis-sample-%d.png' %(FLAGS.figure_dir, sample_index))
        plt.close()

    ## results:
    # box plot for explainable ratio, pval, signal-to-noise ratio
    # distribution of signals and noises
    # explainable ratio, box plot
    saveData(explainable_ratio_list, 'explainable_ratio')
    drawBoxplot(explainable_ratio_list, 'explainable_ratio')

    saveData(sn_ratio_list, 'sn_ratio_list')
    saveData(sn_ratio_control_list, 'sn_ratio_control_list')
    drawBoxplot([sn_ratio_list, sn_ratio_control_list], 'signal_to_noise_ratio')
    
    # histogram for signals and noises
    saveData(signal_list_total, 'signal_list_total')
    saveData(noise_list_total, 'noise_list_total')
    drawHistogram_dual(signal_list_total, noise_list_total, 'dist_signals_and_noises')
    return

def saveData(data, filename):
    save_path = "%s/%s.txt" %(FLAGS.report_dir, filename)
    with open(save_path, 'w') as ofile:
        for dt in data:
            line_to_save = "%f\n" %(dt)
            ofile.write(line_to_save)
    return

def drawBoxplot(data, filename):
    plt.figure()
    plt.boxplot(data)
    plt.title(filename)
    plt.savefig("%s/%s.png" %(FLAGS.report_dir, filename))
    plt.close()
    return

def drawHistogram_dual(y1, y2, filename):
    fig, ax1 = plt.subplots()
    colors=['red', 'green']
    ax2 = ax1.twinx()
    n, bins, patches = ax1.hist([y1,y2], bins=100, color=colors)
    ax1.cla() #clear the axis

    #plots the histogram data
    width = (bins[1] - bins[0]) * 0.4
    bins_shifted = bins + width
    ax1.bar(bins[:-1], n[0], width, align='edge', color=colors[0])
    ax2.bar(bins_shifted[:-1], n[1], width, align='edge', color=colors[1])

    #finishes the plot
    ax1.set_ylabel("Count", color=colors[0])
    ax2.set_ylabel("Count", color=colors[1])
    ax1.tick_params('y', colors=colors[0])
    ax2.tick_params('y', colors=colors[1])
    plt.tight_layout()
    plt.savefig("%s/%s.png" %(FLAGS.report_dir, filename))
    plt.close()
    return

@ops.RegisterGradient("GuidedRelu")
def _GuidedReluGrad(op, grad):
    return tf.where(0. < grad, gen_nn_ops.relu_grad(grad, op.outputs[0]), tf.zeros_like(grad))

def test():
    """Eval CNN for a number of steps."""
    with tf.Graph().as_default() as g:
        with g.gradient_override_map({'Relu': 'GuidedRelu'}):
            training = tf.placeholder(tf.bool, shape=[])
            # Data loading
            agent = AgentBind(batch_size=FLAGS.batch_size,
                                    seq_size=FLAGS.seq_size,
                                    n_channels=FLAGS.n_channels,
                                    n_classes=FLAGS.n_classes)
            test_dataset = agent.load(data_dir=FLAGS.data_dir+"/vis-samples/",
                                    n_samples=FLAGS.n_samples,
                                    n_epochs=None, shuffle=False,
                                    n_info=1)
            iterator = test_dataset.make_one_shot_iterator()
            seqs, labels, info = iterator.get_next()

            # Build a Graph that computes the logits predictions from the
            # inference model.
            logits = agent.infer(seqs, training, trainable_conv_layers=False)
            weights = agent.analyze(logits, seqs)

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
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                else:
                    print('No checkpoint file found')
                    return

                # Start the run.
                weights_value_list = []
                logits_value_list = []
                info_value_list = []
                seqs_value_list = []
                
                num_iter = int(math.ceil(FLAGS.n_samples / FLAGS.batch_size))
                for step in range(num_iter): 
                    try:
                        weights_value, logits_value, info_value, seqs_value = sess.run(
                                            [weights, logits, info, seqs],
                                            feed_dict={training:False})
                        weights_value_list.append(weights_value)
                        logits_value_list.append(logits_value)
                        info_value_list.append(info_value)
                        seqs_value_list.append(seqs_value)
                    except tf.errors.OutOfRangeError:
                        break
                weights_value_total = np.concatenate(weights_value_list, axis = 0)
                logits_value_total = np.concatenate(logits_value_list, axis = 0)
                info_value_total = np.concatenate(info_value_list, axis = 0)
                seqs_value_total = np.concatenate(seqs_value_list, axis=0)
                _print_summary(weights_value_total, logits_value_total,
                            info_value_total, seqs_value_total, FLAGS.n_samples)

    return

def main(argv=None):  # pylint: disable=unused-argument
    if tf.gfile.Exists(FLAGS.eval_dir):
        tf.gfile.DeleteRecursively(FLAGS.eval_dir)
    tf.gfile.MakeDirs(FLAGS.eval_dir)
 
    if tf.gfile.Exists(FLAGS.figure_dir):
        tf.gfile.DeleteRecursively(FLAGS.figure_dir)
    tf.gfile.MakeDirs(FLAGS.figure_dir)

    test()

if __name__ == '__main__':
    tf.app.run()

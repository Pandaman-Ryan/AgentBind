from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
from sklearn.metrics import roc_auc_score
import tensorflow as tf
from model import AgentBind

FLAGS = tf.app.flags.FLAGS

# parameters for input data
tf.app.flags.DEFINE_string('data_dir', None, """Directory where to read input data""")
tf.app.flags.DEFINE_string('valid_dir', None,"""Directory where to read validation data""")
tf.app.flags.DEFINE_string('checkpoint_dir', None, """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_string('train_dir', None, """Directory where to write event logs and checkpoint.""")
tf.app.flags.DEFINE_integer('seq_size', None, """The length of each sample.""")
tf.app.flags.DEFINE_integer('n_train_samples', None, """Number of input samples in the training process.""")
tf.app.flags.DEFINE_integer('n_valid_samples', None, """Number of input samples in the validation process.""")

# parameters for model training
tf.app.flags.DEFINE_integer('batch_size', 128, """The size of each batch.""")
tf.app.flags.DEFINE_integer('n_classes', 1, """Number of categories in data.""")
tf.app.flags.DEFINE_integer('n_channels', 4, """Number of input channels.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_float('moving_average_decay', 0.999,
                            """Moving average decay.""")
tf.app.flags.DEFINE_float('initial_learning_rate', 0.001,
                            """Initial learning rate.""")
tf.app.flags.DEFINE_integer('n_steps_per_epoch', None, """Number of steps per epoch.""")
tf.app.flags.DEFINE_integer('n_validation_steps', None, """Number of steps in the validation process.""")


class trainer:
    def __init__(self):
        if tf.gfile.Exists(FLAGS.train_dir):
            tf.gfile.DeleteRecursively(FLAGS.train_dir)
        tf.gfile.MakeDirs(FLAGS.train_dir)

    def start_train(self):
        with tf.Graph().as_default():
            global_step = tf.train.get_or_create_global_step()
            agent = AgentBind(batch_size=FLAGS.batch_size,
                                    seq_size=FLAGS.seq_size,
                                    n_channels=FLAGS.n_channels,
                                    n_classes=FLAGS.n_classes)


            ## Data loading
            training_dataset = agent.load(data_dir=FLAGS.data_dir,
                                        n_samples=FLAGS.n_train_samples,
                                        n_epochs=None, shuffle=True)
            training_handle_tensor = training_dataset.make_one_shot_iterator().string_handle()
            validation_dataset = agent.load(data_dir=FLAGS.valid_dir,
                                        n_samples=FLAGS.n_valid_samples,
                                        n_epochs=None, shuffle=False)
            validation_iterator = validation_dataset.make_initializable_iterator()
            validation_handle_tensor = validation_iterator.string_handle()

            # iterator
            handle = tf.placeholder(tf.string, shape=[], name='handle')
            iterator = tf.data.Iterator.from_string_handle(handle,
                                    training_dataset.output_types, training_dataset.output_shapes)

            # learning rate
            learning_rate = tf.placeholder(tf.float32, shape=[])
            training = tf.placeholder(tf.bool, shape=[])

            ## Graph
            seqs, labels = iterator.get_next()
            logits = agent.infer(seqs, training, trainable_conv_layers=True)
            #logits_sigmoid = tf.sigmoid(logits)
            loss = agent.loss(logits, labels)
            train_op = agent.train(loss, global_step, learning_rate,
                                    moving_average_decay=FLAGS.moving_average_decay)
            # saver
            saver = tf.train.Saver(max_to_keep=1)

            # restore bottom layers
            loader = tf.train.Saver({\
                    u'conv1/weights/ExponentialMovingAverage':\
                            tf.get_default_graph().get_tensor_by_name(u'conv1/weights:0'),\
                    u'conv1/biases/ExponentialMovingAverage':\
                            tf.get_default_graph().get_tensor_by_name(u'conv1/biases:0'),\
                    u'brnn/bidirectional_rnn/fw/lstm_cell/kernel/ExponentialMovingAverage':\
                            tf.get_default_graph().get_tensor_by_name(u'brnn/bidirectional_rnn/fw/lstm_cell/kernel:0'),\
                    u'brnn/bidirectional_rnn/fw/lstm_cell/bias/ExponentialMovingAverage':\
                            tf.get_default_graph().get_tensor_by_name(u'brnn/bidirectional_rnn/fw/lstm_cell/bias:0'),\
                    u'brnn/bidirectional_rnn/bw/lstm_cell/kernel/ExponentialMovingAverage':\
                            tf.get_default_graph().get_tensor_by_name(u'brnn/bidirectional_rnn/bw/lstm_cell/kernel:0'),\
                    u'brnn/bidirectional_rnn/bw/lstm_cell/bias/ExponentialMovingAverage':\
                            tf.get_default_graph().get_tensor_by_name(u'brnn/bidirectional_rnn/bw/lstm_cell/bias:0')})
            

            ######
            # Helper tools
            ######
            def _print_summary(losses, epoch_index, tag,
                                lr=None, loss_best=None,
                                logits=None, labels=None):
                loss_avg = sum(losses)/float(len(losses))
                if tag == "Training":
                    print ("Loss of %s in epoch %d is: %f" %(tag, epoch_index, loss_avg))
                elif tag == "Validation":
                    num_of_features = (logits.shape)[1]
                    for feature_index in range(num_of_features):
                        lbl = labels[:, feature_index]
                        lgt = logits[:, feature_index]

                        ###
                        # AUC
                        auc_value = roc_auc_score(lbl, lgt)
                        
                        ###
                        # precision & recall
                        num_of_samples = (logits.shape)[0]
                        true_pos = 0
                        cond_pos = 0
                        pred_pos = 0
                        for sample_index in range(num_of_samples):
                            if lbl[sample_index] == 1:
                                cond_pos += 1
                            if lgt[sample_index] > 0.5:
                                pred_pos += 1
                            if lbl[sample_index] == 1 and lgt[sample_index] > 0.5:
                                true_pos += 1
                        recall = -1 if (cond_pos == 0) else float(true_pos) / float(cond_pos)
                        precision = -1 if (pred_pos == 0) else float(true_pos) / float(pred_pos)

                        # print results
                        print ("==> AUC: %.3f; Precision: %.3f; Recall: %.3f; lr: %f; Best loss: %f"\
                                                %(auc_value, precision, recall, lr, loss_best))
                    # print the loss value of the whole validation set
                    print ("==> Loss of %s in epoch %d is: %f" %(tag, epoch_index, loss_avg))
                else:
                    exit("Wrong tag: function _print_summary in train.py")
                return

            def _update_learning_rate(losses, loss_best, lr, n_red_flags):
                loss_avg = sum(losses)/float(len(losses))
                if (loss_best == -1) or (loss_avg < loss_best):
                    lr_updated = lr
                    to_save = True
                    loss_best = loss_avg
                    n_red_flags = 0
                else:
                    n_red_flags += 1
                    to_save = False
                    lr_updated = lr
                    if n_red_flags >= 10:
                        lr_updated = lr/10.0
                        n_red_flags = 0
                return lr_updated, loss_best, n_red_flags, to_save

            ######
            # Train
            ######
            with tf.Session(
                config=tf.ConfigProto(allow_soft_placement=True,
                    log_device_placement=False,
                    gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.9))) as sess:
                # Initialization
                training_handle, validation_handle = sess.run(
                                                [training_handle_tensor, validation_handle_tensor])
                sess.run(tf.global_variables_initializer())
                sess.run(tf.local_variables_initializer())
                epoch_index = 0
                learning_rate_value = FLAGS.initial_learning_rate
                loss_best = -1
                n_red_flags = 0

                # load data
                ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
                if ckpt and ckpt.model_checkpoint_path:
                    # Restores from checkpoint
                    loader.restore(sess, ckpt.model_checkpoint_path)
                else:
                    print('No checkpoint file found')
                    return

                # Training
                while True:
                    epoch_index += 1
                    ######################
                    # Training process
                    n_steps_per_epoch = FLAGS.n_steps_per_epoch\
                                            if (FLAGS.n_steps_per_epoch != None)\
                                            else int(FLAGS.n_train_samples/float(FLAGS.batch_size)) + 1
                    n_steps_training_log = max(int(n_steps_per_epoch/10.0), 10)
                    losses = []
                    for step_index in range(n_steps_per_epoch):
                        loss_value, _ = sess.run([loss, train_op], 
                                feed_dict={handle:training_handle,
                                            learning_rate:learning_rate_value,
                                            training:True})
                        losses.append(loss_value)
                        if (step_index % (n_steps_training_log)) == (n_steps_training_log-1):
                            _print_summary(losses, epoch_index, "Training")
                            losses = []

                    #######################
                    # Validation process

                    # init
                    sess.run(validation_iterator.initializer)
                    losses = []
                    logits_value_list = []
                    labels_value_list = []

                    # run a step
                    n_validation_steps = FLAGS.n_validation_steps\
                                            if (FLAGS.n_validation_steps != None)\
                                            else int(FLAGS.n_valid_samples/float(FLAGS.batch_size)) + 1
                    for step_index in range(n_validation_steps):
                        label_value, logit_value, loss_value =\
                                                sess.run([labels, logits, loss],
                                                feed_dict={handle:validation_handle,
                                                            training:False})
                        losses.append(loss_value)
                        logits_value_list.append(logit_value)
                        labels_value_list.append(label_value)

                    # calculate stats
                    logits_value_total = np.concatenate(logits_value_list, axis = 0)
                    labels_value_total = np.concatenate(labels_value_list, axis = 0)

                    # update learning rate
                    learning_rate_value, loss_best, n_red_flags, to_save =\
                                            _update_learning_rate(losses, loss_best,\
                                                           learning_rate_value, n_red_flags)
                    _print_summary(losses, epoch_index, "Validation",
                                        learning_rate_value, loss_best,
                                        logits_value_total, labels_value_total)

                    # save model
                    if to_save:
                        ckpt = os.path.join(FLAGS.train_dir, "model.ckpt")
                        saver.save(sess, ckpt, global_step)

                    # conditions of exiting the loop
                    if learning_rate_value <= 1e-5:
                        print ("Learning rate is low: %f" %(learning_rate_value))
                        break

def main(argv=None):
    trainer_ins = trainer()
    trainer_ins.start_train()

if __name__ == "__main__":
    tf.app.run()
#########################

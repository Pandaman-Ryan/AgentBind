'''
    Agent Bind DanQ model
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys 
import tarfile

from six.moves import urllib
import tensorflow as tf
from load import DataLoader

class AgentBind():
    def __init__(self, batch_size, seq_size, n_channels, n_classes):
        self.batch_size = batch_size
        self.seq_size = seq_size
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.cam_layer = None
        return

    def load(self, data_dir, n_samples, n_epochs, shuffle=False, n_info=0):
        if not data_dir:
            raise ValueError('Please supply a data_dir')
        else:
            filename = os.path.join(data_dir, 'data.txt')
            flnms = [filename]

        loader = DataLoader(batch_size=self.batch_size,
                            seq_size=self.seq_size, n_channels=self.n_channels,
                            n_classes=self.n_classes, n_info=n_info)
        dataset = loader.inputs(filenames=flnms, n_samples=n_samples,
                                        n_epochs=n_epochs, shuffle=shuffle)
        return dataset

    def infer(self, samples, training, trainable_conv_layers=True):
        # convolutional layers
        conv1 = self._conv_layer(inputs=samples, scope_name='conv1',
                        filter_width=26, in_channels=4, out_channels=320, conv_stride=1,
                        pool_op=True, pool_window_size=13, pool_stride=13, training=training,
                        trainable_parameters=trainable_conv_layers, dropout=0.2, record='conv')

        conv_final = conv1
        conv_final_length = conv_final.get_shape()[1]
        conv_final_depth = conv_final.get_shape()[2]
        n_hidden = conv_final_depth

        brnn = self._brnn_layer(inputs=conv_final, scope_name='brnn',
                n_steps=conv_final_length, n_hidden=n_hidden,
                training=training, trainable_parameters=True, dropout=0.5)

        # fully connected layers
        dim = conv_final_length * n_hidden * 2
        reshaped = tf.reshape(brnn, [-1, dim])
        fc1 = self._fully_connected_layer(reshaped, scope_name='fc1',
                        in_neurons=dim, out_neurons=925,
                        activated=True, training=training,
                        trainable_parameters=True)

        output_layer = self._fully_connected_layer(fc1, scope_name='fc2',
                        in_neurons=925, out_neurons=self.n_classes,
                        activated=False, training=training,
                        trainable_parameters=True)
        return output_layer

    def loss(self, logits, labels):
        """Add L2Loss to all the trainable variables.
        Add summary for "Loss" and "Loss/avg".
        Args:
            logits: Logits from inference().
            labels: Labels from distorted_inputs or inputs(). 1-D tensor
                of shape [batch_size]
        Returns:
            Loss tensor of type float.
        """
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
                            labels=labels, logits=logits,
                            name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        tf.add_to_collection('losses', cross_entropy_mean)
        
        # The total loss is defined as the cross entropy loss plus all of the weight
        # decay terms (L2 loss).
        loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
        return loss

    def train(self, total_loss, global_step,
                learning_rate, moving_average_decay):
        """Train CNN model.
        Create an optimizer and apply to all trainable variables. Add moving
        average for all trainable variables.
        Args:
            total_loss: Total loss from loss().
            global_step: Integer Variable counting the number of training steps
                    processed.
            learning_rate: a float32 tensor
            moving_average_Decay: a float32 number
        Returns:
            train_op: op for training.
        """

        # compute gradients
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            opt = tf.train.AdamOptimizer(learning_rate)
            grads = opt.compute_gradients(total_loss)

            # Apply gradients.
            apply_gradients_op = opt.apply_gradients(grads, global_step=global_step)

            # Track the moving averages of all trainable variables
            variable_averages = tf.train.ExponentialMovingAverage(
                                    moving_average_decay, global_step)
            variables_averages_op = variable_averages.apply(tf.trainable_variables())

            # hold up until everything finishes
            with tf.control_dependencies([apply_gradients_op, variables_averages_op]):
                train_op = tf.no_op(name='train')
        return train_op

    def analyze(self, outputs, inputs):
        """Compute gradients
        Args:
            output tensor
            input tensor
        Returns:
            grads: gradients.
        """
        # compute gradients
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            grads = tf.gradients(outputs, inputs)
            grads = tf.squeeze(grads)
        return grads

    def retrieve_cam_layer(self):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            return self.cam_layer

    def compute_cam_gradient(self, outputs):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            grads = tf.gradients(outputs, self.cam_layer)
            grads = tf.squeeze(grads)
        return grads


    ##################
    # local functions
    ##################
    #####
    # Layers
    def _conv_layer(self, inputs, scope_name,
                    filter_width, in_channels, out_channels, conv_stride,
                    pool_op=False, pool_window_size=None, pool_stride=None,
                    training=None, trainable_parameters=None, dropout=None, record=None):
        """
        conv1d parameters:
        value: [batch, in_width, in_channels]
        filter: [filter_width, in_channels, out_channels]
        stride: an integer
        """
        with tf.variable_scope(scope_name) as scope:
            kernel = self._variable_with_weight_decay('weights',
                            shape=[filter_width, in_channels, out_channels],
                            stddev=1e-2, wd_l1=1e-8, wd_l2=5e-7, trainable=trainable_parameters)
            conv = tf.nn.conv1d(value=inputs, filters=kernel,
                                stride=conv_stride, padding='VALID')
            biases = tf.get_variable(name='biases', shape=[out_channels],
                                    initializer=tf.constant_initializer(0.0),
                                    trainable=trainable_parameters)
            pre_activation = tf.nn.bias_add(conv, biases)
            conv_activated = tf.nn.relu(pre_activation)
            outputs = conv_activated

            if record == "conv":
                self.cam_layer = outputs

            if pool_op:
                ## pool
                pooled = tf.nn.pool(conv_activated, window_shape=[pool_window_size],
                            pooling_type='MAX', padding='VALID', strides=[pool_stride],
                            data_format='NWC')
                outputs = pooled
                if record == "pool":
                    self.cam_layer = outputs

            if dropout != None:
                dropoutted = tf.layers.dropout(outputs, rate=dropout, training=training)
                outputs = dropoutted
        return outputs
    
    def _fully_connected_layer(self, inputs, scope_name,
                            in_neurons, out_neurons,
                            activated, training, trainable_parameters):
        """
        FC layer parameters:
        """
        with tf.variable_scope(scope_name) as scope:
            weights = self._variable_with_weight_decay('weights',
                                shape=[in_neurons, out_neurons],
                                stddev=1e-2, wd_l1=1e-8, wd_l2=5e-7, trainable=trainable_parameters)
            biases = tf.get_variable(name='biases', shape=[out_neurons],
                                    initializer=tf.constant_initializer(0.0),
                                    trainable=trainable_parameters)
            if activated:
                fc = tf.nn.relu(tf.matmul(inputs, weights)+biases)
            else:
                fc = tf.add(tf.matmul(inputs, weights), biases)
        return fc

    def _brnn_layer(self, inputs, scope_name, n_steps, n_hidden,
                    training=None, trainable_parameters=None, dropout=None):
        with tf.variable_scope(scope_name) as scope:
            lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(n_hidden)
            lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(n_hidden)
            outputs_fw_bw, _ =\
                        tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, inputs, dtype=tf.float32)
            outputs = tf.concat(outputs_fw_bw, 2)
            if dropout != None:
                dropoutted = tf.layers.dropout(outputs, rate=dropout, training=training)
                outputs = dropoutted
        return outputs
    ######
    # Variables
    def _variable_with_weight_decay(self, name, shape, stddev, wd_l1, wd_l2, trainable):
        """Helper to create an initialized Variable with weight decay.
        Note that the Variable is initialized with a truncated normal distribution.
        A weight decay is added only if one is specified.
        Args:
            name: name of the variable
            shape: list of ints
            stddev: standard deviation of a truncated Gaussian
            wd: add L1Loss/L2Loss weight decay multiplied by this float. If None, weight
                decay is not added for this Variable.
        Returns:
            Variable Tensor
        """
        var = tf.get_variable(
            name,
            shape,
            initializer=tf.truncated_normal_initializer(stddev=stddev),
            trainable=trainable)
        if wd_l2 is not None:
            weight_decay = tf.multiply(tf.nn.l2_loss(var), wd_l2, name='weight_loss')
            tf.add_to_collection('losses', weight_decay)
        if wd_l1 is not None:
            sparsity = tf.multiply(tf.nn.l2_loss(var), wd_l1, name='weight_loss')
            tf.add_to_collection('losses', sparsity)
        return var

####### END OF FILE ########

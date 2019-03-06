'''
    --> Vampire-Hunter <--
    Created by: An Zheng
    Modified by: An Zheng
    Last Update: Jan 31th, 2018
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

class VampireHunter():
    def __init__(self, batch_size, seq_size, n_channels, n_classes):
        self.batch_size = batch_size
        self.seq_size = seq_size
        self.n_channels = n_channels
        self.n_classes = n_classes
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
        samples = tf.identity(samples,  name='absolute_input')
        # convolutional layers
        conv1 = self._conv_layer(inputs=samples, scope_name='conv1',
                        filter_width=11, in_channels=4, out_channels=512, conv_stride=2,
                        pool_op=False, training=training,
                        trainable_parameters=trainable_conv_layers)
        conv2 = self._conv_layer(inputs=conv1, scope_name='conv2',
                        filter_width=5, in_channels=512, out_channels=512, conv_stride=2,
                        pool_op=False, training=training,
                        trainable_parameters=trainable_conv_layers)
        conv3 = self._conv_layer(inputs=conv2, scope_name='conv3',
                        filter_width=3, in_channels=512, out_channels=256, conv_stride=2,
                        pool_op=False, training=training,
                        trainable_parameters=trainable_conv_layers)
        conv4 = self._conv_layer(inputs=conv3, scope_name='conv4',
                        filter_width=3, in_channels=256, out_channels=256, conv_stride=2,
                        pool_op=False, training=training,
                        trainable_parameters=trainable_conv_layers)
        conv5 = self._conv_layer(inputs=conv4, scope_name='conv5',
                        filter_width=10, in_channels=256, out_channels=128, conv_stride=2,
                        pool_op=False, training=training,
                        trainable_parameters=trainable_conv_layers)
        conv_final = conv5
        conv_final_length = conv_final.get_shape()[1]
        conv_final_depth = conv_final.get_shape()[2]

        # fully connected layers
        dim = conv_final_length * conv_final_depth
        reshaped = tf.reshape(conv_final, [-1, dim])
        fc1 = self._fully_connected_layer(reshaped, scope_name='fc1',
                        in_neurons=dim, out_neurons=2048,
                        activated=True, training=training,
                        trainable_parameters=False)

        fc2 = self._fully_connected_layer(fc1, scope_name='fc2',
                        in_neurons=2048, out_neurons=2048,
                        activated=True, training=training,
                        trainable_parameters=False)

        output_layer = self._fully_connected_layer(fc2, scope_name='fc3',
                        in_neurons=2048, out_neurons=self.n_classes,
                        activated=False, training=training,
                        trainable_parameters=False)

        output_layer = tf.identity(output_layer, name='absolute_output')
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
        labels = tf.cast(labels, tf.float32)
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
                            labels=labels, logits=logits,
                            name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        tf.add_to_collection('losses', cross_entropy_mean)
        
        # The total loss is defined as the cross entropy loss plus all of the weight
        # decay terms (L2 loss).
        loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
        return loss

    def analyze(self, outputs, inputs):
        """Compute gradients
        Args:
            outputs
            inputs
        Returns:
            grads: gradients.
        """
        # compute gradients
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            grads = tf.gradients(outputs, inputs)

        return grads

    ##################
    # local functions
    ##################
    #####
    # Layers
    def _conv_layer(self, inputs, scope_name,
                    filter_width, in_channels, out_channels, conv_stride,
                    pool_op=False, pool_window_size=None, pool_stride=None,
                    training=None, trainable_parameters=None):
        """
        conv1d parameters:
        value: [batch, in_width, in_channels]
        filter: [filter_width, in_channels, out_channels]
        stride: an integer
        """
        with tf.variable_scope(scope_name) as scope:
            kernel = self._variable_with_weight_decay('weights',
                            shape=[filter_width, in_channels, out_channels],
                            stddev=1e-2, wd=5e-4, trainable=trainable_parameters)
            conv = tf.nn.conv1d(value=inputs, filters=kernel,
                                stride=conv_stride, padding='SAME')
            biases = tf.get_variable(name='biases', shape=[out_channels],
                                    initializer=tf.constant_initializer(0.0),
                                    trainable=trainable_parameters)
            pre_activation = tf.nn.bias_add(conv, biases)
            pre_activation_norm = tf.layers.batch_normalization(pre_activation,
                                                            training=training,
                                                            trainable=trainable_parameters)
            conv_activated = tf.nn.relu(pre_activation_norm)
            outputs = conv_activated

            if pool_op:
                ## pool
                pool = tf.nn.pool(conv_activated, window_shape=[pool_window_size],
                            pooling_type='MAX', padding='SAME', strides=[pool_stride],
                            data_format='NWC')
                outputs = pool
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
                                stddev=1e-2, wd=5e-4, trainable=trainable_parameters)
            biases = tf.get_variable(name='biases', shape=[out_neurons],
                                    initializer=tf.constant_initializer(0.0),
                                    trainable=trainable_parameters)
            if activated:
                fc = tf.nn.relu(
                            tf.layers.batch_normalization(
                                tf.matmul(inputs, weights)+biases,
                                training=training,
                                trainable=trainable_parameters))
            else:
                fc = tf.add(tf.matmul(inputs, weights), biases)
        return fc

    ######
    # Variables
    def _variable_with_weight_decay(self, name, shape, stddev, wd, trainable):
        """Helper to create an initialized Variable with weight decay.
        Note that the Variable is initialized with a truncated normal distribution.
        A weight decay is added only if one is specified.
        Args:
            name: name of the variable
            shape: list of ints
            stddev: standard deviation of a truncated Gaussian
            wd: add L2Loss weight decay multiplied by this float. If None, weight
                decay is not added for this Variable.
        Returns:
            Variable Tensor
        """
        var = tf.get_variable(
            name,
            shape,
            initializer=tf.truncated_normal_initializer(stddev=stddev),
            trainable=trainable)
        if wd is not None:
            weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
            tf.add_to_collection('losses', weight_decay)
        return var

####### END OF FILE ########

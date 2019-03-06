# This file contains functions for data reading
# and pre-processing
# 
# ==============================================================================
"""Routine for reading DNase-seq data from datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class DataLoader():
    def __init__(self, batch_size, seq_size, n_channels, n_classes, n_info):
        self.batch_size = batch_size
        self.seq_size = seq_size
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.n_info = n_info

    def inputs(self, filenames, n_samples, n_epochs, shuffle):
        """Construct distorted input using the Reader ops
        Args:
            filenames: a list of filenames to read from
            n_samples: the number of samples in the input dataset (to estimate buffer size)
            n_epochs: the number of epochs to run. "-1" or None means running repeatedly
            shuffle: use shuffle_queue or not
            n_info: number of additional features of each sample
        Returns:
            dataset: a prepared dataset
                which contains (seqs, labels):
                seqs: Genome sequences. 3D tensor of [batch_size, length, #nucleotides] size.
                labels: Labels. 2D tensor of [batch_size, number_of_cell_types] size.
        """
        for f in filenames:
            if not tf.gfile.Exists(f):
                raise ValueError("Failed to find file: "+f)

        def parser(record):
            dataset = self._parser(record)
            return dataset

        dataset = tf.data.TextLineDataset(filenames)
        dataset = dataset.map(parser)
        if shuffle:
            base_fraction_of_examples = 0.1
            base_examples = int(min(n_samples * base_fraction_of_examples, 10000))
            buffer_size = base_examples + 3 * self.batch_size
            dataset = dataset.shuffle(buffer_size=buffer_size)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.repeat(n_epochs)
        return dataset

    ###########
    # private functions
    def _parser(self, record):
        ####
        # prepare data and labels
        misc_info = tf.string_split(source=[record], delimiter=";").values

        # sequence prep
        seq = (tf.string_split(source=[misc_info[0]], delimiter=",")).values
        seq.set_shape([self.seq_size*self.n_channels])
        seq = tf.string_to_number(
                tf.reshape(seq, [self.seq_size, self.n_channels]),
                tf.float32)

        # label prep
        label = (tf.string_split(source=[misc_info[1]],delimiter=",")).values
        label = label[:self.n_classes]
        label.set_shape([self.n_classes])
        label = tf.string_to_number(tf.reshape(label, [self.n_classes]), tf.float32)

        if self.n_info == 0:
            return seq, label
        else:
            info = (tf.string_split(source=[misc_info[2]],delimiter=",")).values
            info = info[:self.n_info]
            info.set_shape([self.n_info])
            return seq, label, info
########### END OF FILE ###############

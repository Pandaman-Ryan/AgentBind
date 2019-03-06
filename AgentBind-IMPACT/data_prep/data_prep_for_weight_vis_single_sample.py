'''
    data_prep.py
    
    Last edit: July 2nd, 2018

    This program is used to generate training/validation/test data
    for DNN model training.

    Input:
        fimo.txt: all positions of a motif in the whole human genome
        narrowPeak file: overlaps will be labelled as positives
    Output:
        Positives/Negtives:Training/Validation/Test
'''

import os
import random
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from Bio.Seq import Seq
from converter import converter_template
from optparse import OptionParser

class datasetPrep(converter_template):
    def __init__(self):
        return

    def prepare_labelled_samples(self, dir_data, TF_ID):
        '''
            read samples from fimo_file,
            and label them as positive samples or negatives samples
            according to whether there are matches found in the
            ENCODE dataset or not
                                                                                                                                                                                                                           
            return labelled data in the python list format
        '''
        data_path = "%s/IMPACT/train_test_positive_bed_%sonly_center.txt" %(dir_data, TF_ID)
        labelled_list, n_pos = self._readFromImpact(data_path, 1)
        return labelled_list, n_pos

    def convert(self, samples_coord, len_seq, dir_data):
        '''
            read the sequences of labelled samples

            len_seq: the length of each sequence
        '''
        reference_dir = "%s/genomes/hg19/" %(dir_data)
        genome_limit_path = "%s/genomes/hg19/hg19.fa.fai" %(dir_data)
        reference_genome_dict, chrom_length_dict =\
                self._load_ref_genome(reference_dir, genome_limit_path)

        sequences = []
        # read from reference genome
        for smp in samples_coord:
            (chromID, start, end, label) = smp
            seq_start = start-int((len_seq-(end-start))/2.0)
            seq_end = seq_start + len_seq

            if (chromID in chrom_length_dict) \
                    and (seq_start >= 0 and seq_end < (chrom_length_dict)[chromID]):
                seq = self._readFromRef(chromID, seq_start, seq_end, reference_genome_dict)
                #seq = self._readFromRef(chromID, seq_start, start-4, reference_genome_dict)\
                #        + "".join(['N' for _ in range(10)])\
                #        + self._readFromRef(chromID, end+5, seq_end, reference_genome_dict)
                sequences.append((chromID, seq_start, seq_end, seq, label))

        return sequences

    def make_datasets(self, samples):
        '''
            divide data into training/validation/test datasets
        '''
        #################
        # separate data into training/validation/test
        samples_test = []
        for smpl in samples:
            (chromID, seq_start, seq_end, seq, label) = smpl
            samples_test.append((chromID, seq_start, seq_end, seq, label)) 
        seqs_test = self._prepare_dataset(samples_test)
        return seqs_test

    def save_into_file(self, data, data_dir, tag):
        data_dir = os.path.join(data_dir, tag)
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        output_path = os.path.join(data_dir, "data.txt")

        with open(output_path, 'w') as ofile:
            for dt in data:
                chromID, seq_start, seq_end, seq, label = dt
                line_to_save = "%s;%d;%s,%d,%d\n" %(seq, label, chromID, seq_start, seq_end)
                ofile.write(line_to_save)
        #print ("Size of %s samples: %d" %(tag, len(data)))
        return

    ##################
    ## Private functions
    ##################
    def _readFromImpact(self, ifilename, label):
        smp_list = []
        for line in open(ifilename):
            #chr1    101776157   101776158   Foxp3_Treg  0   + 
            line = line.strip()
            elems = line.split()
            chromID = elems[0]
            start_pos = int(elems[1])
            end_pos = int(elems[2])

            smp = (chromID, start_pos, end_pos, label)
            smp_list.append(smp)
        
        return smp_list, len(smp_list)

    def _prepare_dataset(self, seqs):
        total_seqs = []
        for (chromID, seq_start, seq_end, seq, label) in seqs:
            total_seqs.append((chromID, seq_start, seq_end, self._string_to_one_hot(seq), label))
        return total_seqs

def main():
    usage = "usage: %prog [options]"
    parser = OptionParser(usage)
    parser.add_option('--scope', dest='len_seq', default=None, type='int',
            help='The scope of contexts to be examined for each core motif [Default: %default]')
    parser.add_option('--resultdir', dest='dir_result', default=None,
            help='The directory where all temporary and final results are saved [Default: %default]')
    parser.add_option('--datadir', dest='dir_data', default=None,
            help='The directory where all input and dependent data are stored [Default: %default]')
    parser.add_option('--TF_name', dest='TF_name', default=None,
            help='The name of the core TF [Default: %default]')
    (options, args) = parser.parse_args()

    #####
    # start to prepare and format data
    dp = datasetPrep()
    labelled_list, n_pos = dp.prepare_labelled_samples(options.dir_data, options.TF_name)
    sequences = dp.convert(labelled_list, options.len_seq, options.dir_data)
    test = dp.make_datasets(sequences)
    dp.save_into_file(test, options.dir_result, tag = ("vis-samples/"))

    f_n_samples = "%s/vis-samples/num_of_samples.txt" %(options.dir_result)
    with open(f_n_samples, 'w') as ofile:
        ofile.write("%d" %(len(test)))

if __name__ == "__main__":
    main()
###END OF FILE###########################

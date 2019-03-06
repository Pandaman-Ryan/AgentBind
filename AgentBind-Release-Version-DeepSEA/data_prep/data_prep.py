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

    def prepare_labelled_samples(self, fimo_file, narrowPeak_file,
                                    pval_bin_size, pos_neg_ratio):
        '''
            read samples from fimo_file,
            and label them as positive samples or negatives samples
            according to whether there are matches found in the
            ENCODE dataset or not
                                                                                                                                                                                                                           
            return labelled data in the python list format
        '''
        fimo_list = self._readFromFimo(fimo_file)
        chip_dict = self._readFromNarrowPeak(narrowPeak_file)
        #####
        #TODO
        #dnase_file = "/storage/pandaman/project/VampireHunter/DNase_ENCODE/wgEncodeAwgDnaseUwdukeHepg2UniPk.narrowPeak"
        #dnase_dict = self._readFromNarrowPeak(dnase_file)
        #labelled_list = self._labelSamples(fimo_list, chip_dict, dnase_dict)
        labelled_list = self._labelSamples(fimo_list, chip_dict)
        labelled_list, n_pos, n_neg = self._balancePosAndNeg(
                            labelled_list, pval_bin_size, pos_neg_ratio)
        return labelled_list, n_pos, n_neg

    #def convert(self, samples_coord, len_seq, dir_data, block_core, chromID_validation, chromID_test):
    def convert(self, samples_coord, len_seq, dir_data, block_core):
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
            (chromID, start, end, label, pval, signalVal) = smp
            seq_start = start-int((len_seq-(end-start))/2.0)
            seq_end = seq_start + len_seq
            if seq_start >= 0 and seq_end < (chrom_length_dict)[chromID]:
                if block_core == 'c':
                    seq = self._readFromRef(chromID, seq_start, seq_end, reference_genome_dict)
                elif block_core == 'b':
                    seq = self._readFromRef(chromID, seq_start, start, reference_genome_dict)\
                        + "".join(['N' for _ in range(end-start)])\
                        + self._readFromRef(chromID, end, seq_end, reference_genome_dict)
                else:
                    exit('wrong symbol of block_core: %s' %(block_core))
                sequences.append((chromID, (seq, label), signalVal, pval))
        return sequences

    def make_datasets(self, samples, chromID_validation, chromID_test):
        '''
            divide data into training/validation/test datasets
        '''
        #################
        # shuffle
        random.shuffle(samples)

        #################
        # separate data into training/validation/test
        samples_training = []
        samples_validation = []
        samples_test = []
        for smpl in samples:
            (chromID, (seq, label), signalVal, pval) = smpl
            if chromID == chromID_validation:
                samples_validation.append((seq, label, signalVal, pval))
            elif chromID == chromID_test:
                samples_test.append((seq, label, signalVal, pval))
            else:
                samples_training.append((seq, label, signalVal, pval))
        
        seqs_training = self._prepare_dataset(samples_training)
        seqs_validation = self._prepare_dataset(samples_validation)
        seqs_test = self._prepare_dataset(samples_test)
        
        return seqs_training, seqs_validation, seqs_test

    def save_into_file(self, data, data_dir, tag):
        data_dir = os.path.join(data_dir, tag)
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        output_path = os.path.join(data_dir, "data.txt")

        with open(output_path, 'w') as ofile:
            for dt in data:
                seq, label, signalVal, pval = dt
                if tag == "test":
                    line_to_save = "%s;%d;%f,%s\n" %(seq, label, signalVal, pval)
                else:
                    line_to_save = "%s;%d\n" %(seq, label)
                ofile.write(line_to_save)
        #print ("Size of %s samples: %d" %(tag, len(data)))
        return

    def save_test_coordinate(self, data, data_dir, chromID_validation, chromID_test):
        data_to_save = {0:[], 1:[]}
        for dt in data:
            (chromID, start, end, label, pval, signalVal) = dt
            #if chromID == chromID_test:
            if True:
                line_to_save = "%s;%d;%d;%s;%s\n" %(chromID, start, end, signalVal, pval)
                data_to_save[label].append(line_to_save)

        vis_data_dir = os.path.join(data_dir, "vis-weights-total/")
        if not os.path.exists(vis_data_dir):
            os.makedirs(vis_data_dir)
        output_path_pos = os.path.join(vis_data_dir, "coordinates_pos.txt")
        with open(output_path_pos, 'w') as ofile:
            for dt in data_to_save[1]:
                ofile.write(dt)

        output_path_neg = os.path.join(vis_data_dir, "coordinates_neg.txt")
        with open(output_path_neg, 'w') as ofile:
            for dt in data_to_save[0]:
                ofile.write(dt)
        return

    ##################
    ## Private functions
    ##################
    #def _labelSamples(self, samples_list, ref_dict, boundary_dict):
    def _labelSamples(self, samples_list, ref_dict):
        '''
            check each sample which contains the given motif
            and search for its matches in the ENCODE dataset.

            If a match is found, then this sample is considered
            as a positive sample.
        '''
        # examine each samples
        labelled_list = []
        for spl in samples_list:
            (chromID, start, end, pval) = spl
            # ignore samples within uncommon chromosomes
            if chromID not in ref_dict:
                continue
            
            #dnase_hyper = False
            #for ref in boundary_dict[chromID]:
            #    (chromID_ref, start_ref, end_ref, signalVal) = ref
            #    if (start_ref<start) and (end<end_ref):
            #        dnase_hyper = True
            #        break
            #if dnase_hyper == False:
            #    continue
           
            # if satisfied requirements above:
            #
            # scan whole ENCODE data to find if there is a match
            label = 0
            sample_signal_value = 0
            for ref in ref_dict[chromID]:
                (chromID_ref, start_ref, end_ref, signalVal) = ref
                if (start_ref<start) and (end<end_ref):
                    label = 1
                    sample_signal_value = signalVal
                    break

            # store this sample into a data-list
            labelled_list.append((chromID, start, end, label, pval, sample_signal_value))
        return labelled_list

    def _balancePosAndNeg(self, data_list, bin_size, pos_neg_ratio):
        ####
        # put samples in different bins
        # according to their pval
        data_dict = {}
        for sample in data_list:
            (chromID, start, end, label, pval, signalVal) = sample
            key = int(math.floor((-math.log10(pval))/float(bin_size)))
            if key not in data_dict:
                data_dict[key] = {0:[], 1:[]}
            data_dict[key][label].append(sample)
        
        ####
        # select samples from each bin
        # and make sure the selected positive samples is more or equal
        # than number of negative samples
        labelled_list = []
        n_pos = 0
        n_neg = 0
        for key in data_dict:
            # positive samples
            n_pos += len(data_dict[key][1])
            labelled_list += data_dict[key][1]

            # negative samples
            num_samples = min(len(data_dict[key][0]),
                            int(len(data_dict[key][1])/float(pos_neg_ratio)))
            selected_neg_samples = random.sample(data_dict[key][0], num_samples)
            n_neg += num_samples
            labelled_list += selected_neg_samples
        return labelled_list, n_pos, n_neg

    def _prepare_dataset(self, seqs):
        total_seqs = []
        for (seq,label,signalVal,pval) in seqs:
            total_seqs.append((self._string_to_one_hot(seq), label, signalVal, pval))
            total_seqs.append((self._string_to_one_hot(self._reverse_complement(seq)),
                                label, signalVal, pval))
        random.shuffle(total_seqs)
        return total_seqs

def main():
    usage = "usage: %prog [options]"
    parser = OptionParser(usage)
    parser.add_option('--fimo_file', dest='fimo_file', default=None,
            help='Input fimo file address [Default: %default]')
    parser.add_option('--scope', dest='len_seq', default=None, type='int',
            help='The scope of contexts to be examined for each core motif [Default: %default]')
    parser.add_option('--resultdir', dest='dir_result', default=None,
            help='The directory where all temporary and final results are saved [Default: %default]')
    parser.add_option('--datadir', dest='dir_data', default=None,
            help='The directory where all input and dependent data are stored [Default: %default]')
    parser.add_option('--chipseq', dest='narrowPeak_file', default=None,
            help='Chip-seq file address [Default: %default]')
    parser.add_option('--blockcore', dest='block_core', default=None,
            help='Block the core motif or not [Default: %default]')
    parser.add_option('--recorddir', dest='record_dir', default=None,
            help='The file where to record data infomation [Default: %default]')
    parser.add_option('--TF_name', dest='TF_name', default=None,
            help='The name of the core TF [Default: %default]')
    (options, args) = parser.parse_args()

    pval_bin_size = 0.1
    pos_neg_ratio = 1.0
    chromID_validation = "chr8"
    chromID_test = "chr9"

    
    #####
    # start to prepare and format data
    dp = datasetPrep()
    labelled_list, n_pos, n_neg = dp.prepare_labelled_samples(
                        options.fimo_file, options.narrowPeak_file,
                        pval_bin_size, pos_neg_ratio)
    #sequences = dp.convert(labelled_list, options.len_seq, options.dir_data, options.block_core,
    #                    chromID_validation, chromID_test)
    sequences = dp.convert(labelled_list, options.len_seq, options.dir_data, options.block_core)
    training, validation, test = dp.make_datasets(sequences, chromID_validation, chromID_test)
    dp.save_into_file(training, options.dir_result, tag = "training")
    dp.save_into_file(validation, options.dir_result, tag = "validation")
    dp.save_into_file(test, options.dir_result, tag = "test")

    #record_TF_dir = "%s/%s" %(options.record_dir, options.TF_name)
    #if not os.path.exists(record_TF_dir):
    #    os.makedirs(record_TF_dir)
    dp.save_test_coordinate(labelled_list, options.dir_result, chromID_validation, chromID_test)
    
    # save the number of positive and negative samples
    # training+val+test
    record_file = "%s/data_size.txt" %(options.record_dir)
    with open(record_file, 'a') as ofile:
        line_to_save = "%s\t%d\t%d\n" %(options.TF_name, n_pos, n_neg)
        ofile.write(line_to_save)

if __name__ == "__main__":
    main()
###END OF FILE###########################

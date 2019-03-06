import os
import random
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from Bio.Seq import Seq
from converter import converter_template
from optparse import OptionParser

#################
class datasetPrep(converter_template):
    def __init__(self):
        return

    def prepare_labelled_samples(self, coordinate_file, len_seq):
        '''
            read samples from file
        '''
        labelled_list = []
        for line in open(coordinate_file):
            chromID, start, end, signalVal, pval = (line.strip()).split(";")
            start = int(start)
            end = int(end)
            signalVal = float(signalVal)
            pval = float(pval)
            if (end-start) > len_seq:
                continue
            else:
                smp = (chromID, start, end, (signalVal, pval))
                labelled_list.append(smp)
        return labelled_list

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
            (chromID, start, end, (signalVal, pval)) = smp
            seq_start = start-int((len_seq-(end-start))/2.0)
            seq_end = seq_start + len_seq
            if seq_start >= 0 or seq_end < (chrom_length_dict)[chromID]:
                if block_core == 'c':
                    seq = self._readFromRef(chromID, seq_start, seq_end, reference_genome_dict)
                elif block_core == 'b':
                    seq = self._readFromRef(chromID, seq_start, start, reference_genome_dict)\
                                + "".join(['N' for _ in range(end-start)])\
                                + self._readFromRef(chromID, end, seq_end, reference_genome_dict)
                else:
                    exit('wrong symbol of block_core: %s' %(block_core))
                sequences.append((seq, (signalVal, pval, chromID, start, end, seq_start, seq_end))) 
        return sequences

    def make_datasets(self, samples):
        '''
            divide data into training/validation/test datasets
        '''
        #################
        # separate data into training/validation/test
        samples_test = []
        for smpl in samples:
            (seq, (signalVal, pval, chromID, start, end, seq_start, seq_end)) = smpl
            samples_test.append((seq, (signalVal, pval, chromID, start, end, seq_start, seq_end))) 
        seqs_test = self._prepare_dataset(samples_test)
        return seqs_test

    def save_into_file(self, data, data_dir, tag):
        data_dir = os.path.join(data_dir, tag)
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        output_path = os.path.join(data_dir, "data.txt")

        with open(output_path, 'w') as ofile:
            for dt in data:
                seq, signalVal, pval, chromID, start, end, seq_start, seq_end, pos = dt
                line_to_save = "%s;%d;%f,%s,%d,%s,%d,%d,%d,%d\n"\
                                    %(seq, 1, signalVal, pval, pos,\
                                        chromID, start, end, seq_start, seq_end)
                ofile.write(line_to_save)
        return
    ##################
    ## Private functions
    ##################
    def _prepare_dataset(self, seqs):
        total_seqs = []
        for (seq, (signalVal, pval, chromID, start, end, seq_start, seq_end)) in seqs:
            # forward strand
            seq_arr = (self._string_to_one_hot(seq), signalVal, pval, chromID, start, end, seq_start, seq_end, -1)
            total_seqs.append(seq_arr)
            '''
            seq_arr = [(self._string_to_one_hot(seq), signalVal, pval, chromID, start, end, seq_start, seq_end, -1)]
            for pos in range(len(seq)):
                seq_var = seq[:pos] + "N" + seq[pos+1:]
                seq_arr.append(
                        (self._string_to_one_hot(seq_var), signalVal, pval, chromID, seq_start, seq_end, pos))
            '''
        return total_seqs

def main():
    usage = "usage: %prog [options]"
    parser = OptionParser(usage)
    parser.add_option('--resultdir', dest='dir_result', default=None,
            help='The directory where all temporary and final results are saved [Default: %default]')
    parser.add_option('--datadir', dest='dir_data', default=None,
            help='The directory where all input and dependent data are stored [Default: %default]')
    parser.add_option('--scope', dest='len_seq', default=None, type='int',
            help='The scope of contexts to be examined for each core motif [Default: %default]')
    parser.add_option('--blockcore', dest='block_core', default=None,
            help='Block the core motif or not [Default: %default]')
    (options, args) = parser.parse_args()

    coordinate_file = "%s/vis-weights-total/coordinates_pos.txt" %(options.dir_result)
    dp = datasetPrep()
    labelled_list = dp.prepare_labelled_samples(coordinate_file, options.len_seq)
    sequences = dp.convert(labelled_list, options.len_seq, options.dir_data, options.block_core)
    test = dp.make_datasets(sequences)

    #for sample_index in range(len(test)):
    dp.save_into_file(test, options.dir_result, tag = ("vis-samples/"))

    f_n_samples = "%s/vis-samples/num_of_samples.txt" %(options.dir_result)
    with open(f_n_samples, 'w') as ofile:
        ofile.write("%d" %(len(test)))

if __name__ == "__main__":
    main()
###END OF FILE###########################

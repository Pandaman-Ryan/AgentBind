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
import numpy
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
 
    def generate_datasets(self, dir_data, len_seq, block_core):
        '''
            Generate sequences for training, validation, and test

            Parameters include:
            n_train, n_valid, n_test
            motif_pool
            n_motifs
        '''
        n_train = 50000
        n_valid = 1000
        n_test = 1000
        n_vis = 1000

        GATA_motif = self._read_motif_from_file("%s/simulation-motifs/GATA.motif" %(dir_data))
        TAL1_motif = self._read_motif_from_file("%s/simulation-motifs/TAL1.motif" %(dir_data))
        
        seqs_training = self._make_dataset(n_train, len_seq, GATA_motif, TAL1_motif, block_core)
        seqs_validation = self._make_dataset(n_valid, len_seq, GATA_motif, TAL1_motif, block_core)
        seqs_test = self._make_dataset(n_test, len_seq, GATA_motif, TAL1_motif, block_core)
        seqs_test_vis, seqs_test_vis_prtb = self._make_dataset_pos_only(n_vis, len_seq, GATA_motif, TAL1_motif, block_core)
        return seqs_training, seqs_validation, seqs_test, seqs_test_vis, seqs_test_vis_prtb

    def save_into_file(self, data, data_dir, tag):
        data_dir = os.path.join(data_dir, tag)
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        output_path = os.path.join(data_dir, "data.txt")

        with open(output_path, 'w') as ofile:
            for dt in data:
                seq, label, pos_info = dt
                if tag == "vis-samples":
                    line_to_save = "%s;%d;%s\n" %(seq, label, pos_info)
                elif tag == "vis-samples-prtb":
                    line_to_save = "%s;%d;%s\n" %(seq, label, pos_info)
                else:
                    line_to_save = "%s;%d\n" %(seq, label)
                ofile.write(line_to_save)
        return

    ##################
    ## Private functions
    ##################
    def _read_motif_from_file(self, ifilename):
        header = True
        prob_mat = []
        for line in open(ifilename):
            if header:
                header = False
                continue
            else:
                line = line.strip()
                elems = line.split()
                probs = [float(elems[elem_index]) for elem_index in range(len(elems))\
                                        if elem_index != 0]
                prob_mat.append(probs)
        return prob_mat

    def _make_dataset(self, n_sample, len_seq, GATA_motif, TAL1_motif, block_core):
        seqs = []
        for smpl_index in range(n_sample/2):
            smpl = self._generate_seq(len_seq, GATA_motif, TAL1_motif, block_core, positive=True)
            seqs.append(smpl)
        for smpl_index in range(n_sample/2):
            smpl = self._generate_seq(len_seq, GATA_motif, TAL1_motif, block_core, positive=False)
            seqs.append(smpl)
        return seqs

    def _make_dataset_pos_only(self, n_sample, len_seq, GATA_motif, TAL1_motif, block_core):
        seqs = []
        seqs_prtb = []
        for smpl_index in range(n_sample):
            smpl, smpls_prtb = self._generate_seq(len_seq, GATA_motif, TAL1_motif, block_core, positive=True, prtb=True)
            seqs.append(smpl)
            seqs_prtb.append(smpl)
            for smpl_pb in smpls_prtb:
                seqs_prtb.append(smpl_pb)
        return seqs, seqs_prtb

    def _generate_seq(self, len_seq, GATA_motif, TAL1_motif, block_core, positive, prtb=False):
        # block the core motif or not
        if block_core == 'c':
            GATA_seq = self._sample_seq_from_motif(GATA_motif)
        elif block_core == 'b':
            GATA_seq = 'N'*len(GATA_motif)

        # generate background
        seq = ""
        background_length = len_seq - len(GATA_seq)
        for nt_index in range(background_length):
            nt = random.choice(['A','A','A','T','T','T','C','C','G','G'])
            seq += nt

        # plant the core motif
        len_context = int(background_length/2.0)
        seq = seq[:len_context] + GATA_seq + seq[len_context:]

        if positive == False:
            seq_info = (self._string_to_one_hot(seq), 0, "-1")
        else:
            # generate motifs
            n_TAL1 = self._truncated_poisson()

            motif_positions = []
            positions_taken = [_ for _ in range(len_context, len_context+len(GATA_seq))]
            positions_taken += ([_ for _ in range(len_context-len(TAL1_motif)+1, len_context)]\
                                +[_ for _ in range(len_seq-len(TAL1_motif)+1, len_seq)])
            for motif_index in range(n_TAL1):
                mtf = self._sample_seq_from_motif(TAL1_motif)
                start_position = random.choice([_ for _ in range(len_seq) if _ not in positions_taken])
                motif_positions.append((start_position, start_position+len(mtf)))
                seq = seq[:start_position] + mtf + seq[start_position+len(mtf):]
                positions_taken += [_ for _ in range(start_position, start_position+len(mtf))]

            motif_positions_str = "&".join([("%d-%d" %(start,end)) for (start, end) in motif_positions])
            seq_info = (self._string_to_one_hot(seq), 1, motif_positions_str)

            if prtb == True:
                seqs_prtb = []
                for nt_index in range(len_seq):
                    seqs_pb = self._perturb(seq, nt_index)
                    for sq_pb in seqs_pb:
                        seq_pb_info = (self._string_to_one_hot(sq_pb), 1, motif_positions_str)
                        seqs_prtb.append(seq_pb_info)

        if prtb == False:
            return seq_info
        else:
            return (seq_info, seqs_prtb)

    def _truncated_poisson(self):
        while True:
            sample = numpy.random.poisson(lam=1.0)
            if (sample >= 1) and (sample <= 3):
                break
        return sample

    def _sample_seq_from_motif(self, motif):
        nt_list = ['A', 'C', 'G', 'T']
        seq = ""
        for nt_prob in motif:
            dice = random.random()
            if dice <= nt_prob[0]:
                seq += "A"
            elif dice <= (nt_prob[0] + nt_prob[1]):
                seq += "C"
            elif dice <= (nt_prob[0] + nt_prob[1] + nt_prob[2]):
                seq += "G"
            else:
                seq += "T"
        return seq

    def _perturb(self, seq, index):
        seqs_pb = []
        if seq[index] == 'N':
            sq_pb = seq[:index] + 'N' +seq[index+1:]
            return [sq_pb, sq_pb, sq_pb]
        else:
            for nt in ['A', 'C', 'G', 'T']:
                if nt == seq[index]:
                    continue
                sq_pb = seq[:index] + nt +seq[index+1:]
                seqs_pb.append(sq_pb)
        return seqs_pb

def main():
    usage = "usage: %prog [options]"
    parser = OptionParser(usage)
    parser.add_option('--scope', dest='len_seq', default=None, type='int',
            help='The scope of contexts to be examined for each core motif [Default: %default]')
    parser.add_option('--resultdir', dest='dir_result', default=None,
            help='The directory where all temporary and final results are saved [Default: %default]')
    parser.add_option('--datadir', dest='dir_data', default=None,
            help='The directory where all input and dependent data are stored [Default: %default]')
    parser.add_option('--blockcore', dest='block_core', default=None,
            help='Block the core motif or not [Default: %default]')
    (options, args) = parser.parse_args()

    #####
    # start to prepare and format data
    dp = datasetPrep()
    training, validation, test, test_vis,  test_vis_prtb = dp.generate_datasets(
                                options.dir_data, options.len_seq, options.block_core)
    dp.save_into_file(training, options.dir_result, tag = "training")
    dp.save_into_file(validation, options.dir_result, tag = "validation")
    dp.save_into_file(test, options.dir_result, tag = "test")
    dp.save_into_file(test_vis, options.dir_result, tag = "vis-samples")
    dp.save_into_file(test_vis_prtb, options.dir_result, tag = "vis-samples-prtb")

    f_n_samples = "%s/vis-samples/num_of_samples.txt" %(options.dir_result)
    with open(f_n_samples, 'w') as ofile:
        ofile.write("%d" %(len(test_vis)))

    f_n_samples = "%s/vis-samples-prtb/num_of_samples.txt" %(options.dir_result)
    with open(f_n_samples, 'w') as ofile:
        ofile.write("%d" %(len(test_vis_prtb)))

if __name__ == "__main__":
    main()
###END OF FILE###########################

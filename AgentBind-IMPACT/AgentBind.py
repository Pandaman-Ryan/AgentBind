'''
    -> AgentBind <-
    
    Last Edit: July 2nd, 2018

    Parameters:
'''

from optparse import OptionParser
import os
import subprocess
import time

#######
# f_core_motifs is the file address
# where you

def main():
    ############
    # user-defined options
    ############
    usage = "usage: %prog [options]"
    parser = OptionParser(usage)
    parser.add_option('--scope', dest='len_seq', default=1000, type='int',
            help='The scope of contexts to be examined for each core motif [Default: %default]')
    parser.add_option('--workdir', dest='dir_work',
            default='/storage/pandaman/project/AgentBind-IMPACT/tmp-c0/',
            help='The directory where all temporary and final results are saved [Default: %default]')
    parser.add_option('--datadir', dest='dir_data',
            default='/storage/pandaman/project/AgentBind-IMPACT/data/',
            help='The directory where all input and dependent data are stored [Default: %default]')
    parser.add_option('--resultdir', dest='dir_result',
            default='/storage/pandaman/project/AgentBind-IMPACT/results-c0/',
            help='The directory where all input and dependent data are stored [Default: %default]')
    (options, args) = parser.parse_args()
 
    ########
    # step 1: identify all matched locations across the genome
    #         for each motif
    for path_suffix in ['', 'c']: #'b' stand for block, 'c' stand for core
        summary_path = "%s/%s/" %(options.dir_result, path_suffix)
        if not os.path.exists(summary_path):
            os.makedirs(summary_path)
        
        auc_summary = "%s/auc_summary.txt" %(summary_path)
        if os.path.isfile(auc_summary):
            os.remove(auc_summary)

    if not os.path.exists(options.dir_work):
        os.makedirs(options.dir_work)
    if not os.path.exists(options.dir_result):
        os.makedirs(options.dir_result)

    TF_list = ['Foxp3', 'Gata3', 'Stat3', 'Tbet'] 
    ref_genomes = "%s/genomes/hg19/hg19.fa" %(options.dir_data)
    bgfile = "%s/genomes/hg19/hg19.fna.bfile" %(options.dir_data)

    ######
    for TF_name in TF_list:
        start_time = time.time()

        dir_work_TF = "%s/%s/" %(options.dir_work, TF_name)
        ################
        ## Model training
        ################

        for suffix in ['c']: #'b' stand for block, 'c' stand for core
            #######
            # step 2: prepare data for model training
            dir_seqs = "%s/seqs_one_hot_%s/" %(dir_work_TF, suffix)
            dir_result_with_suffix = "%s/%s/" %(options.dir_result, suffix)
            data_prep_cmd = "python ./data_prep/data_prep.py "\
                    " --scope %d --resultdir %s --datadir %s "\
                    " --TF_name %s " %(options.len_seq,
                                    dir_seqs, options.dir_data, TF_name)
            subprocess.check_call(data_prep_cmd, shell=True)

            vis_data_prep_cmd = "python ./data_prep/data_prep_for_weight_vis_single_sample.py "\
                    " --scope %d --resultdir %s --datadir %s "\
                    " --TF_name %s " %(options.len_seq,
                                    dir_seqs, options.dir_data, TF_name)
            subprocess.check_call(vis_data_prep_cmd, shell=True)

            #######
            # step 3: train
            dir_checkpoint = "%s/encode_model/" %(options.dir_data)
            dir_train_data = "%s/training/" %(dir_seqs)
            size_train_data = sum(1 for line in open("%s/data.txt" %(dir_train_data)))
            dir_valid_data = "%s/validation/" %(dir_seqs)
            size_valid_data = sum(1 for line in open("%s/data.txt" %(dir_valid_data)))

            dir_train_checkpoint = "%s/model_%s/" %(dir_work_TF, suffix)
            train_cmd = "python ./model/train-transfer-learning.py --checkpoint_dir %s "\
                    " --data_dir %s --valid_dir %s " \
                    " --seq_size %d --n_train_samples %d --n_valid_samples %d "\
                    " --train_dir %s " %(dir_checkpoint, dir_train_data, dir_valid_data,
                        options.len_seq, size_train_data, size_valid_data, dir_train_checkpoint)
            subprocess.check_call(train_cmd, shell=True)

            #######
            # step 4: test
            dir_test_data = "%s/test/" %(dir_seqs)
            size_test_data = sum(1 for line in open("%s/data.txt" %(dir_test_data)))
            dir_test_checkpoint = "%s/ckpt-test-phase_%s/" %(dir_work_TF, suffix)
            test_cmd = "python ./model/test.py --TF_name %s --checkpoint_dir %s "\
                    " --data_dir %s --eval_dir %s "\
                    " --n_eval_samples %d --result_dir %s "\
                    " --seq_size %d "%(TF_name, dir_train_checkpoint,
                            dir_test_data, dir_test_checkpoint,
                            size_test_data, dir_result_with_suffix, options.len_seq)
            subprocess.check_call(test_cmd, shell=True)

            ########
            # Step 5: Compute weights
            n_samples = int(open("%s/vis-samples/num_of_samples.txt" %(dir_seqs)).readline())
            f_weight = "%s/vis-samples/weight.txt" %(dir_seqs)
            dir_figure = "%s/%s/weight_figures/" %(dir_result_with_suffix, TF_name)
            compute_weight_cmd = "python ./model/visualize_weights.py "\
                    " --data_dir %s --seq_size %d --n_samples %d "\
                    " --eval_dir %s --checkpoint_dir %s "\
                    " --figure_dir %s --weight_file %s" %(dir_seqs, options.len_seq, n_samples,
                            dir_test_checkpoint, dir_train_checkpoint, dir_figure, f_weight)
            subprocess.check_call(compute_weight_cmd, shell=True)
            
            
        end_time = time.time()
        print ("Analysis of %s used %f seconds" %(TF_name, end_time-start_time))
        start_time = end_time

        #################

if __name__ == "__main__":
    main()

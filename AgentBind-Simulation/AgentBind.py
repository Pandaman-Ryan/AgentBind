'''
    -> AgentBind <-

    With Simulation data
    
    Last Edit: August 2nd, 2018
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
            default='/storage/pandaman/project/AgentBind-Simulation/tmp/',
            help='The directory where all temporary and final results are saved [Default: %default]')
    parser.add_option('--datadir', dest='dir_data',
            default='/storage/pandaman/project/AgentBind-Simulation/data/',
            help='The directory where all input and dependent data are stored [Default: %default]')
    parser.add_option('--resultdir', dest='dir_result',
            default='/storage/pandaman/project/AgentBind-Simulation/results/',
            help='The directory where all input and dependent data are stored [Default: %default]')
    parser.add_option('--method', dest='method',
            default='reg', help='The name of interpretation methods, can be reg, prtb, gb, cam, or gcam')
    (options, args) = parser.parse_args()

    ############
    # analysis of genomic contexts around motifs
    #############
    TF_name = 'simulated-TAL1'
    for core_block_suffix in ['', 'b', 'c']: #'b' stand for block, 'c' stand for core
        summary_path = "%s/%s/%s/" %(options.dir_result, options.method, core_block_suffix)
        if not os.path.exists(summary_path):
            os.makedirs(summary_path)
        auc_summary = "%s/auc_summary.txt" %(summary_path)
        if os.path.isfile(auc_summary):
            os.remove(auc_summary)

    ######
    start_time = time.time()

    dir_work_TF = "%s/%s/" %(options.dir_work, TF_name)
    ################
    ## Model training
    ################

    for suffix in ['c']: #['b', 'c']: #'b' stand for block, 'c' stand for core
        #######
        # step 1: prepare data for model training
        dir_seqs = "%s/seqs_one_hot_%s/" %(dir_work_TF, suffix)
        data_prep_cmd = "python ./data_prep/data_prep.py"\
                " --scope %d --resultdir %s --datadir %s "\
                " --blockcore %s " %(options.len_seq,
                        dir_seqs, options.dir_data, suffix)
        subprocess.check_call(data_prep_cmd, shell=True)

        #######
        # step 2: train
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
        # step 3: test
        dir_test_data = "%s/test/" %(dir_seqs)
        size_test_data = sum(1 for line in open("%s/data.txt" %(dir_test_data)))
        dir_test_checkpoint = "%s/ckpt-test-phase_%s/" %(dir_work_TF, suffix)
        dir_result_with_suffix = "%s/%s/%s/" %(options.dir_result, options.method, suffix)
        test_cmd = "python ./model/test.py --TF_name %s --checkpoint_dir %s "\
                " --data_dir %s --eval_dir %s "\
                " --n_eval_samples %d --result_dir %s "\
                " --seq_size %d "%(TF_name, dir_train_checkpoint,
                        dir_test_data, dir_test_checkpoint,
                        size_test_data, dir_result_with_suffix, options.len_seq)
        subprocess.check_call(test_cmd, shell=True)

        #######
        # Step 4: Compute weights
        dir_figure = "%s/%s/weight_figures/" %(dir_result_with_suffix, TF_name)
        dir_report = "%s/%s/" %(dir_result_with_suffix, TF_name)
        if options.method == "prtb":
            n_samples = int(open("%s/vis-samples-prtb/num_of_samples.txt" %(dir_seqs)).readline())
        else:
            n_samples = int(open("%s/vis-samples/num_of_samples.txt" %(dir_seqs)).readline())
        compute_weight_cmd = "python ./model/visualize_weights_%s.py "\
                " --data_dir %s --seq_size %d --n_samples %d "\
                " --eval_dir %s --checkpoint_dir %s --report_dir %s "\
                " --figure_dir %s " %(options.method, dir_seqs, options.len_seq, n_samples,
                        dir_test_checkpoint, dir_train_checkpoint, dir_report,
                        dir_figure)
        subprocess.check_call(compute_weight_cmd, shell=True)

    end_time = time.time()
    print ("Analysis of %s used %f seconds" %(TF_name, end_time-start_time))

    #################

if __name__ == "__main__":
    main()

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
    parser.add_option('--motif', dest='f_core_motif',
            default="/home/danq-unfixed/danq-unfixed/storage/AgentBind-GM12878-DanQ/data/table_matrix/table_core_motifs.txt",
            help='The file address where you save your motifs of interst [Default: %default]')
    parser.add_option('--scope', dest='len_seq', default=1000, type='int',
            help='The scope of contexts to be examined for each core motif [Default: %default]')
    parser.add_option('--workdir', dest='dir_work',
            default='/home/danq-unfixed/danq-unfixed/storage/AgentBind-GM12878-DanQ/tmp/',
            help='The directory where all temporary and final results are saved [Default: %default]')
    parser.add_option('--datadir', dest='dir_data',
            default='/home/danq-unfixed/danq-unfixed/storage/AgentBind-GM12878-DanQ/data/',
            help='The directory where all input and dependent data are stored [Default: %default]')
    parser.add_option('--resultdir', dest='dir_result',
            default='/home/danq-unfixed/danq-unfixed/storage/AgentBind-GM12878-DanQ/results/',
            help='The directory where all input and dependent data are stored [Default: %default]')
    (options, args) = parser.parse_args()

    ############
    # analysis of genomic contexts around motifs
    #############
    if options.f_core_motif != None and\
            os.path.isfile(options.f_core_motif):
        TF_list = []
        for line in open(options.f_core_motif):
            (TF_name, TF_ID, TF_chipseq) = line.strip().split(";")
            TF_list.append((TF_name, TF_ID, TF_chipseq))
    else:
        exit("Cannot find the given motif file address: %s" %(options.f_core_motif))

    for core_block_suffix in ['', 'b', 'c']: #'b' stand for block, 'c' stand for core
        summary_path = "%s/%s/" %(options.dir_result, core_block_suffix)
        if not os.path.exists(summary_path):
            os.makedirs(summary_path)

        auc_summary = "%s/auc_summary.txt" %(summary_path)
        if os.path.isfile(auc_summary):
            os.remove(auc_summary)

        data_size_path = "%s/data_size.txt" %(summary_path)
        if os.path.isfile(data_size_path):
            os.remove(data_size_path)
    
    ########
    # step 1: identify all matched locations across the genome
    #         for each motif
    
    ref_genomes = "%s/genomes/hg19/hg19.fa" %(options.dir_data)
    bgfile = "%s/genomes/hg19/hg19.fna.bfile" %(options.dir_data)
    dir_jaspar_motif = "%s/Jaspar-motifs/JASPAR2018_CORE_vertebrates_redundant_pfms_meme/"\
                            %(options.dir_data)
    pval_threshold = 1.0e-4
    dir_fimo_out = "%s/fimo_out/" %(options.dir_work)
    if not os.path.exists(dir_fimo_out):
        os.makedirs(dir_fimo_out)

    # comment the following till the final code release
    '''
    for f_fimo in os.listdir(dir_fimo_out):
        os.remove(os.path.join(dir_fimo_out, f_fimo))

    cmd_list = []
    for f_motif in [f_meme for f_meme in os.listdir(dir_jaspar_motif)]:
        motif_id = f_motif[:-5]
        f_motif = os.path.join(dir_jaspar_motif, f_motif)
        f_fimo = os.path.join(dir_fimo_out, motif_id)
        fimo_cmd = "fimo --bgfile %s --max-strand --skip-matched-sequence "\
                " --thresh %f --verbosity 1 "\
                " %s %s > %s" %(bgfile, pval_threshold, f_motif, ref_genomes, f_fimo)
        cmd_list.append(fimo_cmd)

    processes = [subprocess.Popen(cmd, shell=True) for cmd in cmd_list]
    for p in processes:
        p.wait()
    '''

    ######
    for (TF_name, TF_ID, TF_chipseq) in TF_list:
        start_time = time.time()

        dir_work_TF = "%s/%s/" %(options.dir_work, TF_name)
        f_fimo = os.path.join(dir_fimo_out, TF_ID)
        ################
        ## Model training
        ################

        for suffix in ['b', 'c']: #'b' stand for block, 'c' stand for core
            #######
            # step 2: prepare data for model training
            dir_seqs = "%s/seqs_one_hot_%s/" %(dir_work_TF, suffix)
            f_chipseq = "%s/TFBS_ENCODE/%s" %(options.dir_data, TF_chipseq)
            dir_result_with_suffix = "%s/%s/" %(options.dir_result, suffix)
            data_prep_cmd = "python ./data_prep/data_prep.py --fimo_file %s "\
                    " --scope %d --resultdir %s "\
                    " --datadir %s --chipseq %s "\
                    " --blockcore %s --recorddir %s "\
                    " --TF_name %s " %(f_fimo, options.len_seq,
                                    dir_seqs, options.dir_data, f_chipseq, suffix,
                                    dir_result_with_suffix, TF_name)
            subprocess.check_call(data_prep_cmd, shell=True)

            vis_data_prep_cmd = "python ./data_prep/data_prep_for_weight_vis_single_sample.py "\
                    " --resultdir %s --datadir %s "\
                    " --scope %d --blockcore %s " %(dir_seqs, options.dir_data,
                            options.len_seq, suffix)
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

            
            #######
            # Step 5: Compute weights
            dir_figure = "%s/%s/weight_figures/" %(dir_result_with_suffix, TF_name)
            f_weight = "%s/vis-weights-total/weight.txt" %(dir_seqs)
            n_samples = int(open("%s/vis-samples/num_of_samples.txt" %(dir_seqs)).readline())
            compute_weight_cmd = "python ./model/visualize_weights.py "\
                    " --data_dir %s --seq_size %d --n_samples %d "\
                    " --eval_dir %s --checkpoint_dir %s "\
                    " --figure_dir %s --weight_file %s" %(dir_seqs, options.len_seq, n_samples,
                            dir_test_checkpoint, dir_train_checkpoint,
                            dir_figure, f_weight)
            subprocess.check_call(compute_weight_cmd, shell=True)

            #######
            # Step 6: find_relevant_TF.py
            #f_neg_coord = "%s/vis-weights-total/coordinates_neg.txt" %(dir_seqs)
            #analysis_cmd = "python ./data_analysis/data_analysis.py "\
            #        " --datadir %s --fweight %s --negcoord %s "\
            #        " --fimodir %s --resultdir %s "\
            #        " --TFname %s --motif %s " %(options.dir_data, f_weight, f_neg_coord,
            #                dir_fimo_out, dir_result_with_suffix, TF_name, options.f_core_motif)
            #subprocess.check_call(analysis_cmd, shell=True)
            
        end_time = time.time()
        print ("Analysis of %s used %f seconds" %(TF_name, end_time-start_time))
        start_time = end_time

        #################

    # write AUC summary
    #TODO: save AUC, res_stat, data_size into diffrent files
    AUC_dict = {}
    for suffix in ['b', 'c']:
        summary_path = "%s/%s/auc_summary.txt" %(options.dir_result, suffix)
        for line in open(summary_path):
            (TF_name, AUC_value, PR_value) = line.strip().split()
            if TF_name not in AUC_dict:
                AUC_dict[TF_name] = {}
            AUC_dict[TF_name][suffix] = {'ROC': AUC_value, 'PR': PR_value}

    summary_path = "%s/auc_summary.txt" %(options.dir_result)
    with open(summary_path, 'w') as ofile:
        for TF_name in AUC_dict:
            if ('b' not in AUC_dict[TF_name])\
                    or ('c' not in AUC_dict[TF_name]):
                exit('Item lost: %s' %(TF_name))
            else:
                line_to_save = "%s\t%s\t%s\t%s\t%s\n" %(TF_name,\
                            AUC_dict[TF_name]['c']['ROC'], AUC_dict[TF_name]['b']['ROC'],\
                            AUC_dict[TF_name]['c']['PR'], AUC_dict[TF_name]['b']['PR'])
                ofile.write(line_to_save)
    
if __name__ == "__main__":
    main()

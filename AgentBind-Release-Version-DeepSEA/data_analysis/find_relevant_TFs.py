"""
Last Edits: Jun 14th, 2018

This program takes:
    1) Jaspar TF Motifs
    2) weight (predicted) matrix
and calculate TF stats:
    1) average score
    2) p-value vs. negative samples
    3) p-value vs. random shuffled binding sites

Procedures:
    1) Read motifs, feed them into "fimo", acquire
        matched positions accross whole human genome,
        and save the results in seperate files
    2) for each file from step 1, examine the starting and 
        end coordinates and record them if they are inside
        our test samples
    3) calculate stats for each TF
"""


from os import listdir
from os.path import join
import os
import subprocess
import random
from copy import deepcopy
from scipy.special import comb
from math import pow
from math import log10 as log
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from converter import converter_template as converter

'''
    Data path
'''

def execute_fimo():
    # empty the directory where saves fimo results
    if not os.path.exists(fimo_result_dir):
        os.makedirs(fimo_result_dir)
    for fn in listdir(fimo_result_dir):
        os.remove(join(fimo_result_dir, fn))


    print ("execute fimo ...")
    cmd_list = []
    for motif_file in [fn for fn in listdir(jaspar_motif_dir)]:
        motif_id = motif_file[:-5]
        motif_file = join(jaspar_motif_dir, motif_file)
        output_file = join(fimo_result_dir, motif_id)

        fimo_cmd = "fimo --bgfile /storage/pandaman/project/MotifDetective/MEME-suite-db/hg19.fna.bfile "\
                " --max-strand --skip-matched-sequence "\
                " --thresh 1.0e-4 --verbosity 1 "\
                " %s %s > %s "%(motif_file, ref_genomes, output_file)
        cmd_list.append(fimo_cmd)

    processes = [subprocess.Popen(cmd, shell=True) for cmd in cmd_list]

    for p in processes:
        p.wait()
    return

def read_score_map():
    '''read predicted scores
    output:
        score_map: which records the predicted scores
        res_map: which keeps track of areas that overlap binding sites
        total_score: the total scores of the TF of interest
    '''
    score_map = {}
    total_score = 0
    coordinate_list = []
    with open(weight_file) as f:
        while True:
            position_info = f.readline()
            weight_info = f.readline()

            if (position_info != "") and (weight_info != ""):
                chromID, start, end = (position_info.strip()).split(";")
                start = int(start)
                end = int(end)

                weights_arr = [float(wt) for wt in (weight_info.strip()).split(";")]
                score_map[(chromID, start, end)] = weights_arr

                total_score += sum([abs(wt) for wt in weights_arr])

                coordinate_list.append((chromID, start, end))
                len_of_seq = end - start
            else:
                break
    res_map = deepcopy(score_map)
    coordinate_list.sort(key=lambda x:x[1])
    return score_map, res_map, total_score, coordinate_list, len_of_seq

def read_coordinate(len_seq):
    '''Read coordinates of negative samples (samples
    with the core motifs but unbound) for control use
    '''
    coordinates = []
    chrom_length_dict = _read_chromosome_length()
    for line in open(coordinate_file_neg):
        chromID, start, end, signalVal, pval = (line.strip()).split(";")
        start = int(start)
        end = int(end)
        signalVal = float(signalVal)
        pval = float(pval)

        seq_start = start-int((len_seq-(end-start))/2.0)
        seq_end = seq_start + len_seq
        if seq_start >= 0 or seq_end < (chrom_length_dict)[chromID]:
            coordinates.append((chromID, seq_start, seq_end))

    coordinates.sort(key=lambda x:x[1])
    return coordinates


def examine_scores(fimo_file, tf_id, score_map, res_map, 
                coordinates_pos_samples, coordinates_neg_samples):
    if os.stat(fimo_file).st_size == 0:
        return

    # read from fimo
    # motif_id  motif_alt_id    sequence_name   start   stop    strand  score   p-value q-value matched_sequence
    # MA0007.2    AR  chr1    14456   14470   +   10.902  7.91e-05
    fimo_list = []
    skip_header = True
    for line in open(fimo_file):
        if skip_header:
            skip_header = False
            continue

        elems = line.split()
        chromID = elems[2]
        start = int(elems[3])
        end = int(elems[4]) + 1
        pval = float(elems[7])

        if chromID == chromID_test:
            fimo_list.append((chromID, start, end, pval))
    fimo_list.sort(key = lambda x:x[1])

    # compute stats of positive samples
    # including:
    # (1) the scores for the TF of interest
    # (2) the scores for the TF with shuffled binding sites
    #       for the purpose of calculating p-value vs. random
    #       which shows if our scores are random or not
    scores_of_match = []
    scores_of_match_control_list = [[] for exp_index \
                            in range(num_of_control_experiments)]
    num_pos = 0
    init_motif_index = 0
    for key in coordinates_pos_samples:
        (chromID_ref, start_ref, end_ref) = key
        matched_site_length_list = []
        for motif_pos_index in range(init_motif_index, len(fimo_list)):
            (chromID, start, end, pval) = fimo_list[motif_pos_index]
            if (chromID == chromID_ref)\
                    and (start >= start_ref)\
                    and (end <= end_ref):
                avg_weight = 0
                for pos in range(start-start_ref, end-start_ref):
                    avg_weight += score_map[key][pos]
                    res_map[key][pos] = 0
                avg_weight /= float(end-start)
                scores_of_match.append(avg_weight)
                num_pos += 1
                matched_site_length_list.append(end-start)
            elif (chromID != chromID_ref):
                exit("There is something wrong in the data pipline:\n"
                        "chromosome IDs do not match: "
                        "%s; %s" %(chromID_ref, chromID))
            elif (start < start_ref):
                init_motif_index = motif_pos_index + 1
            elif (start >= end_ref):
                break
            elif (start < end_ref) and (end > end_ref):
                continue
            else:
                print (key)
                print (fimo_list[motif_pos_index])
                exit("Wrong coordinates:\n"
                        "start-end: %d;%d\n"
                        "start_ref-end_ref: %d,%d" %(start, end, start_ref, end_ref))

        matched_site_total_length = sum(matched_site_length_list)
        for exp_index in range(num_of_control_experiments):
            random.shuffle(matched_site_length_list)
            #insert_positions = random.sample([_ for _ in range(end_ref-start_ref-matched_site_total_length)],
            #                                len(matched_site_length_list))
            #offset = 0
            #for ins_index in range(len(insert_positions)):
            #    ins_pos = insert_positions[ins_index]
            #    avg_weight = 0
            #    for pos in range(ins_pos+offset, ins_pos+offset+matched_site_length_list[ins_index]):
            #        avg_weight += score_map[key][pos]
            #    avg_weight /= float(matched_site_length_list[ins_index])
            #    scores_of_match_control_list[exp_index].append(avg_weight)
            #    offset += matched_site_length_list[ins_index]
            for site_index in range(len(matched_site_length_list)):
                site_start_pos = random.choice([_ for _ in 
                                range(end_ref - start_ref - matched_site_length_list[site_index])])
                avg_weight = 0
                for pos in range(site_start_pos, site_start_pos+matched_site_length_list[site_index]):
                    avg_weight += score_map[key][pos]
                avg_weight /= float(matched_site_length_list[site_index])
                scores_of_match_control_list[exp_index].append(avg_weight)

    pval_vs_shuffled = _calculate_pval_vs_shuffled(scores_of_match,
                                        scores_of_match_control_list)
    # control samples
    # for the purpose of calculating p-value vs. negative samples
    # to show that if the TF of interest is enriched in the 
    # positive samples
    num_neg = 0
    init_motif_index = 0
    for key in coordinates_neg_samples:
        (chromID_ref, start_ref, end_ref) = key
        for motif_pos_index in range(init_motif_index, len(fimo_list)):
            (chromID, start, end, pval) = fimo_list[motif_pos_index]
            if (chromID == chromID_ref)\
                    and (start >= start_ref) \
                    and (end <= end_ref):
                num_neg += 1
            elif (chromID != chromID_ref):
                exit("There is something wrong in the data pipline:\n"
                        "chromosome IDs do not match: "
                        "%s; %s" %(chromID_ref, chromID))
            elif (start < start_ref):
                init_motif_index = motif_pos_index + 1
            elif (start >= end_ref):
                break
            elif (start < end_ref) and (end > end_ref):
                continue
            else:
                exit("Wrong coordinates:\n"
                        "start-end: %d;%d\n"
                        "start_ref-end_ref: %d,%d" %(start, end, start_ref, end_ref))

    #print (sum(scores_of_match)/len(scores_of_match), pval_vs_shuffled, num_pos, num_neg)
    #############################
    # save results into file
    with open(score_file, 'a') as scr_f:
        if len(scores_of_match) == 0:
            print ("%s: no match" %(tf_id))
            return
        else:
            line_to_write = "%s;%d;%d;%f\n" %(tf_id.strip(), num_pos, num_neg,\
                                            pval_vs_shuffled)
            scr_f.write(line_to_write)

            line_to_write = "%s" %(scores_of_match[0])
            for scr in scores_of_match[1:]:
                line_to_write += (";%s" %scr)
            line_to_write += "\n"
            scr_f.write(line_to_write)
            return
    return

def record():
    record_list = []
    with open(score_file) as scr_f:
        while True:
            tf_id_and_info = (scr_f.readline()).strip()
            score_info = (scr_f.readline()).strip()

            if (tf_id_and_info != "") and (score_info != ""):
                # tf_id
                tf_id, num_pos, num_neg, pval_vs_shuffled =\
                                            tf_id_and_info.split(";")
                num_pos = int(num_pos)
                num_neg = int(num_neg)
                tf_pval = _compute_pval(num_pos, num_neg)
                pval_vs_shuffled = float(pval_vs_shuffled)

                # score_info
                score_arr = [abs(float(scr)) for scr in score_info.split(";")]
                avg_score = sum(score_arr) / float(len(score_arr))
                record_list.append((avg_score, tf_id, num_pos, num_neg,
                                    tf_pval, pval_vs_shuffled))
            else:
                break
        record_list.sort(key=lambda x:x[0], reverse=True)

    TF_profile = _read_TF_profile()
    with open(result_file, 'w') as ofile:
        for rcd in record_list:
            (avg_score, tf_id, num_pos, num_neg, tf_pval, pval_vs_shuffled) = rcd
            #if tf_pval < 0.05 and (tf_id in TF_profile):
            if (tf_id in TF_profile):
                (ID, TF_name, class_family) = TF_profile[tf_id]
                line_to_save = "%s; %s; %f; %d; %d; %s; %f; %s\n"\
                                %(tf_id, TF_name, avg_score, num_pos, num_neg, tf_pval, \
                                    pval_vs_shuffled, class_family)
                ofile.write(line_to_save)
    return

#######
# util functions
def _compute_pval(num_pos, num_neg):
    sum_num = num_pos+num_neg
    try:
        pval = 0
        for current_value in range(min(num_pos,num_neg)+1):
            pval += comb(sum_num,current_value)/pow(2,sum_num)
    except OverflowError:
        pval_inv = ((min(num_pos, num_neg)+1)*(2**sum_num)) / comb(sum_num,current_value, exact=True)
        try:
            pval = 1.0/pval_inv
        except OverflowError:
            pval = 0
    return pval

def _read_TF_profile():
    TF_profile = {}

    skip_header = True
    for line in open(TF_profile_file):
        if skip_header == True:
            skip_header = False
            continue
        
        ID_TF, class_family = ((line.strip()).split("Homo sapiens"))
        ID, TF_name = (ID_TF.strip()).split()
        class_family = class_family.strip()
        prfl = (ID, TF_name, class_family)
        if ID not in TF_profile:
            TF_profile[ID] = prfl
        else:
            print ("Duplicated ID: %s" %(ID))

    return TF_profile

def _read_chromosome_length():
    file_addr = genome_limit_path
    chrom_length_dict = {}
    for line in open(file_addr):
        elems = line.split()
        chromID = elems[0]
        length = int(elems[1])
        chrom_length_dict[chromID] = length
    return chrom_length_dict

def _calculate_pval_vs_shuffled(sample, control_groups):
    #sample_value = sum(sample)
    #control_value_list = [sum(smpl) for smpl in control_groups]
    sample_value = sum([abs(_) for _ in sample])
    control_value_list = [sum([abs(_) for _ in smpl]) for smpl in control_groups]
    control_value_list.sort()
    num_control = len(control_value_list)

    best_index = num_control
    for smpl_index in range(num_control):
        if sample_value < control_value_list[smpl_index]:
            best_index = smpl_index
            break

    pval = float(num_control-best_index)/float(num_control)
    return pval

def compute_res_stats(res_map, total_score, score_map):
    res_score = sum([sum([abs(scr) for scr in res_map[key]]) for key in res_map])
    print ("The binding sites take account for %f score" %(1-float(res_score)/float(total_score)))

    # visualize res locations
    res_acc_list = None
    for key in res_map:
        if res_acc_list == None:
            res_acc_list = deepcopy(res_map[key])
        else:
            for pos in range(len(res_map[key])):
                res_acc_list[pos] += res_map[key][pos]

    plt.figure(figsize=(11,11))
    plt.scatter([pos for pos in range(len(res_acc_list))], res_acc_list, marker=".")
    plt.title("signal vs positions")
    plt.xlabel('positions')
    plt.ylabel('weights')
    plt.savefig('./res_map.png')
    plt.close()

    # visualize res map indivisually
    # figure 1: plot ROC
    sample_index = 0
    for key in res_map:
        (chromID, start, end) = key
        plt.figure(figsize=(11,11))
        plt.scatter([pos for pos in range(len(score_map[key]))], score_map[key], marker=".", c='b')
        plt.scatter([pos for pos in range(len(res_map[key]))], res_map[key], marker=".", c='r')
        plt.title("signal vs positions")
        plt.xlabel('positions-%s:%d-%d' %(chromID, start, end))
        plt.ylabel('weights')
        plt.savefig('./res_map/vis-sample-%d.png' %(sample_index))
        plt.close()

        sample_index += 1


    seq_converter = converter()
    ref_dict = seq_converter._load_ref_genome(ref_genomes_dir)
    # record residue segments
    with open(res_seqs_file,'w') as ofile:
        for key in res_map:
            (chromID, start, end) = key
            positions_in_seq = []
            scores_in_seq = []
            for pos in range(len(res_map[key])):
                if res_map[key][pos] != 0:
                    positions_in_seq.append(pos)
                    scores_in_seq.append(res_map[key][pos])
                else:
                    if len(positions_in_seq) != 0:
                        seq_start = start + positions_in_seq[0]
                        seq_end = start + positions_in_seq[-1] + 1

                        res_seq = seq_converter._readFromRef(
                                    chromID, seq_start, seq_end, ref_dict)
                        scores_str = ";".join([str(scr) for scr in scores_in_seq])
                        ofile.write("%s;%s;%d;%d;%s\n" \
                                    %(res_seq, chromID, seq_start, seq_end, scores_str))

                        #res_seq_rev = seq_converter._reverse_complement(res_seq)
                        #ofile.write("%s\n" %(res_seq_rev))

                        positions_in_seq = []
                        scores_in_seq = []

def main():
    ####
    # execute fimo and identify positions of each motif
    # accross the whole human genome
    # this function only need to be run once, and its results
    # can be used for any TF analysis.
    # Thus, this function is 
    ####
    #execute_fimo()
    

    #####
    # Calculate weight scores for each TF
    #####
    # load data
    
    score_map, res_map, total_score,\
            coordinates_pos_samples, len_of_seq = read_score_map()
    coordinates_neg_samples = read_coordinate(len_of_seq)
    if os.path.isfile(score_file):
        os.remove(score_file)

    # calculate scores for each TF
    print ("calculate scores ...")
    count = 0
    for tf_id in [fn for fn in listdir(fimo_result_dir)]:
        print("Now run experiment %d and analyze %s:" %(count, tf_id))
        fimo_file = join(fimo_result_dir, tf_id)
        examine_scores(fimo_file, tf_id, score_map, res_map,
                        coordinates_pos_samples, coordinates_neg_samples)
        count += 1
    # post analysis
    compute_res_stats(res_map, total_score, score_map)
    
    #####
    # save records
    ######
    print("record the results ...")
    record()

if __name__ == "__main__":
    main()

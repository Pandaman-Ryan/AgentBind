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
    4) analyze 6-mers in the regions without any TF annotations
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
from optparse import OptionParser

from converter import converter_template as converter

def read_score_map(weight_file):
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
                chromID, motif_start, motif_end, seq_start, seq_end =\
                                (position_info.strip()).split(";")
                motif_start = int(motif_start)
                motif_end = int(motif_end)
                seq_start = int(seq_start)
                seq_end = int(seq_end)

                weights_arr = [float(wt) for wt in (weight_info.strip()).split(";")]
                score_map[(chromID, motif_start, motif_end, seq_start, seq_end)] = weights_arr
            else:
                break
    return score_map


def compute_kmer_scores(score_map, kmer_length, ref_genomes_dir):
    # load reference genome
    seq_converter = converter()
    ref_dict = seq_converter._load_ref_genome(ref_genomes_dir)

    # analyze each sample
    kmer_dict = {}
    for key in score_map:
        (chromID, motif_start, motif_end, seq_start, seq_end) = key
        sample_seq = seq_converter._readFromRef(chromID, seq_start, seq_end, ref_dict)

        
        for kmer_start in range(0, seq_end - seq_start - kmer_length + 1):
            kmer_end = kmer_start + kmer_length
            if (kmer_end < (motif_start-seq_start)) or (kmer_start > (motif_end-seq_start)):
                kmer_seq = sample_seq[kmer_start: kmer_end]
                if "N" not in kmer_seq:
                    kmer_score = sum(score_map[key][kmer_start: kmer_end]) / float(kmer_end - kmer_start)
                    if kmer_seq in kmer_dict:
                        kmer_dict[kmer_seq].append(kmer_score)
                    else:
                        kmer_dict[kmer_seq] = [kmer_score]
                else:
                    # remove all kmers containing "N"
                    continue
            else:
                # skip the core motif
                continue

    return kmer_dict

def rank_kmer_based_on_mean(kmer_dict):
    kmer_list = []
    for kmer_seq in kmer_dict:
        weights = kmer_dict[kmer_seq]
        avg_weight = sum(weights)/float(len(weights))

        if len(weights) >= 10:
            kmer_list.append((kmer_seq, avg_weight, len(weights)))
    kmer_list.sort(key = lambda x:abs(x[1]), reverse =True)
    return kmer_list

def record(kmer_list, annotation_dict, result_dir):
    f_kmer_record = "%s/kmer_ranking.txt" %(result_dir)
    with open(f_kmer_record, 'w') as ofile:
        for kmer_info in kmer_list:
            kmer_seq = kmer_info[0]
            weight = kmer_info[1]
            num_occ = kmer_info[2]
            line_to_save = "%s\t%f\t%d\t%s\n" %(kmer_seq, weight, num_occ, annotation_dict[kmer_seq])
            ofile.write(line_to_save)
    return

def annotate_with_motif_id(f_kmer2motifs):
    annotation_dict = {}
    for line in open(f_kmer2motifs):
        kmer_seq = line[:8]
        kmer_info = (line[8:]).strip()
        annotation_dict[kmer_seq] = kmer_info
    return annotation_dict

def _write_into_meme(kmer_seq, kmer_index, ofilename):
    with open(ofilename, 'w') as ofile:
        # header
        headers = []
        headers.append("MEME version 4\n")
        headers.append("\n")
        headers.append("ALPHABET= ACGT\n")
        headers.append("\n")
        headers.append("strands: + -\n")
        headers.append("\n")
        headers.append("Background letter frequencies (from uniform background):\n")
        headers.append("A 0.25000 C 0.25000 G 0.25000 T 0.25000\n")
        headers.append("\n")
        headers.append("MOTIF query.%d %s\n" %(kmer_index, kmer_seq))
        headers.append("letter-probability matrix: alength= 4 w= %d nsites= 20 E= 0\n" %(len(kmer_seq)))

        for hdr in headers:
            ofile.write(hdr)

        # data
        nuc2prob_dict = {
            'A': "1.000000 0.000000 0.000000 0.000000\n",
            'C': "0.000000 1.000000 0.000000 0.000000\n",
            'G': "0.000000 0.000000 1.000000 0.000000\n",
            'T': "0.000000 0.000000 0.000000 1.000000\n",
            'N': "0.250000 0.250000 0.250000 0.250000\n"}
        for nuc in kmer_seq:
            ofile.write(nuc2prob_dict[nuc])

        ofile.write("\n")
    return

def _search_matched_motif(f_query, f_jaspar_database, f_matches):
    cmd = "tomtom %s %s --text -incomplete-scores -verbosity 1 > %s" %(f_query, f_jaspar_database, f_matches)
    subprocess.check_call(cmd, shell=True)
    return

def _read_matched_motifs(f_tomtom):
    motif_list = []
    for line in open(f_tomtom):
        line = line.strip()
        if len(line) > 0 and line[0].isdigit():
            elems = line.split()
            motif_id = elems[1]
            motif_list.append(motif_id)
    return motif_list

def _annotate_with_motifs(annotion_with_id_dict, TF_profile):
    annotation_dict = {}
    for kmer_seq in annotion_with_id_dict:
        motif_id_list = annotion_with_id_dict[kmer_seq]
        annotation_arr = []
        for motif_id in motif_id_list:
            if motif_id in TF_profile:
                motif_name = TF_profile[motif_id]
            else:
                motif_name = "_"
            annotation_arr.append("%s/%s" %(motif_id, motif_name))
        annotation_str = ";".join(annotation_arr)
        annotation_dict[kmer_seq] = annotation_str
    return annotation_dict


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

def _read_TF_profile(TF_profile_file):
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

def _read_chromosome_length(genome_limit_path):
    chrom_length_dict = {}
    for line in open(genome_limit_path):
        elems = line.split()
        chromID = elems[0]
        length = int(elems[1])
        chrom_length_dict[chromID] = length
    return chrom_length_dict

def _calculate_pval_vs_shuffled(sample, control_groups):
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


def main():
    usage = "usage: %prog [options]"
    parser = OptionParser(usage)
    parser.add_option('--datadir', dest='dir_data',
            default='/storage/pandaman/project/AgentBind-Release-Version/data/',
            help='The directory where all input and dependent data are stored [Default: %default]')
    parser.add_option('--storage', dest='dir_storage', default=None,
            help='The directory where to store all the tmp files [Default: %default]')
    parser.add_option('--fweight', dest='weight_file', default=None,
            help='The file where stores weights of positions of test samples [Default: %default]')
    parser.add_option('--negcoord', dest='coordinate_file_neg', default=None,
            help='The file where stores coordinates of negative samples [Default: %default]')
    parser.add_option('--fimodir', dest='fimo_result_dir', default=None,
            help='The directory which contains fimo outputs of all motifs [Default: %default]')
    parser.add_option('--resultdir', dest='result_dir', default=None,
            help='The file where to save the residue sequences [Default: %default]')
    parser.add_option('--TFname', dest='TF_name', default=None,
            help='The name of the core TF [Default: %default]')
    (options, args) = parser.parse_args()

    #chromID_test = "chr9"
    kmer_length = 8
    genome_limit_path = "%s/genomes/hg19/hg19.fa.fai" %(options.dir_data)
    ref_genomes_dir = "%s/genomes/hg19/" %(options.dir_data)
    #f_profile = "%s/Jaspar-motifs/TF_profile.txt"%(options.dir_data)
    f_kmer2motifs = "%s/kmer2motifs/kmer2motifs.txt" %(options.dir_data)
    #f_jaspar_database = "%s/Jaspar-motifs/JASPAR2018_CORE_vertebrates_redundant_pfms_meme.txt" %(options.dir_data)
    result_dir_for_TF = "%s/%s/" %(options.result_dir, options.TF_name)
    dir_storage = options.dir_storage

    if not os.path.exists(dir_storage):
        os.makedirs(dir_storage)

    #####
    # Calculate weight scores for each TF
    #####
    # load data
    score_map = read_score_map(options.weight_file)
    kmer_dict = compute_kmer_scores(score_map, kmer_length, ref_genomes_dir)
    kmer_list = rank_kmer_based_on_mean(kmer_dict)
    annotation_dict = annotate_with_motif_id(f_kmer2motifs)
    record(kmer_list, annotation_dict, result_dir_for_TF)


if __name__ == "__main__":
    main()

##### END OF FILE #####################################








































def original_main():
    ###Abandoned######################
    num_of_control_experiments = 1000
    res_stat_file = "%s/residue_stats.txt" %(options.result_dir)
    score_file = "%s/scores_TFs_in_contexts.txt" %(result_dir_for_TF)
    if os.path.isfile(score_file):
        os.remove(score_file)

    coordinates_neg_samples = read_coordinate(
            options.coordinate_file_neg, len_of_seq, genome_limit_path)


    # calculate scores for each TF
    #print ("calculate scores ...")
    count = 0
    for tf_id in [fn for fn in listdir(options.fimo_result_dir)]:
        print("Now run experiment %d and analyze %s:" %(count, tf_id))
        fimo_file = join(options.fimo_result_dir, tf_id)
        examine_scores(fimo_file, tf_id, score_map, res_map, chromID_test,
                        coordinates_pos_samples, coordinates_neg_samples,
                        num_of_control_experiments, score_file)
        count += 1
    # post analysis
    compute_res_stats(res_map, total_score, score_map, options.TF_name, 
                result_dir_for_TF, res_stat_file, ref_genomes_dir)
    
    #####
    # save records
    ######
    print("record the results ...")
    record(score_file, result_dir_for_TF, f_profile)



def examine_scores(fimo_file, tf_id, score_map, res_map, chromID_test, 
                coordinates_pos_samples, coordinates_neg_samples,
                num_of_control_experiments, score_file):
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

    #############################
    # save results
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

def record(score_file, result_dir, f_profile):
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

    TF_profile = _read_TF_profile(f_profile)
    result_file = "%s/TF_ranking_list.txt" %(result_dir)
    with open(result_file, 'w') as ofile:
        for rcd in record_list:
            (avg_score, tf_id, num_pos, num_neg, tf_pval, pval_vs_shuffled) = rcd
            if (tf_id in TF_profile):
                (ID, TF_name, class_family) = TF_profile[tf_id]
                line_to_save = "%s; %s; %f; %d; %d; %s; %f; %s\n"\
                                %(tf_id, TF_name, avg_score, num_pos, num_neg, tf_pval, \
                                    pval_vs_shuffled, class_family)
                ofile.write(line_to_save)
    return

def read_coordinate(coordinate_file_neg, len_seq, genome_limit_path):
    '''Read coordinates of negative samples (samples
    with the core motifs but unbound) for control use
    '''
    coordinates = []
    chrom_length_dict = _read_chromosome_length(genome_limit_path)
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

def compute_res_stats(res_map, total_score, score_map, TF_name,
                        result_dir, res_stat_file, ref_genomes_dir):
    res_score = sum([sum([abs(scr) for scr in res_map[key]]) for key in res_map])
    binding_site_ratio = 1-float(res_score)/float(total_score)
    with open(res_stat_file, 'a') as ofile:
        line_to_save = "%s\t%f\n" %(TF_name, binding_site_ratio)
        ofile.write(line_to_save)
 
    # visualize res locations
    res_acc_list = None
    for key in res_map:
        if res_acc_list == None:
            res_acc_list = deepcopy(res_map[key])
        else:
            for pos in range(len(res_map[key])):
                res_acc_list[pos] += res_map[key][pos]

    f_res_map = "%s/res_map.png" %(result_dir)
    plt.figure(figsize=(11,11))
    plt.scatter([pos for pos in range(len(res_acc_list))], res_acc_list, marker=".")
    plt.title("signal vs positions")
    plt.xlabel('positions')
    plt.ylabel('weights')
    plt.savefig(f_res_map)
    plt.close()

    # visualize res map indivisually
    '''
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
    '''

    # record residue segments
    res_seqs_file = "%s/res_seqs.txt" %(result_dir)
    seq_converter = converter()
    ref_dict = seq_converter._load_ref_genome(ref_genomes_dir)
    with open(res_seqs_file, 'w') as ofile:
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

                        positions_in_seq = []
                        scores_in_seq = []

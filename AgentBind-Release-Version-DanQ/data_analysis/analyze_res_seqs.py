from Bio import pairwise2
from Bio.pairwise2 import format_alignment
from Bio.Seq import Seq
#from sklearn.cluster import AgglomerativeClustering as clustering
#from sklearn.cluster import DBSCAN as clustering
import random
import os
from optparse import OptionParser


def read_TF_profile(TF_profile_file):
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

def read_motifs(jaspar_motif_dir, TF_profile):
    motif_dict = {}
    for motif_file in [fn for fn in os.listdir(jaspar_motif_dir)]:
        motif_id = motif_file[:-5]
        motif_file = os.path.join(jaspar_motif_dir, motif_file)

        start_record = False
        no_profile = False
        for line in open(motif_file):
            line = line.strip()
            if line.startswith("MOTIF"):
                _, TF_ID, TF_name = line.split()
                if (TF_ID, TF_name) in motif_dict:
                    exit("Duplicate labels: %s, %s" %(TF_ID, TF_name))
                else:
                    if TF_ID in TF_profile:
                        motif_dict[(TF_ID, TF_name)] = []
                    else:
                        no_profile = True
                        break
            elif line.startswith("letter-probability matrix"):
                start_record = True
            elif (line == "") or (line.startswith('URL')):
                start_record = False
            elif (start_record == True) and (line != "")\
                    and (line.startswith('URL') == False):
                pa, pc, pg, pt = [max(float(p), 10e-6) for p in line.split()]
                prob = {'A':pa, 'C':pc, 'G':pg, 'T':pt}
                motif_dict[(TF_ID, TF_name)].append(prob)

    return motif_dict

def search_for_resemble_motif(kmer, motif_dict, TF_profile):
    best_score = -1
    best_score_ID = None
    kmer_rc = str((Seq(kmer)).reverse_complement())

    for (TF_ID, TF_name) in motif_dict:
        prob_mat = motif_dict[(TF_ID, TF_name)]
        if len(kmer) > len(prob_mat):
            continue
        else:
            for pos_index in range(len(prob_mat)-len(kmer)+1):
                prob_mat_seg = prob_mat[pos_index: pos_index+len(kmer)]
                current_score = 1
                current_score_rc = 1
                for mat_seg_pos_index in range(len(prob_mat_seg)):
                    current_score *= prob_mat_seg[mat_seg_pos_index][kmer[mat_seg_pos_index]]
                    current_score_rc *= prob_mat_seg[mat_seg_pos_index][kmer_rc[mat_seg_pos_index]]
                if (current_score > best_score)\
                        or (current_score_rc > best_score):
                    best_score = max(current_score, current_score_rc)
                    best_score_ID = (TF_ID, TF_name)

    best_score_profile_ID, best_score_profile_name, best_score_TF_classfamily =\
                    TF_profile[best_score_ID[0]]
    
    return (best_score_ID[0], best_score_ID[1], best_score_TF_classfamily)

def compute_kmer_stat(res_seqs_file, k, motif_dict, TF_profile, res_seqs_profile_file):
    count = 0
    seqs_list = []
    kmer_dict = {}
    kmer_pos_dict = {}
    for line in open(res_seqs_file):
        line = line.strip()
        elems = (line.strip()).split(";")
        seq = elems[0]
        chromID = elems[1]
        seq_start = int(elems[2])
        seq_end = int(elems[3]) 
        scores = [float(em) for em in elems[4:]]
        if len(seq) >= k:
            for nuc_index in range(len(seq)-k+1):
                kmer = seq[nuc_index:nuc_index+k]
                kmer_scores = scores[nuc_index:nuc_index+k]
                score_avg = sum(kmer_scores)/float(k)

                kmer_start = seq_start + nuc_index
                kmer_end = kmer_start + k

                if kmer in kmer_dict:
                    kmer_dict[kmer].append(score_avg)
                    kmer_pos_dict[kmer].append((chromID, kmer_start, kmer_end))
                else:
                    kmer_dict[kmer] = [score_avg]
                    kmer_pos_dict[kmer] = \
                                [(chromID, kmer_start, kmer_end)]
                count += 1

    ranking_list = []
    for kmer in kmer_dict:
        if 'N' in kmer:
            continue

        n_occ = len(kmer_dict[kmer])
        kmer_score = sum(kmer_dict[kmer])/float(n_occ)
        positions = kmer_pos_dict[kmer]

        TF_ID, TF_name, TF_family = search_for_resemble_motif(kmer, motif_dict, TF_profile)

        if n_occ > 10:
            ranking_list.append((kmer, kmer_score, n_occ, TF_ID, TF_name, TF_family, positions))
    ranking_list.sort(key=lambda x:x[1], reverse=True)

    with open(res_seqs_profile_file, 'w') as ofile:
        for (seq, score, n_occ, TF_ID, TF_name, TF_family, positions) in ranking_list:
            '''
            pos_str_list = []
            for (chromID, start, end) in positions:
                pos_str_list.append(
                        "%s:%d-%d" %(chromID, start, end))
            pos_str = ";".join(pos_str_list)

            print ("%s\t%f\t%d\t%s" %(seq,score, n_occ, pos_str))
            '''
            line_to_print = "%s\t%f\t%d\t%s\t%s\n" %(seq, score, n_occ, TF_name, TF_family)
            ofile.write(line_to_print)

def main():
    usage = "usage: %prog [options]"
    parser = OptionParser(usage)
    parser.add_option('--datadir', dest='dir_data',
        default="/storage/pandaman/project/AgentBind-Release-Version/data/",
        help='The file directory where stores all input data [Default: %default]')
    parser.add_option('--resultdir', dest='dir_result',
        default='/storage/pandaman/project/AgentBind-Release-Version/results/b/HNF4A+HepG2/',
        help='The file directory where stores all output data [Default: %default]')
    (options, args) = parser.parse_args()

    jaspar_motif_dir = "%s/Jaspar-motifs/JASPAR2018_CORE_vertebrates_redundant_pfms_meme/" %(options.dir_data)
    TF_profile_file = "%s/Jaspar-motifs/TF_profile.txt"%(options.dir_data)
    res_seqs_file = "%s/res_seqs.txt" %(options.dir_result)
    res_seqs_profile_file = "%s/res_kmer_profile.txt" %(options.dir_result)
    k=6

    TF_profile = read_TF_profile(TF_profile_file)
    motif_dict = read_motifs(jaspar_motif_dir, TF_profile)
    compute_kmer_stat(res_seqs_file, k, motif_dict, TF_profile, res_seqs_profile_file)

    return

if __name__ == "__main__":
    main()

###############################



'''
def _compute_hamming(seq1, seq2):
    (kmer1, score1, avg_scr1) = seq1
    (kmer2, score2, avg_scr2) = seq2

    hamming_dist = 0
    for nuc_index in range(k):
        if kmer1[nuc_index] != kmer2[nuc_index]:
            hamming_dist += 1
            #hamming_dist += (abs(score1[nuc_index])
            #                    + abs(score2[nuc_index]))

    hamming_dist_rc = 0
    kmer2_rc = str((Seq(kmer2)).reverse_complement())
    score2_rc = score2[::-1]
    for nuc_index in range(k):
        if kmer1[nuc_index] != kmer2_rc[nuc_index]:
            hamming_dist_rc += 1
            #hamming_dist_rc += (abs(score1[nuc_index])
            #                    + abs(score2_rc[nuc_index]))
    return min(hamming_dist, hamming_dist_rc)


dist_matrix = [[0 for sq in seqs_list] for sq in seqs_list]
for seq_1_index in range(len(seqs_list)):
    seq_1 = seqs_list[seq_1_index]
    for seq_2_index in range(seq_1_index, len(seqs_list)):
        seq_2 = seqs_list[seq_2_index]
        #alignment_score = pairwise2.align.globalcc(
        #                    seq_1, seq_2, match_function,\
        #                    gap_function_seq_1, gap_function_seq_2,\
        #                    gap_char=['-'], score_only=True)
        alignment_score = _compute_hamming(seq_1, seq_2)
        dist_matrix[seq_1_index][seq_2_index] = alignment_score
        dist_matrix[seq_2_index][seq_1_index] = alignment_score

print ("start clustering")
#cluster_labels = clustering(n_clusters=7, affinity='precomputed', linkage='average').\
cluster_labels = clustering(eps=2, metric='precomputed', n_jobs=-1).fit_predict(dist_matrix)
for sample_index in range(len(seqs_list)):
    label = cluster_labels[sample_index]
    seqs_list[sample_index].append(label)

seqs_list.sort(key=lambda x:x[2], reverse=True)
for (kmer, scores, score_avg, label) in seqs_list:
    print ("%s\t%f\t%d" %(kmer, score_avg, label))
exit("finished")



def match_function(e1, e2):
    e1_nuc = e1[0]
    e1_score = float(e1[1:])
    e2_nuc = e2[0]
    e2_score = float(e2[1:])

    if e1_nuc == e2_nuc:
        return (e1_score+e2_score)
    else:
        return -(e1_score+e2_score)

def gap_function_seq_1(x, y):
    if (x == 0 or x == len(seq_1))\
        and (len(seq_1) < len(seq_2)):
        return 0

    if y == 0:
        return 0
    else:
        return -10e9

def gap_function_seq_2(x, y):
    if (x == 0 or x == len(seq_2))\
        and (len(seq_1) > len(seq_2)):
        return 0

    if y == 0:
        return 0
    else:
        return -10e9


dist_matrix = [[0 for sq in seqs_list] for sq in seqs_list]
for seq_1_index in range(len(seqs_list)):
    seq_1 = seqs_list[seq_1_index]
    for seq_2_index in range(seq_1_index, len(seqs_list)):
        seq_2 = seqs_list[seq_2_index]
        alignment_score = pairwise2.align.globalcc(
                            seq_1, seq_2, match_function,\
                            gap_function_seq_1, gap_function_seq_2,\
                            gap_char=['-'], score_only=True)
        dist_matrix[seq_1_index][seq_2_index] = (-alignment_score)
        dist_matrix[seq_2_index][seq_1_index] = (-alignment_score)

with open(dist_mat_file, 'w') as ofile:
    for dist_entry in dist_matrix:
        line_to_write = "%f" %(dist_entry[0])
        for dist in dist_entry[1:]:
            line_to_write += ";%f" %(dist)
        line_to_write += "\n"


#print (clustering(n_clusters=2, affinity='precomputed', linkage='average').\
#                fit_predict(alignment_score))
'''



###############
#seq_1 = ['A0.1','T0.1','T0.3','C0.4','C0.7','G0.3','T0.1']
#seq_2 = ['T0.4','C0.3','G0.3']


#print pairwise2.align.globalcc(seq_1, seq_2, match_function,
#                        gap_function_seq_1, gap_function_seq_2,
#                        gap_char=['-'], score_only=True)

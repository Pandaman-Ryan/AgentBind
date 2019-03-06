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
import itertools
from converter import converter_template as converter

def annotate_with_motif_id(kmer_length, dir_storage, f_jaspar_database, f_profile):
    dir_query = "%s/kmer_meme/" %(dir_storage)
    if not os.path.exists(dir_query):
        os.makedirs(dir_query)
    dir_matches = "%s/tomtom_out/" %(dir_storage)
    if not os.path.exists(dir_matches):
        os.makedirs(dir_matches)

    TF_profile = _read_TF_profile(f_profile)
    kmer_index = 0
    num_kmers = 4**kmer_length

    f_result_table = "%s/kmer2motifs.txt" %(dir_storage)
    with open(f_result_table, 'w') as ofile:
        for kmer_seq in itertools.product('ATCG', repeat=kmer_length):
            f_query = "%s/%d.txt" %(dir_query, kmer_index)
            _write_into_meme(kmer_seq, kmer_index, f_query)

            f_matches = "%s/%d.txt" %(dir_matches, kmer_index)
            _search_matched_motif(f_query, f_jaspar_database, f_matches)

            matched_motifs = _read_matched_motifs(f_matches)

            annotation_str = _annotate_with_motifs(matched_motifs, TF_profile)

            line_to_write = "%s\t%s\n" %("".join(kmer_seq), annotation_str)
            ofile.write(line_to_write)

            kmer_index += 1
            if (kmer_index) % 100 == 0:
                print ("Processed %d/%d kmers" %(kmer_index, num_kmers))
    return

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
    cmd = "tomtom %s %s --text -incomplete-scores -thresh 1.0 -verbosity 1 > %s" %(f_query, f_jaspar_database, f_matches)
    subprocess.check_call(cmd, shell=True)
    return

def _read_matched_motifs(f_tomtom):
    motif_list = []
    thresh_num_motifs = 10
    line_index = 0
    for line in open(f_tomtom):
        line = line.strip()
        if len(line) > 0 and line[0] != "#":
            elems = line.split()
            motif_id = elems[1]
            values = "%s,%s,%s" %(elems[3], elems[4], elems[5])
            motif_list.append((motif_id, values))

        line_index += 1
        if line_index > thresh_num_motifs:
            break
    return motif_list

def _annotate_with_motifs(motif_list, TF_profile):
    annotation_arr = ["_"]
    for (motif_id, values) in motif_list:
        if motif_id in TF_profile:
            motif_name = TF_profile[motif_id]
        else:
            motif_name = "_"
        annotation_arr.append("%s/%s/%s" %(motif_id, values, motif_name))
    annotation_str = ";".join(annotation_arr)
    return annotation_str

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

def main():
    usage = "usage: %prog [options]"
    parser = OptionParser(usage)
    parser.add_option('--datadir', dest='dir_data',
            default='/storage/pandaman/project/AgentBind-Release-Version/data/',
            help='The directory where all input and dependent data are stored [Default: %default]')
    (options, args) = parser.parse_args()

    kmer_length = 8
    f_profile = "%s/Jaspar-motifs/TF_profile.txt"%(options.dir_data)
    f_jaspar_database = "%s/Jaspar-motifs/JASPAR2018_CORE_vertebrates_redundant_pfms_meme.txt" %(options.dir_data)
    dir_storage = "%s/kmer2motifs/"%(options.dir_data)
    if not os.path.exists(dir_storage):
        os.makedirs(dir_storage)

    #####
    # Calculate weight scores for each TF
    #####
    # load data
    annotate_with_motif_id(kmer_length, dir_storage, f_jaspar_database, f_profile)


if __name__ == "__main__":
    main()

##### END OF FILE #####################################

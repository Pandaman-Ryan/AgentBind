import os
import subprocess
import glob
import random
from Bio import SeqIO
from Bio.Seq import Seq

class converter_template:
    ############
    # Download reference genome hg19
    ############
    def _load_ref_genome(self, reference_dir, genome_limit_path):
        chrom_length_dict = self._read_chromosome_length(genome_limit_path)
        ref_dict = self._download_and_read_reference(reference_dir)
        return ref_dict, chrom_length_dict

    def _read_chromosome_length(self, file_addr):
        chrom_length_dict = {}
        for line in open(file_addr):
            elems = line.split()
            chromID = elems[0]
            length = int(elems[1])
            chrom_length_dict[chromID] = length
        return chrom_length_dict

    def _download_and_read_reference(self, reference_dir):
        self._download_genome(reference_dir)
        ref_dict = self._read_genome(reference_dir)
        return ref_dict

    def _download_genome(self, reference_dir):
        current_work_dir = os.getcwd()
        os.chdir(reference_dir)
        ref_path = "%s/hg19.fa" %(reference_dir)
        if not os.path.isfile(ref_path):
            cmd = 'wget ftp://hgdownload.cse.ucsc.edu/goldenPath/hg19/bigZips/chromFa.tar.gz -O chromFa.tar.gz'
            subprocess.call(cmd, shell=True)

            cmd = 'tar -xzvf chromFa.tar.gz'
            subprocess.call(cmd, shell=True)

            cmd = 'cat chr?.fa chr??.fa > hg19.fa'
            subprocess.call(cmd, shell=True)

            os.remove('chromFa.tar.gz')
            for chrom_fa in glob.glob('chr*.fa'):
                os.remove(chrom_fa)
        os.chdir(current_work_dir)
        return

    def _read_genome(self, reference_dir):
        ref_dict = {}
        ref_path = "%s/hg19.fa" %(reference_dir)
        for seq in SeqIO.parse(ref_path, "fasta"):
            chromID = seq.id
            chromSeq = (str(seq.seq)).upper()
            ref_dict[chromID] = chromSeq
        return ref_dict

    ########################
    # File readers
    ########################
    def _readFromFimo(self, fimo_file):
        '''
            read from a fimo file
            fimo-file:
                # motif_id  motif_alt_id    sequence_name   start   stop    strand  score   p-value q-value matched_sequence
                1   HNF4A_HUMAN.H11MO.0.A   chr11   712976  712989  -   21.6571 1.42e-09    0.226   GGGGCCAAAGTCCA
        '''
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

            fimo_list.append((chromID, start, end, pval))
        return fimo_list

    def _readFromNarrowPeak(self, narrowPeak_file):
        '''
            read from a narrow peak file
            narrowPeak:
                chr2    43019655    43020009    .   1000    .   411.825061923654    -1  4.97133616322774    175
        '''
        np_dict = {}
        for line in open(narrowPeak_file):
            elems = line.split()
            chromID = elems[0]
            start = int(elems[1])
            end = int(elems[2])
            signalVal = float(elems[6])

            if chromID in np_dict:
                np_dict[chromID].append((chromID, start, end, signalVal))
            else:
                np_dict[chromID] = [(chromID,start,end,signalVal)]

        return np_dict 

    ############################
    # Useful tools
    ############################
    def _readFromRef(self, chromID, start, end, ref_dict):
        return (ref_dict[chromID][start:end]).upper()

    def _string_to_one_hot(self, seq):
        data = []
        for nuc in seq:
            data.append(self._to_one_hot(nuc))
        data_str = ",".join(data)
        return data_str

    def _to_one_hot(self, nuc):
        nucleotides = ["A", "T", "C", "G"]
        if nuc == "N":
            return ",".join(["0.25" for _ in range(len(nucleotides))])
        else:
            index = nucleotides.index(nuc)
            onehot = ["0" for _ in range(len(nucleotides))]
            onehot[index] = "1"
            return ",".join(onehot)

    def _reverse_complement(self, seq):
        seq_rc = (Seq(seq)).reverse_complement()
        return str(seq_rc)
###END OF FILE #############################################3

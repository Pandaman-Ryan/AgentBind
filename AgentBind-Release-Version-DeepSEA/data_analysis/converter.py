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
    def _load_ref_genome(self, reference_dir):
        ref_dict = self._download_and_read_reference(reference_dir)
        return ref_dict

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

    ############################
    # Useful tools
    ############################
    def _readFromRef(self, chromID, start, end, ref_dict):
        return (ref_dict[chromID][start:end]).upper()

    def _reverse_complement(self, seq):
        seq_rc = (Seq(seq)).reverse_complement()
        return str(seq_rc)
###END OF FILE #############################################3

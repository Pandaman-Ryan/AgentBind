#!/bin/bash

factor=$1
experiment=$2
home_path="/storage/pandaman/project/singletons/"
data_path=${home_path}/${experiment}
weight_path=/storage/pandaman/project/AgentBind-GM12878-DanQ-unfixed-rnn-trans-GC-balanced/storage/AgentBind-GM12878-DanQ/tmp/${factor}+GM12878/seqs_one_hot_c/vis-weights-total/weight.txt
# weight_path=/storage/pandaman/project/AgentBind-GM12878-DanQ-control-group/storage/AgentBind-GM12878-DanQ/tmp/${factor}+GM12878/seqs_one_hot_c/vis-weights-total/weight.txt
# weight_path=/storage/pandaman/project/AgentBind-GM12878-DeepSEA-control-group/storage/AgentBind-GM12878-DeepSEA/tmp/${factor}+GM12878/seqs_one_hot_c/vis-weights-total/weight.txt

echo $factor
mkdir -p ${data_path}/${factor}

# Get per-SNP scores
python get_snp_annots.py --fweight ${weight_path} --resultdir ${data_path} --TFname ${factor}

# Get regions
cat ${data_path}/${factor}/scores.tab | \
    grep -v chrom | awk '{print $1 "\t" $2 "\t" $2+1 "\t" $3 "\t" $4 "\t" $5 "\t" $6}' | \
    sort -k1,1 -k2,2n -T ${data_path}/${factor}/ > \
    ${data_path}/${factor}/regions.bed

# Combine with allele scores. All these commands needed to deal with off by one errors...
echo "chrom,start,end,ref,AC,AN,raw.score,snr.score,rank,core" | sed 's/,/\t/g' > ${data_path}/${factor}/factor_singletons.tab
intersectBed -a /storage/pandaman/project/gnomAD/gnomAD.r3.0.hg19.bed -b ${data_path}/${factor}/regions.bed -wa -wb | \
    cut -f 9-11 --complement | \
    datamash -g 1,2,3,4,5 max 6 max 7 max 8 max 9 max 10 max 11 max 12 | \
    cut -f 5,8 --complement | \
    datamash -g 1,2,3,4 sum 5 max 6 max 7 max 8 max 9 max 10 >> ${data_path}/${factor}/factor_singletons.tab

echo "chrom,start,end,ref,AC,AN,raw.score,snr.score,rank,core" | sed 's/,/\t/g' > ${data_path}/${factor}/factor_singletons_r2.tab
intersectBed -a /storage/pandaman/project/gnomAD/gnomAD.r2.1.1.hg19.bed -b ${data_path}/${factor}/regions.bed -wa -wb | \
    cut -f 9-11 --complement | \
    datamash -g 1,2,3,4,5 max 6 max 7 max 8 max 9 max 10 max 11 max 12 | \
    cut -f 5,8 --complement | \
    datamash -g 1,2,3,4 sum 5 max 6 max 7 max 8 max 9 max 10 >> ${data_path}/${factor}/factor_singletons_r2.tab

echo "chrom,start,end,ref,AF,raw.score,snr.score,rank,core" | sed 's/,/\t/g' > ${data_path}/${factor}/factor_singletons_1k.tab
cat /storage/pandaman/project/singletons/1000Genomes/1KG_afreqs.bed | \
    awk '{print "chr" $1 "\t" $2-1 "\t" $3-1 "\t" $4 "\t" $5 "\t" $6}' | \
    intersectBed -a stdin -b ${data_path}/${factor}/regions.bed -wa -wb | \
    cut -f 7-9 --complement | \
    datamash -g 1,2,3,4,5 max 6 max 7 max 8 max 9 max 10 | \
    cut -f 5 --complement | \
    datamash -g 1,2,3,4 sum 5 max 6 max 7 max 8 max 9 > ${data_path}/${factor}/factor_singletons_1k.tab

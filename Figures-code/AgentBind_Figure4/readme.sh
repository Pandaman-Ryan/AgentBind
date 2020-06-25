#!/bin/bash

# TODO: get tabix indexed file of chrom, start, end, ref, alt, afreq. only include SNPs
# see 1000Genomes folder here: /storage/pandaman/project/AgentBind-GM12878-analysis/data/
# THen use intersectbed to get info
#for chr in $(seq 1 22); do cat /storage/pandaman/project/AgentBind-GM12878-analysis/data/1000genomes/allele_chr${chr}.txt; done | \
#    awk '(length($3)==1 && length($4)==1 && $3~/A|C|G|T/ && $4~/A|C|G|T/) {print $1 "\t" $2 "\t" $2+1 "\t" $3 "\t" $4 "\t" $5}' | \
#    bgzip -c > /storage/mgymrek/agent-bind/singletons/1000Genomes/1KG_afreqs.vcf.gz
#tabix -p vcf /storage/mgymrek/agent-bind/singletons/1000Genomes/1KG_afreqs.vcf.gz

factors="FOS STAT1 CEBPB JunD STAT3 RFX5 ETS1 NFYA CTCF EBF1 SP1 PU1 RUNX3 NFYB Nrf1 ELF1 NFKB TCF3 Mxi1 USF1 YY1 USF2 ZEB1 PAX5 POU2F2 NRSF PBX3 MEF2A E2F4 BHLHE40 ELK1 NFIC MEF2C Max SRF Znf143 IRF4 ZBTB33"

for factor in ${factors}
do
    echo "./run_factor.sh ${factor} AgentBind"
done | xargs -P1 -I% -n1 sh -c "%" # uses lots of memory. don't do more than this

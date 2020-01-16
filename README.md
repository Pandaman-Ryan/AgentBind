# AgentBind #

AgentBind is a machine-learning framework for analyzing the context regions of binding sites and identifying specific non-coding nucleotides with the strong effects on binding activities. This code repository contains code for the classification + visualization experiments with the DanQ and DeepSEA architectures respectively.

## System Requirement & Installation ##
All experiments are executed on CentOS Linux 7 (core) with Python (v2.7.5). Prior to your code execution, you need to make sure you have downloaded:

**FIMO from the MEME-suite**

You can download the MEME-suite from http://meme-suite.org/doc/download.html. This will give you a package of tools including FIMO. You also need to run the following command line to ensure your FIMO can be executed directly.

`export PATH={YOUR-PATH}/MEME-Suite/bin:$PATH`

### python libraries ###
Our code requires external python libraries including: tensorflow v1.9.0 GPU-version, biopython, numpy, and matplotlib. You can install them with the pip package manager:

`pip install numpy six matplotlib biopython tensorflow-gpu`

## Data Download ##
**Data for experiments with the DanQ architecture**

**Data for experiments with the DeepSEA architecture**
https://drive.google.com/file/d/1UaaqgFlce9FSaBX2RoIz9pDaXacwQ3lW/view?usp=sharing

## Run ##
**AgentBind.py** is the python script which helps you execute all experiments.

**Required parameters:**
* --datadir: the directory where you stored the downloaded data
* --motif: the names, motif-IDs, and ChIPseq files. They are included in the given data: `{your-data-path}/table_matrix/table_core_motifs.txt`
* --workdir: the directory that stores all the intermediate/oversized files including the well-trained models and one-hot-encoded input sequences
* --resultdir: the directory that stores all the results

To run AgentBind, you can simply execute:
```
python AgentBind.py 
--motif {your-data-path}/table_matrix/table_core_motifs.txt 
--workdir {your-work-path}
--datadir {your-data-path}
--resultdir {your-result-path}
```

AgentBind reports the results of two situations, core motifs present (c) and blocked (b). You can find the correspounding classifications results (AUC curves) in: `{your-result-path}/{b or c}/{TF-name}+GM12878/`. And the annoation scores are saved at `{your-work-path}/ETS1+GM12878/seqs_one_hot_{b or c}/vis-weights-total/weight.txt`.

The python program takes ~24-48 hours to complete. If you only want the annotation scores, you can download them here directly (DanQ version only):
* https://drive.google.com/file/d/1HB-_bG1K6rbbtBxh2OQ2ldL5BVp3NlBQ/view?usp=sharing

For questions on usage, please open an issue, submit a pull request, or contact An Zheng (anz023@eng.ucsd.edu).

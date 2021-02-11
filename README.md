# AgentBind #
<a href="https://zenodo.org/badge/latestdoi/174050946"><img src="https://zenodo.org/badge/174050946.svg" alt="DOI"></a>

AgentBind is a machine-learning framework for analyzing context regions of binding sites and identifying specific non-coding nucleotides with strong effects on binding activities. This code repository contains code for the classification + visualization experiments with the DanQ and DeepSEA architectures respectively.

Published paper: [**click here**](https://rdcu.be/cdMmE)

Please cite: \
`Zheng, A., Lamkin, M., Zhao, H. et al. Deep neural networks identify sequence context features predictive of transcription factor binding. Nat Mach Intell (2021).`

## System Requirement & Installation ##
All experiments are executed on CentOS Linux 7 (core) with Python (v2.7.5). Prior to your code execution, please make sure you have installed the following tools/libraries.

**Install FIMO from the MEME-suite**

You can download the MEME-suite from http://meme-suite.org/doc/download.html. This will give you a package of tools including FIMO. You also need to run the following command line to set up a short-cut for FIMO:

`export PATH={YOUR-PATH}/MEME-Suite/bin:$PATH`

### python libraries ###
Our code requires external python libraries including tensorflow v1.9.0 GPU-version, biopython v1.71, numpy v1.15.4, six v1.14.0, scikit-image v0.14.5, and matplotlib. You can install them with the pip package manager:

`pip install numpy six matplotlib biopython sklearn scikit-image tensorflow-gpu==1.9.0`

## Data Download ##
**Data for experiments with the DanQ architecture**
https://drive.google.com/file/d/12mrLk9Ci7u2tKB8kuqldGXE9ghAzpbUk/view?usp=sharing

**Data for experiments with the DeepSEA architecture**
https://drive.google.com/file/d/1UaaqgFlce9FSaBX2RoIz9pDaXacwQ3lW/view?usp=sharing

## Run ##
**AgentBind.py** is the go-to python script which execute all the experiments.

**Required parameters:**
* --datadir: the directory where you stored the downloaded data.
* --motif: a text file containing the names, motifs, and ChIPseq files of TFs of interest. This text file can be found in the given data at `{your-data-path}/table_matrix/table_core_motifs.txt`.
* --workdir: a directory where to store all the intermediate/oversized files including the well-trained models, one-hot-encoded input sequences, and Grad-CAM annoation scores.
* --resultdir: a directory where to store all the results.

To run AgentBind, you can simply execute:
```
python AgentBind.py 
--motif {your-data-path}/table_matrix/table_core_motifs.txt 
--workdir {your-work-path}
--datadir {your-data-path}
--resultdir {your-result-path}
```

AgentBind reports results of two situations, core motifs present (c) and blocked (b). You can find the correspounding classification results (AUC curves) in: `{your-result-path}/{b or c}/{TF-name}+GM12878/`. And the Grad-CAM annoation scores are available at `{your-work-path}/{TF-name}+GM12878/seqs_one_hot_{b or c}/vis-weights-total/weight.txt`.

The python program "AgentBind.py" takes ~24-48 hours to complete. If you need the Grad-CAM annotation scores only, you can directly download them here (DanQ version only):
* https://drive.google.com/file/d/1HB-_bG1K6rbbtBxh2OQ2ldL5BVp3NlBQ/view?usp=sharing

For questions on usage, please open an issue, submit a pull request, or contact Melissa Gymrek (mgymrek@ucsd.edu) or An Zheng (anz023@eng.ucsd.edu).

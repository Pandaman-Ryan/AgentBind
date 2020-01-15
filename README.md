# AgentBind #

AgentBind is a machine-learning framework for analyzing the context regions of binding sites and identifying specific non-coding nucleotides with the strong effects on binding activities. This code repository contains code for the classification and visualization experiments for different TF binding sites using the DanQ and DeepSEA architectures.

## System Requirement & Installation ##
All experiments are executed on CentOS Linux 7 (core) with Python 2.7. Prior to your code execution, you need to make sure you have downloaded:

**Fimo from the MEME-suite**

You can download the MEME-suite from http://meme-suite.org/doc/download.html. This will give you a package of tools including Fimo. You also need to run the following command to ensure your Fimo can be executed directly.

`export PATH=/home/pandaman/tools_and_dependencies/MEME-Suite/bin:$PATH`

### python libaries ###
You can install them with pip package manager:
`pip install numpy six matplotlib biopython tensorflow-gpu`

## Data Download ##

## Demo ##


The annotation scores produced from the experiment type 1 can be found here:
* https://drive.google.com/file/d/1HB-_bG1K6rbbtBxh2OQ2ldL5BVp3NlBQ/view?usp=sharing

All the experiments were implemented using Python scripts (v2.7.5). The machine learning architectures were implemented with the help of Tensorflow v1.9.0. For the installation of Tensorflow, you can find a download link from the Tensorflow website: https://www.tensorflow.org/install. Other exterior python packages we used include: numpy, sklearn, subprocess, and OptionParser. For questions on usage, please open an issue, submit a pull request, or contact An Zheng (anz023@eng.ucsd.edu).

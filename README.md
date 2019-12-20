# AgentBind

AgentBind is a machine-learning framework for analyzing the context regions of binding sites and identifying specific non-coding nucleotides with the strong effects on binding activities. The details of this framework and its applications can be found in our paper.

This code repository contains code for four types of experiments described in our paper:

1. the classification and visualization experiments for different TF binding sites using the DanQ architecture
2. the classification experiments for TF binding sites using the DeepSEA architecture
3. the benchmarking experiments against IMPACT
4. the simulation experiments for the evaluations of interpretation methods.

The annotation scores produced from the experiment type 1 can be found here:
https://drive.google.com/file/d/1HB-_bG1K6rbbtBxh2OQ2ldL5BVp3NlBQ/view?usp=sharing

All the experiments were implemented using Python scripts (v2.7.5). The machine learning architectures were implemented with the help of Tensorflow v1.9.0. For the installation of Tensorflow, you can find a download link from the Tensorflow website: https://www.tensorflow.org/install. Other exterior python packages we used include: numpy, sklearn, subprocess, and OptionParser. For questions on usage, please open an issue, submit a pull request, or contact An Zheng (anz023@eng.ucsd.edu).

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 

f_results_ab = "/storage/pandaman/project/AgentBind-GM12878/results/c/auc_summary.txt"
f_data_size = "/storage/pandaman/project/AgentBind-GM12878-Run-1/results/c/data_size.txt"
f_results_ds = "/storage/pandaman/project/AgentBind-DeepSEA/results/auc_on_DeepSEA.txt"

roc_scores = {'ab':{}, 'ds':{}}
pr_scores = {'ab':{}, 'ds':{}}
data_sizes = {}
TF_names = []

dir_test_prefix = "/storage/pandaman/project/AgentBind-GM12878/tmp/"
dir_test_suffix = "/seqs_one_hot_c/test/data.txt"
for line in open(f_data_size):
    TF_name, n_pos, n_neg = line.strip().split()
    n_pos = int(n_pos)
    n_neg = int(n_neg)

    size_test_data = sum(1 for line in open("%s/%s/%s" %(dir_test_prefix, TF_name, dir_test_suffix)))
    TF_names.append(TF_name)
    #data_sizes[TF_name] = n_pos + n_neg
    data_sizes[TF_name] = size_test_data

for line in open(f_results_ab):
    TF_name, roc_score, pr_score = line.strip().split()
    roc_score = float(roc_score)
    pr_score = float(pr_score)

    roc_scores['ab'][TF_name] = roc_score
    pr_scores['ab'][TF_name] = pr_score
    
for line in open(f_results_ds):
    TF_name, roc_score, pr_score = line.strip().split()
    roc_score = float(roc_score)
    pr_score = float(pr_score)

    roc_scores['ds'][TF_name] = roc_score
    pr_scores['ds'][TF_name] = pr_score

roc_scores_list = {'ab':[], 'ds':[]}
pr_scores_list = {'ab':[], 'ds':[]}
TF_names_thresholded = []
for TF_name in TF_names:
    if data_sizes[TF_name] > 100:
        roc_scores_list['ab'].append(roc_scores['ab'][TF_name])
        pr_scores_list['ab'].append(pr_scores['ab'][TF_name])
        roc_scores_list['ds'].append(roc_scores['ds'][TF_name])
        pr_scores_list['ds'].append(pr_scores['ds'][TF_name])
        TF_names_thresholded.append(TF_name)

# plot
def draw_bar_graph(data_list, TF_names, f_output):
    fig, ax = plt.subplots()
    index = np.arange(len(TF_names))
    bar_width = 0.35
    opacity = 0.4

    rects1 = ax.bar(index, data_list['ab'], bar_width,
                    alpha=opacity, color='b',
                    label='AgentBind')

    rects2 = ax.bar(index + bar_width, data_list['ds'], bar_width,
                    alpha=opacity, color='r',
                    label='DeepSEA')

    ax.set_xlabel('TF')
    ax.set_ylabel('scores')
    ax.set_title('ROC scores')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(TF_names)
    ax.legend()
    fig.tight_layout()
    plt.savefig("./%s" %(f_output))
    plt.close()

def draw_boxplot(data, filename):
    plt.figure()
    plt.boxplot([data['ab'], data['ds']])
    plt.title(filename)
    plt.savefig("%s" %(filename))
    plt.close()
    return

def draw_scatter(data, filename):
    plt.figure()
    plt.scatter(data['ds'], data['ab'])
    plt.plot([0.7,1], [0.7,1], 'k--')
    plt.savefig("%s" %(filename))
    plt.close()
    return

draw_bar_graph(roc_scores_list, TF_names_thresholded, "roc_agentbind_vs_deepsea.png")
draw_bar_graph(pr_scores_list, TF_names_thresholded, "pr_agentbind_vs_deepsea.png")
draw_boxplot(roc_scores_list, "roc_boxplot.png")
draw_boxplot(pr_scores_list, "pr_boxplot.png")
draw_scatter(roc_scores_list, "roc_scatter.png")
draw_scatter(pr_scores_list, "pr_scatter.png")

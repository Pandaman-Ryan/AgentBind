from sklearn.metrics import roc_auc_score
import numpy as np
import sys

def get_auc():
    score_file = sys.argv[1]
    true_file = sys.argv[2]

    # read in labels from both files to find roc_auc_score
    gen_score_data = np.genfromtxt(score_file, delimiter=",", dtype=str)
    print (gen_score_data[0])
    exit(1)
    print (gen_score_data[0][561])
    print (gen_score_data[0][:5])
    print (gen_score_data[1][:5])
    print (len(gen_score_data[0]))
    print (len(gen_score_data[1]))
    HNF4a_index = [i for i,s in enumerate(gen_score_data[0]) if 'GM12878|STAT1' in s]
    print HNF4a_index
    score_labels = {}

    score_info = []
    header = True
    for loc in gen_score_data:
        if header:
            header = False
            continue

        score_labels[(int(loc[2]), int(loc[3]))] = float(loc[HNF4a_index])
        #score_labels.append(float(loc[HNF4a_index]))
        score_info.append((int(loc[2]), int(loc[3])))


    true_label_info = []
    true_labels = {}
    true_data = np.genfromtxt(true_file, delimiter='\t', dtype=str)
    for loc in true_data:
        #true_labels.append(int(loc[4]))
        true_label_info.append((int(loc[1]), int(loc[2])))
        true_labels[(int(loc[1]), int(loc[2]))] = int(loc[4])

    '''
    high_dif = [] 
    for i,s in enumerate(true_labels):
        if abs(true_labels[i] - score_labels[i]) > 0.50:
            print "High difference in probability: %f and %d"%(score_labels[i], true_labels[i])
            high_dif.append((score_labels[i], true_labels[i]))

    print len(high_dif)
    '''
    # get score
    true_labels_list = []
    score_labels_list = []
    for (start, end) in true_labels:
        if ((start,end) in score_labels):
            true_labels_list.append(true_labels[(start,end)])
            score_labels_list.append(score_labels[(start,end)])
        elif ((start-1, end-1) in score_labels):
            true_labels_list.append(true_labels[(start,end)])
            score_labels_list.append(score_labels[(start-1,end-1)])
        elif ((start+1, end+1) in score_labels):
            true_labels_list.append(true_labels[(start,end)])
            score_labels_list.append(score_labels[(start+1,end+1)])
        else:
            exit ("wrong position! %d-%d" %(start, end))
    #true_label_info.sort()
    #score_info.sort()
    #for smpl_index in range(len(true_label_info)):
    #    print (true_label_info[smpl_index], score_info[smpl_index])
    auc_score = roc_auc_score(true_labels_list, score_labels_list)
    print("AUC score for HNF4A of DeepSea: %f"%(auc_score))
    
    return


if __name__ == '__main__':
    get_auc()

''' Unless stated otherwise, all software is provided free of charge. 
As well, all software is provided on an "as is" basis without warranty 
of any kind, express or implied. Under no circumstances and under no legal 
theory, whether in tort, contract, or otherwise, shall Liang Zhang be liable 
to you or to any other person for any indirect, special, incidental, 
or consequential damages of any character including, without limitation, 
damages for loss of goodwill, work stoppage, computer failure or malfunction, 
or for any and all other damages or losses. If you do not agree with these terms, 
then you are advised to not use the software.'''

from uril_tools import *
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn import metrics
from math import sqrt
import numpy as np
import pyprind
np.set_printoptions(threshold=np.inf)

def get_evaluate_result(actual_labels, pred_prob):
    rmse = sqrt(mean_squared_error(actual_labels, pred_prob))
    fpr, tpr, thresholds = metrics.roc_curve(actual_labels, pred_prob, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    r2 = r2_score(actual_labels, pred_prob)
    return rmse,auc,r2

def run_epoch(session, m, students, eval_op, verbose=False):
    pred_prob = []
    actual_labels = []                   # use for whole comparasion

    skill_id_origin_list = []
    target_id_origin_list = []
    iteration = int(len(students)/m.batch_size)

    for i_iter in pyprind.prog_percent(range(iteration)):
        #bar.update(m.batch_size)
        x = np.zeros((m.batch_size, m.num_steps, m.seq_width))

        target_id = np.array([],dtype=np.int32)
        skill_id_origin = np.array([],dtype=np.int32)
        target_id_origin = np.array([],dtype=np.int32)
        target_correctness = []         # use for just a batch

        #load data for a batch
        # tuple formate
        # 0: user_id
        # 1: record_numb
        # 2: data
        # 3: Target_Id
        # 4: correctness
        for i_batch in range(m.batch_size):
            student = students[i_iter*m.batch_size+i_batch]
            record_num = student[1]
            #record_content_pd = student[2].reset_index(drop=True)
            record_content = student[2].as_matrix()
            temp_skill_id_list = list(student[2]['skill_id'])
            skill_id = student[3]
            correctness = student[4]

            # construct data for training:
            # data ~ x
            # target_id ~ skill_id
            # target_correctness ~ correctness
            for i_recordNumb in range(record_num):
                if(i_recordNumb<m.num_steps):
                    x[i_batch, i_recordNumb,:] = record_content[i_recordNumb,:]

                    if skill_id[i_recordNumb] in m.skill_set:
                        temp =i_batch*m.num_steps*m.skill_num + i_recordNumb*m.skill_num + skill_id[i_recordNumb]
                        temp_i = skill_id[i_recordNumb]
                        temp_s = temp_skill_id_list[i_recordNumb]
                    else:
                        temp = i_batch*m.num_steps + i_recordNumb*m.skill_num + 0
                        temp_i = 0
                        temp_s = temp_skill_id_list[i_recordNumb]

                    target_id = np.append(target_id,[[temp]])
                    target_id_origin  = np.append(target_id_origin,[[temp_i]])
                    skill_id_origin = np.append(skill_id_origin,[[temp_s]])

                    target_correctness.append(int(correctness[i_recordNumb]))
                    actual_labels.append(int(correctness[i_recordNumb]))
                else:
                    break

            #test inter_skill and intra_skill
            """
            if (record_num<=m.num_steps):
                skill_id_origin = np.append(skill_id_origin,temp_skill_id_list)
            else:
                skill_id_origin = np.append(skill_id_origin,temp_skill_id_list[:m.num_steps])
            """
        pred, _ = session.run([m.pred, eval_op],feed_dict={m.inputs: x,
                                                           m.target_id: target_id,
                                                           m.target_correctness: target_correctness})

        for s in skill_id_origin:
            skill_id_origin_list.append(s)

        for t in target_id_origin:
            target_id_origin_list.append(t)

        for p in pred:
            pred_prob.append(p)

    # print ("------------------len ",len(skill_id_origin_list),"\t",len(target_id_origin_list))
    # print (skill_id_origin_list[:100])
    # print (target_id_origin_list[:100])
    rmse,auc,r2 = get_evaluate_result(actual_labels, pred_prob)

    #print ("==> predict_prob shape\t",np.shape(pred_prob),'\tactual_labels\t',np.shape(actual_labels),'\ttarget_id_list\t',np.shape(target_id_origin_list))
    #print (target_id_origin_list[1:100])
    intra_skill_actual = []
    intra_skill_pred = []

    inter_skill_actual = []
    inter_skill_pred = []

    for idx in np.arange(len(target_id_origin_list)):
        if skill_id_origin_list[idx]==target_id_origin_list[idx]:
            intra_skill_actual.append(actual_labels[idx])
            intra_skill_pred.append(pred_prob[idx])
        else:
            inter_skill_actual.append(actual_labels[idx])
            inter_skill_pred.append(pred_prob[idx])

    inter_rmse,inter_auc,inter_r2 = get_evaluate_result(inter_skill_actual, inter_skill_pred)
    intra_rmse,intra_auc,intra_r2 = get_evaluate_result(intra_skill_actual, intra_skill_pred)

    return rmse, auc, r2,inter_rmse,inter_auc,inter_r2,intra_rmse,intra_auc,intra_r2

if __name__=="__main__":
    pass


from aux import *
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn import metrics
from math import sqrt
import pyprind
np.set_printoptions(threshold=np.inf)

def run_epoch(session, m, students, eval_op, verbose=False):
    pred_prob = []
    actual_labels = []                   # use for whole comparasion
    iteration = int(len(students)/m.batch_size)

    for i_iter in pyprind.prog_percent(range(iteration)):
        #bar.update(m.batch_size)
        x = np.zeros((m.batch_size, m.num_steps, m.seq_width))

        target_id = np.array([],dtype=np.int32)
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
                    else:
                        temp = i_batch*m.num_steps + i_recordNumb*m.skill_num + 0
                    target_id = np.append(target_id,[[temp]])
                    target_correctness.append(int(correctness[i_recordNumb]))
                    actual_labels.append(int(correctness[i_recordNumb]))
                else:
                    break
        pred, _ = session.run([m.pred, eval_op],feed_dict={m.inputs: x,
                                                           m.target_id: target_id,
                                                           m.target_correctness: target_correctness})

        for p in pred:
            pred_prob.append(p)

    rmse = sqrt(mean_squared_error(actual_labels, pred_prob))
    fpr, tpr, thresholds = metrics.roc_curve(actual_labels, pred_prob, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    r2 = r2_score(actual_labels, pred_prob)
    return rmse, auc, r2

if __name__=="__main__":
    pass

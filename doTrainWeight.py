"""
these codes are based on
1. https://github.com/Cospel/rbm-ae-tf
"""

from uril_rbm import RBM
from uril_autoEncoder import AutoEncoder
import tensorflow as tf
import numpy as np
import pyprind, os
import code0_parameter as code0
import code1_data as code1
from uril_oneHotEncoder import ONEHOTENCODERINPUT
from code0_parameter import TARGETSIZE, AUTOENCODER_LIST,AUTOENCODER_ACT,BASELINE


def train_RBM_AE_Weights(data,dp,ap):
    if AUTOENCODER_LIST[-1]!=TARGETSIZE:
        raise ValueError("check last value of AUTOENCODER_LIST should be TARGETSIZE in code0")
    if AUTOENCODER_ACT == 'tanh':
        transfer_function=tf.nn.tanh
    elif AUTOENCODER_ACT == 'sigmoid':
        transfer_function=tf.nn.sigmoid

    weightSaveNameList = ['./weights/' + dp.dataSetType + '/rbmw1.chp', './weights/' + dp.dataSetType + '/rbmw2.chp',
                          './weights/' + dp.dataSetType + '/rbmw3.chp', ]
    for fileName in weightSaveNameList:
        if os.path.exists(fileName):
            os.remove(fileName)
            print(fileName, " is delete")

    startWidth = np.shape(data)[-1]
    iterations = int(len(data) / ap.batch_size)

    # RBMs
    weightsList = []
    weightSaveNameList = []
    rbm_data = data
    for i_rbm in range(len(AUTOENCODER_LIST)):
        weightSaveNameList.append('./weights/' + dp.dataSetType + '/rbmw'+str(i_rbm+1)+'.chp')
        
        if i_rbm ==0: tempStart=startWidth
        else: tempStart=AUTOENCODER_LIST[i_rbm-1]
        weightsList.append(['rbmw'+str(i_rbm+1),'rbmhb'+str(i_rbm+1)])

        rbmobject = RBM(tempStart, int(AUTOENCODER_LIST[i_rbm]), ['rbmw'+str(i_rbm+1), 'rbvb'+str(i_rbm+1), 'rbmhb'+str(i_rbm+1)], 0.3,transfer_function)
   
        print('-' * 25, str(i_rbm+1)+' rbm','-' * 25)
        for i_epoch in range(ap.epoch_rbm):
            for j in pyprind.prog_percent(range(iterations), title=str(i_epoch+1) + '/' + str(ap.epoch_rbm) + " epoch"):
                batch_xs = rbm_data[j * ap.batch_size:(j + 1) * ap.batch_size, :]
                rbmobject.partial_fit(batch_xs)
            print('\tCost is\t', rbmobject.compute_cost(rbm_data))
        rbmobject.save_weights(weightSaveNameList[i_rbm])
        rbm_data = rbmobject.transform(rbm_data)

    # Autoencoder
    autoencoder = AutoEncoder(startWidth, AUTOENCODER_LIST,
                             weightsList, tied_weights=False,transfer_function=transfer_function)
    for i in range(len(AUTOENCODER_LIST)):
        autoencoder.load_rbm_weights(weightSaveNameList[i], weightsList[i], i)
    
    print('-' * 25, 'Train autoEncoder','-' * 25)
    for i_epoch in range(ap.epoch_autoencoder):
        cost = 0.0
        for j in pyprind.prog_percent(range(iterations), title=str(i_epoch+1) + '/' + str(ap.epoch_autoencoder) + " epoch"):
            batch_xs = data[j * ap.batch_size:(j + 1) * ap.batch_size, :]
            cost += autoencoder.partial_fit(batch_xs)
        print('\tCost is\t', cost)

    autoencoder.save_weights('./weights/' + dp.dataSetType)

def Transfer_data(dataset,dp,ap):
    g = tf.Graph()
    with g.as_default():
        inputs = tf.placeholder(tf.float32, [ap.batch_size, ap.num_steps,
                                             dp.seq_width])
        m = ONEHOTENCODERINPUT(ap, dp, inputs,printControl=False)

    with tf.Session(graph=g) as sess:
        iterations = int(len(dataset) / (ap.batch_size * ap.num_steps))
        dataset = dataset.as_matrix()
        x_sum = []
        for j in pyprind.prog_percent(range(iterations), title="transfer data"):
            tmpData = dataset[j * ap.batch_size * ap.num_steps:(j + 1) * ap.batch_size * ap.num_steps, :]
            record_content = tmpData.reshape([ap.batch_size, ap.num_steps, dp.seq_width])
            tmpResult = sess.run(m.get_init_value_for_train_weights(), feed_dict={inputs: record_content})
            if j == 0:
                x_sum = tmpResult
            else:
                x_sum = np.vstack([x_sum, tmpResult])
    return x_sum


if __name__ == "__main__":
    if not BASELINE:
        dp = code0.DatasetParameter()
        dataset, labels= code1.load_data(dp)

        dp.skill_num = len(dataset['skill_id'].unique()) + 1
        dp.skill_set = list(dataset['skill_id'].unique())
        dp.columns_max, dp.columns_numb, dp.columnsName_to_index = code1.get_columns_info(dataset)
        dp.seq_width = len(dp.columnsName_to_index)

        ap = code0.autoencoderParameter()

        data = Transfer_data(dataset,dp,ap)
        train_RBM_AE_Weights(data,dp=dp,ap=ap)
    else:
        print ("not need train weights")

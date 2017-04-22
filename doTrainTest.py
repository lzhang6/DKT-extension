""" Code of deep knowledge tracing-assistment 2014-2015 dataset
Reference:
    1. https://github.com/siyuanzhao/2016-EDM/
    2. https://www.tensorflow.org/versions/0.6.0/tutorials/recurrent/index.html
    3. https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/rnn/ptb/ptb_word_lm.py
    4. https://github.com/Cospel/rbm-ae-tf

Run code:
    1. only set the hyperparameter in code0_params.py
    2. train your autoencoder parameters
       python trainWeights.py
    3. python doAll.py

Environment:
    1. ubuntu 14.04
    2. python3
    3. tensorflow : 0.10
    4. cuda 7.5
    5. GPU GTX1070 (8G)
    6. CPU i5-6600k
    7. RAM: 16G
"""
from __future__ import print_function

import code0_parameter as code0
import code1_data as code1
import code2_model as code2
import code3_runEpoch as code3
import uril_tools as aux
import tensorflow as tf
import numpy as np
import pandas as pd
import datetime
from trainAutoEncoder import trainAEWeights

np.set_printoptions(threshold=np.inf)


def main(unused_args):
    aux.check_directories()

    if not code0.BASELINE and code0.AUTOENCODER_LABEL:
        trainAEWeights()

    dp = code0.DatasetParameter()
    dataset, labels = code1.load_data(dp)
    tuple_data = code1.convert_data_labels_to_tuples(dataset, labels)

    skill_num = len(dataset['skill_id'].unique()) + 1
    dp.skill_num = skill_num
    dp.skill_set = list(dataset['skill_id'].unique())
    dp.columns_max, dp.columns_numb, dp.columnsName_to_index = code1.get_columns_info(dataset)
    dp.seq_width = len(dp.columnsName_to_index)

    print("-" * 50, "\ndp.columns_max\n", dp.columns_max, "\n")
    print("-" * 50, "\ndp.columns_numb\n", dp.columns_numb, "\n")
    print("-" * 50, "\ndp.columnsName_to_index\n", dp.columnsName_to_index, "\n")

    config = code0.ModelParamsConfig(dp)
    eval_config = code0.ModelParamsConfig(dp)

    if dp.dataSetType == 'kdd':
        config.num_steps = 1500
    elif dp.dataSetType == 'cmu_stat_f2011':
        config.num_steps = 1500
    else:
        config.num_steps = aux.get_num_step(dataset)

    eval_config.num_steps = config.num_steps
    eval_config.batch_size = 2

    config.skill_num = skill_num
    eval_config.skill_num = config.skill_num

    name_list = ['cv', 'epoch', 'type', 'rmse', 'auc', 'r2', 'inter_rmse', 'inter_auc', 'inter_r2', 'intra_rmse',
                 'intra_auc', 'intra_r2']
    result_data = pd.DataFrame(columns=name_list)
    CVname = ['c1', 'c2', 'c3', 'c4', 'c5']
    size = len(tuple_data)

    # write all the records to log file
    aux.printConfigration(config=config, dp=dp, train_numb=int(size * 0.8), test_numb=int(size * 0.2))
    aux.logwrite(["==> model_continues_columns\n" + ','.join(dp.model_continues_columns)], dp, True)
    aux.logwrite(["==> model_category_columns\n" + ','.join(dp.model_category_columns)], dp, True)
    str_cross_columns_list = ['-'.join(i) for i in dp.model_cross_columns]
    str_cross_columns = ','.join(str_cross_columns_list)
    aux.logwrite(["==> model_cross_columns\n" + str_cross_columns], dp, True)

    for index, cv_num_name in enumerate(CVname):
        aux.logwrite(["\nCross-validation: \t" + str(index + 1) + "/5"], dp, prt=True)
        timeStampe = datetime.datetime.now().strftime("%m-%d-%H:%M")
        aux.logwrite(["\ntime:\t" + timeStampe], dp)

        train_tuple_rows = tuple_data[:int(index * 0.2 * size)] + tuple_data[int((index + 1) * 0.2 * size):]
        test_tuple_rows = tuple_data[int(index * 0.2 * size): int((index + 1) * 0.2 * size)]

        with tf.Graph().as_default(), tf.Session() as session:
            initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)
            # training model
            print("\n==> Load Training model")
            with tf.variable_scope("model", reuse=None, initializer=initializer):
                m = code2.Model(is_training=True, config=config, dp=dp)
            # testing model
            print("\n==> Load Testing model")
            with tf.variable_scope("model", reuse=True, initializer=initializer):
                mtest = code2.Model(is_training=False, config=eval_config, dp=dp)

            tf.initialize_all_variables().run()

            print("==> begin to run epoch...")
            for i in range(config.max_max_epoch):
                lr_decay = config.lr_decay ** max(i - config.max_epoch, 0)
                m.assign_lr(session, config.learning_rate * lr_decay)

                rt = session.run(m.lr)
                rmse, auc, r2, inter_rmse, inter_auc, inter_r2, intra_rmse, intra_auc, intra_r2 = code3.run_epoch(
                    session, m, train_tuple_rows, m.train_op, verbose=True)

                aux.print_result(dp, cv_num_name, i, rt, rmse, auc, r2, inter_rmse, inter_auc, inter_r2, intra_rmse,
                                 intra_auc, intra_r2, 'train')

                result_data = result_data.append(pd.Series(
                    [cv_num_name, i, 'train', rmse, auc, r2, inter_rmse, inter_auc, inter_r2, intra_rmse, intra_auc,
                     intra_r2], index=name_list), ignore_index=True)

                display = 5
                if ((i + 1) % display == 0):
                    print('BEGIN', "-" * 80)
                    rmse, auc, r2, inter_rmse, inter_auc, inter_r2, intra_rmse, intra_auc, intra_r2 = code3.run_epoch(
                        session, mtest, test_tuple_rows, tf.no_op())
                    aux.print_result(dp, cv_num_name, i, rt, rmse, auc, r2, inter_rmse, inter_auc, inter_r2, intra_rmse,
                                     intra_auc, intra_r2, 'test', display)
                    print('END--', "-" * 80)

                    result_data = result_data.append(pd.Series(
                        [cv_num_name, (i + 1) / display, 'test', rmse, auc, r2, inter_rmse, inter_auc, inter_r2,
                         intra_rmse, intra_auc, intra_r2], index=name_list), ignore_index=True)

                #print ("-*"*50,"\n",result_data)

    print("==> Finsih! whole process, save result and print\t" + dp.currentTime)

    temp_data = result_data[result_data['type'] == 'test']
    for idx in set(temp_data['epoch']):
        tp = temp_data[temp_data['epoch'] == idx]
        result_data = result_data.append(pd.Series(
            ['average', idx, 'test_mean', tp['rmse'].mean(), tp['auc'].mean(), tp['r2'].mean(), tp['inter_rmse'].mean(),
             tp['inter_auc'].mean(), tp['inter_r2'].mean(), tp['intra_rmse'].mean(), tp['intra_auc'].mean(),
             tp['intra_r2'].mean()], index=name_list), ignore_index=True)

    print(result_data[result_data['cv']=='average'])
    result_data.to_csv('./result/'+code0.DATASETTYPE+'/result_'+timeStampe+'.csv')
    print('==> save to ./result/'+code0.DATASETTYPE+'/result_'+timeStampe+'.csv')


if __name__ == "__main__":
    tf.app.run()

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
import aux
import tensorflow as tf
import numpy as np
import pandas as pd
import datetime
from trainAutoEncoder import trainAEWeights

np.set_printoptions(threshold=np.inf)

def main(unused_args):
    if not code0.BASELINE and code0.AUTOENCODER_LABEL:
        trainAEWeights()

    dp = code0.DatasetParameter()
    dataset, labels = code1.load_data(dp)
    tuple_data = code1.convert_data_labels_to_tuples(dataset, labels)

    skill_num = len(dataset['skill_id'].unique()) + 1  # 0 for unlisted skill_id
    dp.skill_num = skill_num
    dp.skill_set = list(dataset['skill_id'].unique())
    dp.columns_max, dp.columns_numb, dp.columnsName_to_index = code1.get_columns_info(dataset)
    dp.seq_width = len(dp.columnsName_to_index)

    print("-" * 50, "\ndp.columns_max\n", dp.columns_max, "\n")
    print("-" * 50, "\ndp.columns_numb\n", dp.columns_numb, "\n")
    print("-" * 50, "\ndp.columnsName_to_index\n", dp.columnsName_to_index, "\n")

    config = code0.ModelParamsConfig(dp)
    eval_config = code0.ModelParamsConfig(dp)

    if dp.dataSetType=='kdd':
        config.num_steps = 2000
    else:
        config.num_steps = aux.get_num_step(dataset)

    eval_config.num_steps = config.num_steps
    eval_config.batch_size = 2

    config.skill_num = skill_num
    eval_config.skill_num = config.skill_num

    auc_train,r2_train,rmse_train,auc_test,r2_test,rmse_test = aux.defineResult()
    CVname = auc_test.columns
    size = len(tuple_data)

    # write all the records to log file
    aux.printConfigration(config=config, dp=dp, train_numb=int(size * 0.8), test_numb=int(size * 0.2))
    aux.logwrite(["==> model_continues_columns\n" + ','.join(dp.model_continues_columns)],
                            dp,True)
    aux.logwrite(["==> model_category_columns\n" + ','.join(dp.model_category_columns)],
                            dp,True)
    str_cross_columns_list = ['-'.join(i) for i in dp.model_cross_columns]
    str_cross_columns = ','.join(str_cross_columns_list)
    aux.logwrite(["==> model_cross_columns\n" + str_cross_columns], dp,True)

    for index, cv_num_name in enumerate(CVname):
        aux.logwrite(["\nCross-validation: \t" + str(index + 1) + "/5"], dp,prt=True)
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
                rmse, auc, r2 = code3.run_epoch(session, m, train_tuple_rows, m.train_op, verbose=True)
                train_result = "\n==> %s cross-valuation: Train Epoch: %d\tLearning rate: %.3f\t rmse: %.3f \t auc: %.3f \t r2: %.3f" % (
                    cv_num_name, i + 1, rt, rmse, auc, r2)
                print(train_result)
                auc_train.loc[i, cv_num_name] = auc
                rmse_train.loc[i, cv_num_name] = rmse
                r2_train.loc[i, cv_num_name] = r2
                aux.logwrite(train_result, dp,False)

                display = 5
                if ((i + 1) % display == 0):
                    print("-" * 80)
                    rmse, auc, r2 = code3.run_epoch(session, mtest, test_tuple_rows, tf.no_op())
                    test_result = "\n==> %s cross-valuation: Test Epoch: %d \t rmse: %.3f \t auc: %.3f \t r2: %.3f" % (
                        cv_num_name, (i + 1) / display, rmse, auc, r2)
                    print(test_result)
                    print("=" * 80)
                    auc_test.loc[(i + 1) / display - 1, cv_num_name] = auc
                    rmse_test.loc[(i + 1) / display - 1, cv_num_name] = rmse
                    r2_test.loc[(i + 1) / display - 1, cv_num_name] = r2
                    aux.logwrite(test_result, dp,False)
    print("==> Finsih! whole process, save result and print\t" + dp.currentTime)

    try:
        mean_result = pd.DataFrame({"AUC": list(auc_test.mean(1)), "RMSE": list(rmse_test.mean(1)),
                           "R2": list(r2_test.mean(1))})
        print(mean_result)
        aux.saveResult(dp,auc_train,rmse_train,r2_train,auc_test,rmse_test,r2_test,mean_result)
    except:
        print("except during save result")
        pass

if __name__ == "__main__":
    tf.app.run()

''' Unless stated otherwise, all software is provided free of charge. 
As well, all software is provided on an "as is" basis without warranty 
of any kind, express or implied. Under no circumstances and under no legal 
theory, whether in tort, contract, or otherwise, shall Liang Zhang be liable 
to you or to any other person for any indirect, special, incidental, 
or consequential damages of any character including, without limitation, 
damages for loss of goodwill, work stoppage, computer failure or malfunction, 
or for any and all other damages or losses. If you do not agree with these terms, 
then you are advised to not use the software.'''

import math
import os
import pyprind
import random
import sys

import numpy as np
import pandas as pd

import code0_parameter as code0
import uril_assistment2009
import uril_cmu_statistic
import uril_tools as aux



def create_label_and_delete_last_one(dp):
    dataFileName = os.path.dirname(dp.csv_file_name) + "/dataset_" + str(dp.dataSetSize) + ".csv"
    labelFileName = os.path.dirname(dp.csv_file_name) + "/labels_" + str(dp.dataSetSize) + ".csv"

    if os.path.exists(dataFileName) and os.path.exists(labelFileName):
        dataset = pd.read_csv(dataFileName)
        labels = pd.read_csv(labelFileName)
        print('==> ', dataFileName, " exists,load directly")
        print('==> ', labelFileName, " exists,load directly")
        return dataset, labels

    if dp.dataSetType == "assistment2009":
        data = uril_assistment2009.read_asssistment2009_data_from_csv(dp)
    elif dp.dataSetType == "kdd":
        data = uril_kdd.read_kdd_data_from_csv(dp)
    elif dp.dataSetType == "cmu_stat_f2011":
        data = uril_cmu_statistic.read_data_from_csv()

    userID_Quest_number_matrix = aux.getUserQuesNumList(data['user_id'])  # user_id: number of questions
    print("==> creat skill_id+label, last record of every user is deleted")
    print("==> delete user whose problem number is less than 2")
    row_size = len(data);
    index = 0
    kindex = 0
    dataset = pd.DataFrame()
    labels = pd.DataFrame()

    bar = pyprind.ProgPercent(row_size, stream=sys.stdout)
    while (index < row_size):
        id_number = userID_Quest_number_matrix[kindex, 1]
        if id_number > 2:
            dataTemp = data.loc[index:index + id_number - 2]
            labeTemp = pd.DataFrame({'user_id': int(data.loc[index, 'user_id']),
                                     'label_skill_id': data.loc[index + 1:index + id_number - 1, "skill_id"],
                                     'label_correct': data.loc[index + 1:index + id_number - 1, "correct"]})
            assert len(dataTemp) == len(labeTemp)
            dataset = dataset.append(dataTemp)
            labels = labels.append(labeTemp)
            del dataTemp, labeTemp
        bar.update(id_number)
        index += id_number
        kindex += 1
    dataset = dataset.reset_index(drop=True)
    labels = labels.reset_index(drop=True)

    if os.path.exists(dataFileName): os.remove(dataFileName)
    if os.path.exists(labelFileName): os.remove(labelFileName)
    dataset.to_csv(dataFileName, index=False)
    labels.to_csv(labelFileName, index=False)
    print("==> save ", dataFileName)
    print("==> save ", labelFileName)

    assert len(dataset) == len(labels), "dateset size\t" + str(len(dataset)) + "\tlabels size\t" + str(len(labels))
    return dataset, labels


def convert_data_labels_to_tuples(dataset, labels):
    index = 0
    kindex = 0
    tuple_rows = []
    userID_Quest_number_matrix = aux.getUserQuesNumList(dataset['user_id'])
    print("==> convert data and labels to tuples")
    # tuple formate
    # 0: user_id
    # 1: record_numb
    # 2: data
    # 3: Target_Id
    # 4: correctness
    dataset_size = len(dataset)
    bar = pyprind.ProgPercent(dataset_size, stream=sys.stdout)
    while index < dataset_size:
        numb = int(userID_Quest_number_matrix[kindex, 1])
        assert int(userID_Quest_number_matrix[kindex, 0]) == int(dataset.loc[index, "user_id"])
        tup = (dataset.loc[index, "user_id"], numb, dataset.iloc[index:index + numb],
               list(labels.loc[index:index + numb - 1, "label_skill_id"]),
               # the input is a list but not pd.DataFrame, don't need to reset the index.
               list(labels.loc[index:index + numb - 1, "label_correct"]))
        # pd.DataFrame, loc and iloc cut differentsize!
        tuple_rows.append(tup)
        index += numb
        kindex += 1
        bar.update(numb)
    random.shuffle(tuple_rows)
    return tuple_rows


def get_columns_info(dataset):
    columns_max = {}
    columns_numb = {}
    columnsName_to_index = {}
    for i, column_name in enumerate(dataset.columns):
        try:
            columns_max[column_name] = max(dataset[column_name])
            columns_numb[column_name] = len(dataset[column_name].unique())
            columnsName_to_index[column_name] = i
        except:
            print(dataset.columns)
            print(np.shape(dataset))
            print(dataset[column_name])
            raise ValueError(column_name)

    return columns_max, columns_numb, columnsName_to_index


def add_cross_feature_to_dataset(dataset, dp):
    if len(dp.dataset_columns_for_cross_feature) == 0:
        print("==> no need to add cross feature to dataset")
        return dataset
    else:
        print("==> add cross feature to dataset")
        columns_max, columns_numb, _ = get_columns_info(dataset)
        d_size = len(dataset)
        for item in dp.dataset_columns_for_cross_feature:
            print("==> add", aux.connectStringfromList(item))
            temp = []
            for i in pyprind.prog_percent(range(d_size), stream=sys.stdout, title=item):
                if len(item) == 2:

                    value = dataset.loc[i, item[0]] + dataset.loc[i, item[1]] * (columns_max[item[0]] + 1)
                    #print(" dataset.loc[i, item[0]]\t", dataset.loc[i, item[0]], "\tdataset.loc[i, item[1]]\t",
                    #      dataset.loc[i, item[1]], "\t(columns_max[item[0]] + 1)\t",(columns_max[item[0]] + 1),
                    #      "\tvalue\t", value)
                elif len(item) == 3:
                    value = dataset.loc[i, item[0]] + dataset.loc[i, item[1]] * (columns_max[item[0]] + 1) + \
                            dataset.loc[i, item[2]] * (columns_max[item[0]] + 1) * (columns_max[item[1]] + 1)
                else:
                    raise ValueError('cross features only support 3 at most')
                temp.append(value)
            dataset[aux.connectStringfromList(item)] = temp
        return dataset


# only for assistment 2009 and 2014 data
def normalization_continues_data(data):
    print('==> normalize continues data')
    columns_name_list = ["attempt_count", "time", "hint_count"]
    data = data.reset_index(drop=True)

    size = len(data)
    for column_name in columns_name_list:
        if column_name == "time":
            bins = [-1, 60, 300, 1200, 3600, 60000000]
            data[column_name] = pd.cut(data[column_name], bins, labels=False)
            tmpList = []

            for i in pyprind.prog_percent(range(size), stream=sys.stdout, title=column_name):
                try:
                    tmp = int(data.loc[i, column_name])
                except:
                    tmp = 0
                    # raise ValueError(str(data.loc[i, column_name])+"_"+str(i))
                tmpList.append(math.log((tmp + 2), 6))
            data['time_normal'] = tmpList
        elif column_name == "attempt_count":
            bins = [-10, 1, 20, 100, 40000]
            data[column_name] = pd.cut(data[column_name], bins, labels=False)
            data[column_name] += 1
            tmpList = []

            for i in pyprind.prog_percent(range(size), stream=sys.stdout, title=column_name):
                # print ("attempt_count\t",str(i))
                tmp = int(data.loc[i, column_name])
                tmpList.append(math.log((tmp + 1), 5))
            data['attempt_count_normal'] = tmpList
        elif column_name == "hint_count":
            bins = [-1, 0, 2, 4, 3000]
            data[column_name] = pd.cut(data[column_name], bins, labels=False)
            data[column_name] += 1
            tmpList = []
            for i in pyprind.prog_percent(range(size), stream=sys.stdout, title=column_name):
                try:
                    tmp = int(data.loc[i, column_name])
                except:
                    tmp = 0
                tmpList.append(math.log((tmp + 1), 5))
            data['hint_count_normal'] = tmpList
        else:
            raise ValueError("check your continus_columns parameter!")
    return data


def load_data(dp):
    if len(dp.dataset_columns_for_cross_feature) == 0:
        dataFileName = os.path.dirname(dp.csv_file_name) + "/dataset_" + str(dp.dataSetSize) + ".csv"
    else:
        tmp = aux.connectStringfromList(dp.convertCrossCoumnsToNameList())
        dataFileName = os.path.dirname(dp.csv_file_name) + '/' + tmp + "_" + str(dp.dataSetSize) + '_' + ".csv"
    labelFileName = os.path.dirname(dp.csv_file_name) + "/labels_" + str(dp.dataSetSize) + ".csv"

    if os.path.exists(dataFileName) and os.path.exists(labelFileName):
        data = pd.read_csv(dataFileName)
        labels = pd.read_csv(labelFileName)
        return data, labels
    else:
        data, labels = create_label_and_delete_last_one(dp)
        dataset_with_crossFeatures = add_cross_feature_to_dataset(data, dp)
        dataset_with_crossFeatures.to_csv(dataFileName, index=False)
        print("==> save ", dataFileName)
        return dataset_with_crossFeatures, labels


if __name__ == "__main__":
    dp = code0.DatasetParameter()
    load_data(dp)

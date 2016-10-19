import pandas as pd
import numpy as np
import sys, pyprind
import aux
import code1_data as code1
import code0_parameter as code0
import os

def read_asssistment2014_data_from_csv(dp):
    if os.path.exists(dp.processedFileName):
        print ("==> read ",dp.processedFileName)
        data = pd.read_csv(dp.processedFileName)
        secNumber, userNumber, skillNumber = aux.stastic_SecNumber_UserNumber_SkillNumber(data, dp)
        return data

    print("==> read assistment2014 from csv")
    try:
        data = pd.read_csv(dp.csv_file_name)
    except:
        raise NameError("can't load " + dp.csv_file_name + " pleace check your file")
    print(data.columns)
    data.rename(columns={'first_response_time': 'time'}, inplace=True)
    data = data[dp.filtedColumnNameList].fillna(0)
    if dp.dataSetSize == "small":
        data = data[0:50000]
    print("==> run ", dp.dataSetSize, " dataset")

    data = data[data['original'] == 1]
    data = data.reset_index(drop=True)
    print("==> consider original==1, data shape\t", data.shape)

    # time process
    print("==> transfer time unit: millsecond to second")
    tempTimeList = list(data['time'])
    newTimeList = [int(x / 1000) for x in tempTimeList]
    data['time'] = newTimeList
    del newTimeList, tempTimeList

    # correct process
    print("==> change correct to 1 or 0")
    tempCorrectList = list(data['correct'])
    for i in range(len(data)):
        if tempCorrectList[i] != 1 and tempCorrectList[i] != 0:
            if tempCorrectList[i] <= 0.5:
                tempCorrectList[i] = 0
            else:
                tempCorrectList[i] = 1
    data['correct'] = tempCorrectList
    del tempCorrectList

    data = code1.normalization_continues_data(data)

    # project sequence_id to skill_id
    # in ass_skill_file_name file, all sequence_id has related skill_id
    # not all sequence_id in csv_file_name are included in ass_skill_file_name
    print("==> project sequence_id to skill_id")
    try:
        skill_data = pd.read_csv(dp.ass_skill_file_name)
    except:
        raise "can't load " + dp.ass_skill_file_name + " pleace check your file"
    skill_data_size = len(skill_data)
    dict_sequence_to_skill_dict = {}
    skill_data_sequence_list = list(skill_data['sequence_id'])
    skill_data_skill_list = list(skill_data['skill_id'])
    assert len(skill_data_sequence_list) == len(skill_data_skill_list)
    del skill_data

    for i in range(skill_data_size):
        dict_sequence_to_skill_dict[skill_data_sequence_list[i]] = skill_data_skill_list[i]
    print("==> skill number without sequence id projection\t", len(dict_sequence_to_skill_dict))

    # some sequence_id in csv_file_name has no skill_id, use sequence_id directly
    skill_data_sequence_set = np.unique(skill_data_sequence_list)
    data_sequence_set = set(data['sequence_id'])
    for i in data_sequence_set:
        if i not in skill_data_sequence_set:
            dict_sequence_to_skill_dict[i] = i
    print("==> skill number include non-projected sequence_id\t", len(dict_sequence_to_skill_dict))

    tempList = []
    for i in pyprind.prog_percent(range(len(data)), stream=sys.stdout,title='add skill_id column to data_set'):
        tempList.append(dict_sequence_to_skill_dict[int(data[i:i + 1]['sequence_id'])])

    assert len(tempList) == len(data)
    data['skill_id'] = tempList

    skill_set, skill_dict = aux.create_column_dict_and_set(data,'skill_id',dp)
    problem_set, problem_dict = aux.create_column_dict_and_set(data,'problem_id',dp)
    for i in pyprind.prog_percent(range(len(data)),stream=sys.stdout,title='reorder skill_id and problem_id to reduce max number'):
        data.loc[i, "skill_id"] = skill_dict[data.loc[i, "skill_id"]]
        data.loc[i, "problem_id"] = problem_dict[data.loc[i, "problem_id"]]

    if os.path.exists(dp.processedFileName):
        os.remove(dp.processedFileName)
        print ('==> remove ',dp.processedFileName)
    data.to_csv(dp.processedFileName, index=False)
    print("==> dataset column name\n", data.columns)
    print("==> dataset shape\t", data.shape)
    print("==> save file to ", dp.processedFileName)
    secNumber, userNumber, skillNumber = aux.stastic_SecNumber_UserNumber_SkillNumber(data, dp)
    return data


if __name__ == "__main__":
    dp = code0.DatasetParameter()
    read_asssistment2014_data_from_csv(dp)

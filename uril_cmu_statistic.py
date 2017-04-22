import pandas as pd
import numpy as np
import sys, pyprind
import uril_tools as aux
import code1_data as code1
import code0_parameter as code0
import os
import pyprind as pp
import matplotlib.pyplot as plt
import datetime


def read_data_from_csv():
    processedFileName = './data/cmu_stat_f2011/processded_data.csv'
    raw_data_txt = "./data/cmu_stat_f2011/cmu.txt"

    if os.path.exists(processedFileName):
        data = pd.read_csv(processedFileName)
        print("==> read ", processedFileName, " directly")

    else:
        if os.path.exists(raw_data_txt):
            data = pd.read_csv(raw_data_txt, sep=" ", delimiter='\t')
            print(data.columns)
            data.rename(columns={'Duration (sec)': 'time', 'Outcome': 'correct',
                         'KC (F2011)': 'skill_id', 'Problem Name': 'problem_id', 'Step Name': 'step_id',
                         'Anon Student Id': 'user_id',"Student Response Type":"first_action",'Attempt At Step':"attempt_level"}, inplace=True)

            data = data.fillna(-1)

            filer_data = data[code0.DatasetParameter('cmu_stat_f2011').filtedColumnNameList]
            filer_data = filer_data[(filer_data['correct'] != -1) & (filer_data['correct'] != 'HINT') & (
                filer_data['skill_id'] != '-1') & (filer_data['time'] != '.')]

            filer_data['correct'].replace({'CORRECT': 1, 'INCORRECT': 0}, inplace=True)

            # change str to integar
            for feature in ['skill_id', 'step_id', 'problem_id', 'user_id', 'Level (Unit)', 'Level (Module)','first_action','attempt_level']:
                print("==> BEGIN ", feature)
                temp_set = set(list(filer_data[feature]))
                temp_dict = {key: value+1 for value, key in enumerate(temp_set)}
                filer_data[feature].replace(temp_dict, inplace=True)
                print("==> END   ", feature)

            #print ("==> first_action",set(filer_data['first_action']))
            #print ("==> attempt_level",set(filer_data['attempt_level']))
            data = attempt_process(filer_data)
            data = time_basic_process(data)
            data = time_add_level_process(data)
            data.to_csv(processedFileName, index=False)

        else:
            raise ('No data file exists!')
    return data

def attempt_process(data):
    temp_list = list(data['attempt_level'])
    new_list = []

    for i in range(len(temp_list)):
        if temp_list[i]==1:
            new_list.append(0)
        elif temp_list[i]<=5 and temp_list[i]>1:
            new_list.append(1)
        elif temp_list[i]>5:
            new_list.append(2)
        else:
            new_list.append(3)
    data['attempt_level'] = new_list
    return data

def test_data():
    data = read_data_from_csv()

    k1 = []
    k2 = []
    for item in data.columns:
        num = len(set(data[item]))
        print("****%10d--%s" % (num, item))
        k2.append(item)
        if num < 10:
            print("-" * 10, item, "-" * 10, "\n", np.unique(data[item]), "\n", "--" * 15)

    print('--' * 30)
    print("more than 1 elements\n", k2)

    print(np.shape(data))


def time_basic_process(data):
    # -1-transfer time to 'integar' from 'str'
    # -2-remove outlier records
    old_time_list = list(data['time'])
    new_time_list = []
    for i in old_time_list:
        kp = int(float(i))
        if kp > 150: kp = 150
        new_time_list.append(kp)
    data['time'] = new_time_list

    # -3-transfer to z-score
    time_z_level = 'skill_id'
    print('==> preprocerss time to z-score based on ', time_z_level)
    time_z_id_set = np.unique(data[time_z_level])
    std_dict = {}
    mean_dict = {}
    for itme_id in pp.prog_percent(time_z_id_set, stream=sys.stdout, title='==> extract mean and std of time'):
        temp_data = data[data[time_z_level] == itme_id]
        temp_list = list(temp_data['time'])
        # print ('-- problem_id ',problem_id,' -- ',len(temp_list),' --')
        std_dict[itme_id] = np.std(temp_list, axis=0)
        mean_dict[itme_id] = np.mean(temp_list, axis=0)

    assert len(std_dict) == len(mean_dict)

    data = data.reset_index(drop=True)

    for id in pp.prog_percent(range(len(data)), stream=sys.stdout, title='==> cast time to z-score'):
        data.loc[id, 'time'] = (data.loc[id, 'time'] - mean_dict[data.loc[id, time_z_level]]) / (
            std_dict[data.loc[id, time_z_level]] * 1.0)

    return data


def temp(data):
    # -1-transfer time to 'integar' from 'str'
    old_time_list = list(data['time'])
    new_time_list = []
    for i in old_time_list:
        new_time_list.append(int(float(i)))
    data['time'] = new_time_list

    plt.hist(new_time_list, bins=np.arange(min(new_time_list), max(new_time_list), ))
    plt.show()


def time_add_level_process(data):
    time_interval = 0.025
    boundary_list = [0.5, 0.7]
    data = data.reset_index(drop=True)
    bins = np.arange(min(data['time']), max(data['time']), time_interval * 2)

    correct_mean_list = []
    correct_std_list = []
    correct_num_list = []
    for item_index in pp.prog_percent(range(len(bins)), stream=sys.stdout, title='==> get correctness'):
        up_bin = bins[item_index] + time_interval
        down_bin = bins[item_index] - time_interval

        temp_data = data[(data['time'] >= down_bin) & (data['time'] < up_bin)]
        temp_correct_list = list(temp_data['correct'])

        """
        if up_bin<=-1:
            print ("---"*20)
            print ("*\t",down_bin)
            print ("*\t",up_bin)
            print (temp_correct_list)
            #print (temp_data)
            print ("---"*20)
        """

        correct_num_list.append(len(temp_correct_list))
        if (len(temp_correct_list) != 0):
            if np.mean(temp_correct_list, axis=0) > 1:
                print("******\t", np.mean(temp_correct_list, axis=0), "\t", temp_correct_list)
            correct_mean_list.append(np.mean(temp_correct_list, axis=0))
            correct_std_list.append(np.std(temp_correct_list, axis=0))
        else:
            correct_mean_list.append(0)
            correct_std_list.append(0)

    # plot the relationship
    fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True)
    ax = axs[0]
    ax.plot(bins, correct_mean_list, "r.")
    ax.set_title('correctness')

    for nmber in boundary_list:
        ax.axhline(y=nmber, xmin=0, xmax=1, c="red", linewidth=0.5, zorder=0)

    ax = axs[1]
    ax.plot(bins, correct_num_list, "b--")
    ax.set_title("time z score distribution")

    ax.set_xlim([-2, 4])
    plt.savefig('./result/cmu_stat_f2011/time_distribution_correctness_' + str(
        datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")) + '.png')
    #plt.show()

    # add a colum according to correctness boundary
    time_level_list = []
    temp_list = list(data['time'])
    bd = [-1.2, -0.7, 0.75]

    # 0 ~        time < -1.2
    # 1 ~ -1.2 < time < -0.7
    # 2 ~ -0.7 < time < 0.75
    # 3 ~ 0.75 < time
    for idx in range(len(temp_list)):
        if temp_list[idx] <= bd[0]:
            time_level_list.append(0)
        elif (bd[0] < temp_list[idx] and temp_list[idx] <= bd[1]):
            time_level_list.append(1)
        elif (bd[1] < temp_list[idx] and temp_list[idx] <= bd[2]):
            time_level_list.append(2)
        elif (temp_list[idx] > bd[2]):
            time_level_list.append(3)
        else:
            raise Exception("Error in time division")
    print("==> add time_level")
    data['time_level'] = time_level_list
    return data

def read_data_from_csv2():
    processedFileName = './data/cmu_stat_f2011/test_data.csv'
    raw_data_txt = "./data/cmu_stat_f2011/cmu.txt"

    if os.path.exists(processedFileName):
        data = pd.read_csv(processedFileName)
        print("==> read ", processedFileName, " directly")

    else:
        if os.path.exists(raw_data_txt):
            data = pd.read_csv(raw_data_txt, sep=" ", delimiter='\t')
            print(data.columns)
            data.rename(columns={'Duration (sec)': 'time', 'Outcome': 'correct',
                         'KC (F2011)': 'skill_id', 'Problem Name': 'problem_id', 'Step Name': 'step_id',
                         'Anon Student Id': 'user_id',"Student Response Type":"first_action",'Attempt At Step':"attempt_level"}, inplace=True)

            data = data.fillna(-1)

            filer_data = data[code0.DatasetParameter('cmu_stat_f2011').filtedColumnNameList]
            filer_data = filer_data[(filer_data['correct'] != -1) & (filer_data['correct'] != 'HINT') & (
                filer_data['skill_id'] != '-1') & (filer_data['time'] != '.')]

            filer_data['correct'].replace({'CORRECT': 1, 'INCORRECT': 0}, inplace=True)

            # change str to integar
            for feature in ['skill_id', 'step_id', 'problem_id', 'user_id', 'Level (Unit)', 'Level (Module)','first_action','attempt_level']:
                print("==> BEGIN ", feature)
                temp_set = set(list(filer_data[feature]))
                temp_dict = {key: value+1 for value, key in enumerate(temp_set)}
                filer_data[feature].replace(temp_dict, inplace=True)
                print("==> END   ", feature)

            print ("==> first_action",set(filer_data['first_action']))
            print ("==> attempt_level",set(filer_data['attempt_level']))
            data.to_csv(processedFileName,index=False)
        else:
            raise ('No data file exists!')
    return data

def test():
    processedFileName = './data/cmu_stat_f2011/test_data.csv'
    data = pd.read_csv(processedFileName)
    plt.hist(list(data['attempt_level']),np.arange(min(data['attempt_level']), max(data['attempt_level']), 1))
    plt.show()

if __name__ == '__main__':
    data = read_data_from_csv()


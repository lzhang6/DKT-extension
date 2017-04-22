import pandas as pd
import uril_tools as aux
import code1_data as code1
import code0_parameter as code0
import pyprind, os
import sys
import uril_connectUser
import numpy as np
import pyprind as pp
import pylab as pl
import datetime
import matplotlib.pyplot as plt


def read_asssistment2009_data_from_csv(dp):
    # read process file directly if exists
    if os.path.exists(dp.processedFileName):
        print("==> read ", dp.processedFileName)
        data = pd.read_csv(dp.processedFileName)
        print(aux.stastic_SecNumber_UserNumber_SkillNumber(data, dp))
        return data

    # processfile not exist, load connect data and process it
    if os.path.exists(dp.connect_file_name):
        print("==> read ", dp.connect_file_name)
        data = pd.read_csv(dp.connect_file_name)
    else:  # read raw data and connect
        try:
            data = pd.read_csv(dp.csv_file_name, encoding='latin-1', error_bad_lines=False, index_col=False)
            if dp.csv_file_name == "./data/assistment2009/skill_builder_data_corrected.csv":
                data = data.loc[:338000]
            elif dp.csv_file_name == "./data/assistment2009/skill_builder_data.csv":
                data = data.loc[:450000]
            else:
                pass
            print("==> read ", dp.csv_file_name)
        except:
            raise NameError("can't load " + dp.csv_file_name + " pleace check your file")
        print('==> columns names\t', data.columns)

        data.rename(columns={'ms_first_response': 'time', 'hint_count': '_hint_count', 'hint_total': 'hint_count'},
                    inplace=True)

        data = data[dp.filtedColumnNameList].fillna(0)
        if dp.dataSetSize == "small":
            data = data[0:50000]
        print("==> run ", dp.dataSetSize, " dataset")

        data = data[data['original'] == 1]
        data = data.reset_index(drop=True)
        print("==> consider original==1, data shape\t", data.shape)

        data = uril_connectUser.connectUser(data, dp.connect_file_name)
        print("==> save ", dp.connect_file_name)

    ### data process
    # correct process
    print("==> remove records whose correct is not 1 or 0")
    data = data[(data['correct'] == 1) | (data['correct'] == 0)]
    data = data.reset_index(drop=True)

    # time process
    data = time_basic_process(data)
    data = time_add_level_process(data)
    data = data.reset_index(drop=True)

    # attempt process
    data = attempt_add_level_process(data)

    print("==> dataset column name\n", data.columns)
    print("==> dataset shape\t", data.shape)

    data.to_csv(dp.processedFileName, index=False)
    print("==> save file to ", dp.processedFileName)

    aux.stastic_SecNumber_UserNumber_SkillNumber(data, dp)
    return data


def time_basic_process(data):
    # -1-transfer to second unit
    print("==> transfer time unit: millsecond to second")
    tempTimeList = list(data['time'])
    newTimeList = [int(x / 1000) for x in tempTimeList]
    data['time'] = newTimeList
    del newTimeList, tempTimeList

    # -2-remove outlier records
    print('==> delete outlier of time feature')
    print('==> length before delete\t', len(data))
    data = data[(data['time'] <= code0.DatasetParameter().time_threshold) & (data['time'] > 0)]
    print('==> length after delete\t', len(data))

    # -3-transfer to z-score
    time_z_level = code0.DatasetParameter().time_z_level
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

    data = data.fillna(0)

    """
    plt.hist(list(data['time']), bins=np.arange(min(data['time']), max(data['time']), code0.DatasetParameter().time_interval*2))
    plt.title("time z score distribution")
    plt.savefig('./result/assistment2009/time_distribution' + str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")) + '.png')
    """
    return data


def time_add_level_process(data):
    data = data.reset_index(drop=True)
    bins = np.arange(min(data['time']), max(data['time']), code0.DatasetParameter().time_interval * 2)
    correct_mean_list = []
    correct_std_list = []
    correct_num_list = []
    for item_index in pp.prog_percent(range(len(bins)), stream=sys.stdout, title='==> get correctness'):
        up_bin = bins[item_index] + code0.DatasetParameter().time_interval
        down_bin = bins[item_index] - code0.DatasetParameter().time_interval

        temp_data = data[data['time'] >= down_bin]
        temp_data = temp_data[temp_data['time'] < up_bin]

        temp_correct_list = list(temp_data['correct'])
        correct_num_list.append(len(temp_correct_list))
        if (len(temp_correct_list) != 0):
            correct_mean_list.append(np.mean(temp_correct_list, axis=0))
            correct_std_list.append(np.std(temp_correct_list, axis=0))
        else:
            correct_mean_list.append(0)
            correct_std_list.append(0)

    # plot the relationship
    fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True)
    ax = axs[0]
    ax.plot(bins, correct_mean_list)
    ax.set_title('correctness')
    boundary_list = code0.DatasetParameter().correct_boundary_list
    for nmber in boundary_list:
        ax.axhline(y=nmber, xmin=0, xmax=1, c="red", linewidth=0.5, zorder=0)

    ax = axs[1]
    ax.plot(bins, correct_num_list)
    ax.set_title("time z score distribution")

    ax.set_xlim([-2, 4])
    plt.savefig('./result/assistment2009/time_distribution_correctness_' + str(
        datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")) + '.png')
    # plt.show()

    # add a colum according to correctness boundary
    time_level_list = []
    temp_list = list(data['time'])
    bd = code0.DatasetParameter().time_boundary_list
    # 0 ~        time <-0.8
    # 1 ~ -0.8 < time < -0.6
    # 2 ~ -0.6 < time < 0
    # 3 ~    0 < time
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

    data['time_level'] = time_level_list
    return data


def attempt_add_level_process(data):
    """
    based on correctness and attempt relationship
    0 - attempt: 0 - 0
    1 - attempt: 1 - 81.7%
    2 - attempt: 2 -
    3 - attempt: 0 - 0
    """
    temp_list = []

    for item in pp.prog_percent(list(data['attempt_count']), stream=sys.stdout, title='==> cast attmept to attempt_level'):
        if item == 0:
            temp = 0
        elif item == 1:
            temp = 1
        else:
            temp = 2

        temp_list.append(temp)
    data['attempt_level'] = temp_list
    return data


def attempt_and_hint_process(data):
    print('==> remove records whose attempt_account is more than 15')
    data = data[data['attempt_count'] <= 15]
    data = data.reset_index(drop=True)

    problem_list = np.unique(data['problem_id'])
    attempt_dict = {}
    hint_dict = {}
    attempt_list = []
    hint_list = []
    for idx in pp.prog_percent(range(len(problem_list)), stream=sys.stdout,
                               title='==> get attmept and hint max value at problem level'):
        temp_data = data[data['problem_id'] == problem_list[idx]]
        attempt_dict[problem_list[idx]] = max(temp_data['attempt_count'])
        attempt_list.append(max(temp_data['attempt_count']))
        hint_dict[problem_list[idx]] = max(temp_data['hint_count'])
        hint_list.append(max(temp_data['hint_count']))

    fig, axs = plt.subplots(nrows=2, ncols=1, sharex=False)
    ax = axs[0]
    ax.hist(attempt_list, bins=np.arange(0, 16, 1))
    ax.set_title('max attempt distribution')
    ax.set_xlabel("attempt(max)")
    ax.set_ylabel("number")

    ax = axs[1]
    ax.hist(hint_list)
    ax.set_title("max hint distribution")
    ax.set_xlabel("hint(max)")
    ax.set_ylabel("number")

    plt.savefig(
        './result/assistment2009/attempt_hint_number_' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M") + '.png')

    for idx in pp.prog_percent(range(len(data)), stream=sys.stdout,
                               title='==> cast attempt  count and hint count to value/max'):
        if attempt_dict[data.loc[idx, 'problem_id']] == 0:
            data.loc[idx, 'attempt_count_level'] = -1
        else:
            data.loc[idx, 'attempt_count_level'] = data.loc[idx, 'attempt_count'] / (
                attempt_dict[data.loc[idx, 'problem_id']] * 1.0)

        if hint_dict[data.loc[idx, 'problem_id']] == 0:
            data.loc[idx, 'hint_count_level'] = -1
        else:
            data.loc[idx, 'hint_count_level'] = data.loc[idx, 'hint_count'] / (
                hint_dict[data.loc[idx, 'problem_id']] * 1.0)

    return data


def attemp_hint_and_correctness_analysis(data):
    data = data.reset_index(drop=True)
    bins = np.concatenate([[-1], np.arange(0.0, 1.1, 0.1)])

    for attri in ['hint_count_level', 'attempt_count_level']:
        correct_mean_list = []
        correct_std_list = []
        correct_num_list = []

        for item_index in pp.prog_percent(range(len(bins)), stream=sys.stdout,
                                          title='==> get correctness according to ' + attri):
            up_bin = bins[item_index] + 0.05
            down_bin = bins[item_index] - 0.05

            temp_data = data[(data[attri] >= down_bin) & (data[attri] < up_bin)]
            temp_correct_list = list(temp_data['correct'])
            correct_num_list.append(len(temp_correct_list))

            if (len(temp_correct_list) != 0):
                correct_mean_list.append(np.mean(temp_correct_list, axis=0))
                correct_std_list.append(np.std(temp_correct_list, axis=0))
            else:
                correct_mean_list.append(0)
                correct_std_list.append(0)

        fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True)
        ax = axs[0]
        ax.plot(bins, correct_mean_list)
        ax.set_title('correctness ' + attri)

        boundary_list = code0.DatasetParameter().correct_boundary_list
        for nmber in boundary_list:
            ax.axhline(y=nmber, xmin=0, xmax=1, c="red", linewidth=0.5, zorder=0)

        ax = axs[1]
        ax.plot(bins, correct_num_list)
        ax.set_title(attri + " number distribution")
        ax.set_xlim([-1.1, 1.1])
        plt.savefig('./result/assistment2009/' + attri + '_correctness_' + str(
            datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")) + '.png')


def attempt_correct_analysis(data):
    data = data[data['attempt_count'] <= code0.DatasetParameter().attemp_max]
    u, c = aux.counter(list(data['attempt_count']))

    atempt_list = np.arange(code0.DatasetParameter().attemp_max + 1)
    correct_num_list = []
    for item in atempt_list:
        temp_data = data[(data['attempt_count'] == item)]
        if len(temp_data) != 0:
            correct_num_list.append(sum(temp_data['correct']) * 1.0 / len(temp_data))
        else:
            correct_num_list.append(0)
    print(u, "\n", c)
    print(atempt_list, "\n", correct_num_list)

    for a in correct_num_list:
        print("%.3f" % a)


if __name__ == "__main__":
    dp = code0.DatasetParameter()
    data = read_asssistment2009_data_from_csv(dp)
    attempt_correct_analysis(data)
    """
    data = pd.read_csv("./data/assistment2009/time_connect_data.csv")
    data = data[:30000]
    data  = attempt_and_hint_process(data)
    attemp_hint_and_correctness_analysis(data)

    data = pd.read_csv(dp.connect_file_name)
    data = data[:10000]
    print(data[:10])
    data = attempt_and_hint_process(data)
    print(data[:10])

    data = pd.read_csv(dp.connect_file_name)
    data = data[:10000]
    print(data[:10])
    data = time_process(data)
    print(data[:10])
    data.to_csv('./data/assistment2009/kkk.csv')
    data = time_correctness_relation_analysis(data)
    print(data[:10])
    data = pd.read_csv('./data/assistment2009/kkk.csv')
    """

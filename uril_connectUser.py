import numpy as np
import pandas as pd
import pyprind as pp
import sys

import numpy as np
import pandas as pd
import pyprind as pp
import sys
import uril_tools as aux
import code0_parameter as code0


def counter(a):
    a = list(a)
    unique, counts = np.unique(a, return_counts=True)
    return unique, counts


def getUserQuesNumIndexList(dataList):
    a = list(dataList)
    target = np.empty((0, 3))
    size = len(a)
    temp = [a[0], 1, 0]
    for i in range(1, size):
        if a[i] == a[i - 1]:
            temp[1] += 1
        else:
            target = np.vstack((target, temp))
            temp = [a[i], 1, i]
    return np.vstack((target, temp))


def connectUser(data, connected_file_name):
    print("==> load data successful")
    u, c = counter(data['user_id'])
    # UserNumberDict = dict(zip(u, c))

    userQuesNumIndexList = getUserQuesNumIndexList(data['user_id'])
    newdata = pd.DataFrame()

    print('==> begin concatenate dataset')
    for i in pp.prog_percent(range(len(u)), stream=sys.stdout):
        for k in range(len(userQuesNumIndexList)):
            if userQuesNumIndexList[k, 0] == u[i]:
                temp = data.iloc[
                       int(userQuesNumIndexList[k, 2]):int(userQuesNumIndexList[k, 2] + userQuesNumIndexList[k, 1])]
                newdata = newdata.append(temp)

    newdata.reset_index(drop=True)
    newdata.to_csv(connected_file_name, index=False)

    print('==> before connect\t', aux.stastic_SecNumber_UserNumber_SkillNumber(data, code0.DatasetParameter()))
    print('==> after connect\t', aux.stastic_SecNumber_UserNumber_SkillNumber(newdata, code0.DatasetParameter()))

    return newdata


if __name__ == "__main__":
    filename = './data/assistment2009/skill_builder_data_corrected.csv'
    newfile = './data/assistment2009/connect_dataset_small.csv'

    data = pd.read_csv(filename, encoding='latin-1', error_bad_lines=False, index_col=False)
    data = data[0:50000]
    connectUser(data, newfile)

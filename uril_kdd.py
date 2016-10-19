import pandas as pd
import os, sys, pyprind, csv, math
import aux
import code0_parameter as code0


def read_kdd_data_from_csv(dp):
    if os.path.exists(dp.processedFileName):
        print ("==> read ",dp.processedFileName)
        data = pd.read_csv(dp.processedFileName)
        secNumber, userNumber, skillNumber = aux.stastic_SecNumber_UserNumber_SkillNumber(data, dp)
        return data

    print("==> read kdd algebra_2005_2006_train dataset from csv")
    try:
        data = pd.read_csv(dp.csv_file_name, delimiter='\t')
    except:
        raise NameError("can't load " + dp.csv_file_name + " pleace check your file")
    print(data.columns)

    data.rename(columns={'Anon Student Id': 'user_id',
                         'KC(Default)': "skill_id",
                         'Step Duration (sec)': "time",
                         'Correct First Attempt': "correct",
                         'Hints': "hint_count",
                         'Step Name': "step_id",
                         'Problem Hierarchy': "unit_id",
                         'Problem Name': "problem_id",
                         'Problem View': "problem_view",
                         'Incorrects': "incorrect",
                         'Corrects': "correct_num",
                         'Opportunity(Default)': "opportunity"}, inplace=True)

    data = data[dp.filtedColumnNameList].fillna(0)
    if dp.dataSetSize == "small":
        data = data[0:50000]
    print("==> run ", dp.dataSetSize, " dataset")

    # convert text to integar and save the dict
    changeList = ['user_id', 'skill_id']
    size = len(data)
    for columnName in changeList:
        print("==> process ", columnName)
        c_dict = getColumnDict(data[columnName], columnName, dp)
        newcontent = []
        for i in pyprind.prog_percent(range(size), stream=sys.stdout):
            try:
                value = data.loc[i, columnName]
                temp = c_dict[value]
            except:
                temp = 0
            newcontent.append(temp)
        data[columnName] = newcontent

    data = normalization_continues_data(data)
    if os.path.exists(dp.processedFileName):
        os.remove(dp.processedFileName)
        print ('==> remove ',dp.processedFileName)
    data.to_csv(dp.processedFileName, index=False)
    print("==> dataset column name\n", data.columns)
    print("==> dataset shape\t", data.shape)
    print("==> save file to ", dp.processedFileName)
    secNumber, userNumber, skillNumber = aux.stastic_SecNumber_UserNumber_SkillNumber(data, dp)
    return data


def getColumnDict(dataList, name, dp):
    dt = {}
    newname = "data/kdd/" + name + "_" + str(dp.dataSetSize) + "_dict.csv"
    if os.path.exists(newname):
        os.remove(newname)
        print ('==> remove ',newname)
    dataUnique = list(dataList.unique())
    for index, value in enumerate(dataUnique):
        dt[value] = index + 1
    dt[0] = 0

    with open(newname, 'w') as f:
        w = csv.writer(f)
        for key, val in dt.items():
            w.writerow([key, val])
    print("==> save ", newname)
    return dt


def normalization_continues_data(data):
    print('==> normalize continues data')
    columns_name_list = ["time", "hint_count", 'problem_view']
    data = data.reset_index(drop=True)

    size = len(data)
    for column_name in columns_name_list:
        if column_name == "time":
            bins = [-1, 10, 60, 150, 300, 10000]
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
        elif column_name == "problem_view":
            bins = [-1, 2, 5, 10, 400]
            data[column_name] = pd.cut(data[column_name], bins, labels=False)
            data[column_name] += 1
            tmpList = []
            for i in pyprind.prog_percent(range(size), stream=sys.stdout, title=column_name):
                # print ("attempt_count\t",str(i))
                tmp = int(data.loc[i, column_name])
                tmpList.append(math.log((tmp + 1), 5))
            data['problem_view_normal'] = tmpList
        elif column_name == "hint_count":
            bins = [-1, 2, 5, 10, 100]
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


if __name__ == "__main__":
    dp = code0.DatasetParameter()
    read_kdd_data_from_csv(dp)

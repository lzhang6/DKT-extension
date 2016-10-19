import pandas as pd
import aux
import code1_data as code1
import code0_parameter as code0
import pyprind,os
import sys


def read_asssistment2009_data_from_csv(dp):
    if os.path.exists(dp.processedFileName):
        print ("==> read ",dp.processedFileName)
        data = pd.read_csv(dp.processedFileName)
        secNumber, userNumber, skillNumber = aux.stastic_SecNumber_UserNumber_SkillNumber(data, dp)
        return data

    print("==> read assistment2009 from csv")
    try:
        data = pd.read_csv(dp.csv_file_name, encoding='latin-1', error_bad_lines=False, index_col=False)
        if dp.csv_file_name == "data/assistment2009/skill_builder_data_corrected.csv":
            data = data.loc[:338000]
        elif dp.csv_file_name == "data/assistment2009/skill_builder_data.csv":
            data = data.loc[:450000]
        else:
            pass
    except:
        raise NameError("can't load " + dp.csv_file_name + " pleace check your file")
    print(data.columns)

    data.rename(columns={'ms_first_response': 'time', 'hint_count': '_hint_count','hint_total': 'hint_count'}, inplace=True)

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

    template_set, template_dict = aux.create_column_dict_and_set(data,'template_id',dp)
    problem_set, problem_dict = aux.create_column_dict_and_set(data,'problem_id',dp)
    for i in pyprind.prog_percent(range(len(data)),stream=sys.stdout,title='reorder template_id and problem_id to reduce max number'):
        data.loc[i, "template_id"] = template_dict[data.loc[i, "template_id"]]
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
    read_asssistment2009_data_from_csv(dp)

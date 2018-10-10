''' Unless stated otherwise, all software is provided free of charge. 
As well, all software is provided on an "as is" basis without warranty 
of any kind, express or implied. Under no circumstances and under no legal 
theory, whether in tort, contract, or otherwise, shall Liang Zhang be liable 
to you or to any other person for any indirect, special, incidental, 
or consequential damages of any character including, without limitation, 
damages for loss of goodwill, work stoppage, computer failure or malfunction, 
or for any and all other damages or losses. If you do not agree with these terms, 
then you are advised to not use the software.'''

import tensorflow as tf
import datetime

# Hyperparameter for all kinds of file
DATASETTYPE = 'assistment2009'  # 'assistment2009'|'cmu_stat_f2011'
if DATASETTYPE == 'cmu_stat_f2011':
    TARGETSIZE = 250
elif DATASETTYPE == 'assistment2009':
    TARGETSIZE = 1000
AUTOENCODER_ACT = 'tanh'  # tanh, sigmoid
CONNECT_DATASET_2009 = True

DATASETSIZE = "large"  # 'large | small'
RNN_layer_number = 1  # '1|2'
CELLTYPE = "LSTM"  # "RNN | LSTM | GRU"

BASELINE = True
AUTOENCODER_LABEL = False


class DatasetParameter(object):
    def __init__(self, data_type=DATASETTYPE):
        if data_type == DATASETTYPE:
            self.dataSetType = DATASETTYPE  # "assistment2009 | cmu_stat_f2011 "
        else:
            self.dataSetType = data_type
        self.dataSetSize = DATASETSIZE  # 'small |large'

        if self.dataSetType == "assistment2009":
            self.csv_file_name = "./data/assistment2009/skill_builder_data_corrected.csv"
            if CONNECT_DATASET_2009:
                self.processedFileName = "./data/assistment2009/processded_" + str(self.dataSetSize) + "_connected.csv"
            else:
                self.processedFileName = "./data/assistment2009/processded_" + str(
                    self.dataSetSize) + "_nonconnected.csv"

            self.filtedColumnNameList = ['skill_id', 'user_id', 'original', 'correct', 'attempt_count', 'time',
                                         'hint_count', 'problem_id', 'first_action', 'template_id', 'opportunity']
            self.connect_dataset = CONNECT_DATASET_2009
            self.connect_file_name = "./data/assistment2009/connected_" + str(self.dataSetSize) + ".csv"
            self.time_z_level = 'skill_id'
            self.time_threshold = 400
            self.time_interval = 0.05
            self.attemp_max = 10
            self.correct_boundary_list = [0.5, 0.7]
            self.time_boundary_list = [-0.8, -0.6, 0]

        elif self.dataSetType == "cmu_stat_f2011":
            self.csv_file_name = "./data/cmu_stat_f2011/cmu.txt"
            self.filtedColumnNameList = ['time', 'correct', 'skill_id', 'step_id', 'problem_id', 'user_id',
                                         'Level (Unit)', 'Level (Module)',"first_action", "attempt_level"]

        elif self.dataSetType == "kdd":
            self.csv_file_name = "data/kdd/algebra_2005_2006_train.txt"
            self.processedFileName = "data/kdd/processded_" + str(self.dataSetSize) + ".csv"
            self.filtedColumnNameList = ['skill_id', 'user_id', 'correct', 'time', 'hint_count', 'problem_view']
            # 'step_id','unit_id','problem_id','incorrect','correct_num','opportunity'
        else:
            raise ValueError("check DATASETTYPE")

        self.currentTime = datetime.datetime.now().strftime("%m-%d-%H:%M")

        if self.dataSetType == "assistment2009":
            ##config
            self.dataset_columns_for_cross_feature = [['skill_id', 'correct'], ['first_action', 'correct'],
                                                      ['time_level', 'correct'], ['attempt_level', 'correct'],
                                                      ['first_action', 'time_level', 'correct'],['skill_id', 'time_level'],
                                                      ['attempt_level', 'time_level', 'correct'], ]
            self.model_continues_columns = ["time", "hint_count", "attempt_count"]
            self.model_category_columns = ["first_action", "time_level", "attempt_level"]
            self.model_cross_columns = [['skill_id', 'time_level'],['time_level', 'correct']]  # "the continues data columns needed to consider"
        elif self.dataSetType == 'cmu_stat_f2011':
            self.dataset_columns_for_cross_feature = [['skill_id', 'correct'], ['skill_id', 'time_level'],['time_level', 'correct']]
            self.model_continues_columns = ["time"]
            self.model_category_columns = ["first_action", "time_level", "attempt_level"]
            self.model_cross_columns = [['time_level', 'correct']]  # "the continues data columns needed to consider"

        elif self.dataSetType == 'kdd':
            self.dataset_columns_for_cross_feature = [['skill_id', 'correct'], ['time_level', 'correct']]
            self.model_continues_columns = ["time", "hint_count", "problem_view"]
            self.model_category_columns = ["time", "hint_count", "problem_view"]
            self.model_cross_columns = [['time_level', 'correct']]  # "the continues data columns needed to consider"

        if [['skill_id', 'correct']] in self.model_cross_columns:
            self.model_cross_columns.remove(['skill_id', 'correct'])
        elif [['correct', 'skill_id']] in self.model_cross_columns:
            self.model_cross_columns.remove(['correct', 'skill_id'])

        self.dataset_columns_for_cross_feature = self.__sortList(self.dataset_columns_for_cross_feature)
        self.model_cross_columns = self.__sortList(self.model_cross_columns)
        for items in self.model_cross_columns:
            if items not in self.dataset_columns_for_cross_feature:
                raise ValueError('model_cross_columns must in dataset_columns_for_cross_feature')
            for item in items:
                if item not in self.filtedColumnNameList + ['skill_id'] + ['time_level'] + ['attempt_level']:
                    raise ValueError(item, " not in filtedColumnNameList")
        # need to change value
        self.columnsName_to_index = {}
        self.columns_max = {}
        self.columns_numb = {}
        self.seq_width = 0
        self.skill_num = 0

    def __sortList(self, listName):
        return sorted(listName)

    def convertCrossCoumnsToNameList(self, Flag=True):
        if Flag:
            mcu = self.dataset_columns_for_cross_feature
        else:
            mcu = self.model_cross_columns
        crossFeatureNameList = []
        if len(mcu) != 0:
            for index_ccl, crossColumnsList in enumerate(mcu):
                crossFeatureName = ''
                if len(set(crossColumnsList)) <= 1:
                    raise ValueError("need two different feature at least ")

                for index_cc, crossColumn in enumerate(crossColumnsList):
                    if index_cc == 0:
                        crossFeatureName = crossColumn
                    else:
                        crossFeatureName = crossFeatureName + " " + crossColumn
                crossFeatureNameList.append(crossFeatureName)
        return crossFeatureNameList


class autoencoderParameter(object):
    def __init__(self):
        self.epoch_rbm = 10
        self.epoch_autoencoder = 10
        self.batch_size = 50
        self.num_steps = 100


class SAEParamsConfig(object):
    def __init__(self):
        self.learning_rate = 0.005
        self.min_lr = 0.0001
        self.lr_decay = 0.98
        self.layer_num = 1
        self.init_scale = 0.05
        self.target_size = TARGETSIZE
        self.max_max_epoch = 5
        self.display_step = 1

        self.batch_size = 300
        self.num_steps = 0  # need to resign value of time stampes
        self.seq_width = 0  # need to resign value


# Parameter for RNN
class ModelParamsConfig(object):
    def __init__(self, dp):
        self.num_steps = 0  # need to resign value of time stampes
        self.skill_num = 0  # need to resign value of skill number
        self.seq_width = 0  # need to resign value
        if dp.dataSetType == 'kdd':
            self.batch_size = 5
        elif dp.dataSetType == 'cmu_stat_f2011':
            self.batch_size = 10
        else:
            self.batch_size = 30
        self.max_max_epoch = 40
        self.num_layer = RNN_layer_number
        self.cell_type = CELLTYPE  # "RNN | LSTM | GRU"
        self.hidden_size = 200
        self.hidden_size_2 = 150

        self.init_scale = 0.05
        self.learning_rate = 0.05
        self.max_grad_norm = 4
        self.max_epoch = 5
        self.keep_prob = 0.6
        self.lr_decay = 0.9
        self.momentum = 0.95
        self.min_lr = 0.0001


if __name__ == "__main__":
    param_ass = DatasetParameter()
    print(param_ass.convertCrossCoumnsToNameList())

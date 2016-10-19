import tensorflow as tf
import numpy as np
import code0_parameter as code0
import code1_data as code1


class ONEHOTENCODERINPUT(object):
    def __init__(self, ap, dp, inputs,printControl=True):
        self.batch_size = batch_size = ap.batch_size
        self.num_steps = num_steps = ap.num_steps
        self.seq_width = seq_width = len(dp.columnsName_to_index)
        self.skill_num = dp.skill_num
        self.dp = dp
        self.ap = ap
        self.model_continues_columns = dp.model_continues_columns
        self.model_category_columns = dp.model_category_columns
        self.model_cross_columns = dp.model_cross_columns
        self.inputs = inputs
        self.printControl=printControl

        if dp.dataSetType == "assistment2009":
            width_deep_width_dict = {"skill_id": dp.columns_max['skill_id'] + 1,
                                     "correct": dp.columns_max['correct'] + 1,

                                     "attempt_count_normal": 1, "time_normal": 1, "hint_count_normal": 1,

                                     "time": dp.columns_max['time'] + 1, "hint_count": dp.columns_max['hint_count'] + 1,
                                     "attempt_count": dp.columns_max['attempt_count'] + 1,

                                     "first_action": dp.columns_max['first_action'] + 1, "problem_id": 100,
                                     "template_id": 100}
            self.data_attempt_count = tf.slice(self.inputs, [0, 0, dp.columnsName_to_index['attempt_count']],
                                               [-1, -1, 1])
            tmp_attempt_count = tf.to_int32(self.data_attempt_count)
            self.data_attempt_count_process = tf.to_float(tf.squeeze(
                tf.one_hot(indices=tmp_attempt_count, depth=width_deep_width_dict['attempt_count'], on_value=1.0,
                           off_value=0.0, axis=-1)))
            self.data_attempt_count_normal = tf.slice(self.inputs,
                                                      [0, 0, dp.columnsName_to_index['attempt_count_normal']],
                                                      [-1, -1, 1])
            self.data_first_action = tf.to_int32(
                tf.slice(self.inputs, [0, 0, dp.columnsName_to_index['first_action']], [-1, -1, 1]))
            self.data_first_action_process = tf.to_float(tf.squeeze(
                tf.one_hot(indices=self.data_first_action, depth=width_deep_width_dict['first_action'], on_value=1.0,
                           off_value=0.0, axis=-1)))

            # category， embedding
            self.data_problem_id = tf.to_int32(
                tf.slice(self.inputs, [0, 0, dp.columnsName_to_index['problem_id']], [-1, -1, 1]))
            embedding_problem_id = tf.get_variable("embedding_problem_id",
                                                   [196500, width_deep_width_dict['problem_id']], dtype=tf.float32)
            self.data_problem_id_process = tf.to_float(
                tf.squeeze(tf.nn.embedding_lookup(embedding_problem_id, self.data_problem_id)))

            self.data_template_id = tf.to_int32(
                tf.slice(self.inputs, [0, 0, dp.columnsName_to_index['template_id']], [-1, -1, 1]))
            embedding_template_id = tf.get_variable("embedding_template_id",
                                                    [103000, width_deep_width_dict['template_id']], dtype=tf.float32)
            self.data_template_id_process = tf.to_float(
                tf.squeeze(tf.nn.embedding_lookup(embedding_template_id, self.data_template_id)))

        elif dp.dataSetType == "assistment2014":
            width_deep_width_dict = {"skill_id": dp.columns_max['skill_id'] + 1,
                                     "correct": dp.columns_max['correct'] + 1,

                                     "attempt_count_normal": 1, "time_normal": 1, "hint_count_normal": 1,

                                     "time": dp.columns_max['time'] + 1, "hint_count": dp.columns_max['hint_count'] + 1,
                                     "attempt_count": dp.columns_max['attempt_count'] + 1,

                                     "first_action": dp.columns_max['first_action'] + 1, "problem_id": 100,
                                     "bottom_hint": 2}
            self.data_attempt_count = tf.slice(self.inputs, [0, 0, dp.columnsName_to_index['attempt_count']],
                                               [-1, -1, 1])
            tmp_attempt_count = tf.to_int32(self.data_attempt_count)
            self.data_attempt_count_process = tf.to_float(tf.squeeze(
                tf.one_hot(indices=tmp_attempt_count, depth=width_deep_width_dict['attempt_count'], on_value=1.0,
                           off_value=0.0, axis=-1)))
            self.data_attempt_count_normal = tf.slice(self.inputs,
                                                      [0, 0, dp.columnsName_to_index['attempt_count_normal']],
                                                      [-1, -1, 1])
            self.data_first_action = tf.to_int32(
                tf.slice(self.inputs, [0, 0, dp.columnsName_to_index['first_action']], [-1, -1, 1]))
            self.data_first_action_process = tf.to_float(tf.squeeze(
                tf.one_hot(indices=self.data_first_action, depth=width_deep_width_dict['first_action'], on_value=1.0,
                           off_value=0.0, axis=-1)))
            # category， embedding
            self.data_problem_id = tf.to_int32(
                tf.slice(self.inputs, [0, 0, dp.columnsName_to_index['problem_id']], [-1, -1, 1]))
            embedding_problem_id = tf.get_variable("embedding_problem_id",
                                                   [196500, width_deep_width_dict['problem_id']], dtype=tf.float32)
            self.data_problem_id_process = tf.to_float(
                tf.squeeze(tf.nn.embedding_lookup(embedding_problem_id, self.data_problem_id)))

            self.data_bottom_hint = tf.slice(self.inputs, [0, 0, dp.columnsName_to_index['bottom_hint']], [-1, -1, 1])


        else:  # kdd
            width_deep_width_dict = {"skill_id": dp.columns_max['skill_id'] + 1,
                                     "correct": dp.columns_max['correct'] + 1,
                                     "time": dp.columns_max['time'] + 1,
                                     "time_normal": 1,
                                     "hint_count": dp.columns_max['hint_count'] + 1,
                                     "hint_count_normal": 1,
                                     "problem_view_normal": 1,
                                     "problem_view": dp.columns_max['problem_view'] + 1}
            self.data_problem_view = tf.slice(self.inputs, [0, 0, dp.columnsName_to_index['problem_view']], [-1, -1, 1])
            tmp_problem_view = tf.to_int32(self.data_problem_view)
            self.data_problem_view_process = tf.to_float(tf.squeeze(
                tf.one_hot(indices=tmp_problem_view, depth=width_deep_width_dict['problem_view'], on_value=1.0, off_value=0.0, axis=-1)))
            self.data_problem_view_normal = tf.slice(self.inputs, [0, 0, dp.columnsName_to_index['problem_view_normal']], [-1, -1, 1])

        self.data_skill_id = tf.to_int32(
            tf.slice(self.inputs, [0, 0, dp.columnsName_to_index['skill_id']], [-1, -1, 1]))
        self.data_skill_id_process = tf.to_float(tf.squeeze(
            tf.one_hot(indices=self.data_skill_id, depth=width_deep_width_dict['skill_id'], on_value=1.0, off_value=0.0,
                       axis=-1)))
        self.data_correct = tf.slice(self.inputs, [0, 0, dp.columnsName_to_index['correct']], [-1, -1, 1])

        self.data_time = tf.slice(self.inputs, [0, 0, dp.columnsName_to_index['time']], [-1, -1, 1])
        tmp_time = tf.to_int32(self.data_time)
        self.data_time_process = tf.to_float(tf.squeeze(
            tf.one_hot(indices=tmp_time, depth=width_deep_width_dict['time'], on_value=1.0, off_value=0.0, axis=-1)))
        self.data_time_normal = tf.slice(self.inputs, [0, 0, dp.columnsName_to_index['time_normal']], [-1, -1, 1])

        self.data_hint_count = tf.slice(self.inputs, [0, 0, dp.columnsName_to_index['hint_count']], [-1, -1, 1])
        tmp_hint_count = tf.to_int32(self.data_hint_count)
        self.data_hint_count_process = tf.to_float(tf.squeeze(
            tf.one_hot(indices=tmp_hint_count, depth=width_deep_width_dict['hint_count'], on_value=1.0, off_value=0.0,
                       axis=-1)))
        self.data_hint_count_normal = tf.slice(self.inputs, [0, 0, dp.columnsName_to_index['hint_count_normal']],
                                               [-1, -1, 1])


    def getSkillCorrectMerge(self):
        featureList = [self.data_skill_id_process, self.data_correct]
        TensorskillCorrect = tf.concat(2, featureList)
        if self.printControl: print("==> [Tensor Shape] skill_id and correct merge formate\t", TensorskillCorrect.get_shape())
        return TensorskillCorrect

    def getSkillCorrectCrossFeature(self):
        TensorCrossFeatures = self._getCrossFeature(['skill_id correct'])
        if self.printControl: print("==> [Tensor Shape] skill_id and correct cross feature\t", TensorCrossFeatures.get_shape())
        return TensorCrossFeatures

    def getContinuesFeatureInputs(self):
        featureList = []
        for columnName in set(self.model_continues_columns):
            if columnName == 'time':
                featureList.append(self.data_time_normal)
            elif columnName == 'attempt_count':
                featureList.append(self.data_attempt_count_normal)
            elif columnName == 'hint_count':
                featureList.append(self.data_hint_count_normal)
            elif columnName == 'problem_view':
                featureList.append(self.data_problem_view_normal)
            elif columnName in ['skill_id', 'correct']:
                pass
            else:
                raise ValueError('only support time、attempt_count、hint_count')

        TensorContinuesFeature = tf.concat(2, featureList)
        if self.printControl: print("==> [Tensor Shape] continues features\t", TensorContinuesFeature.get_shape())
        return TensorContinuesFeature

    def getCategoryFeatureInputs(self):
        featureList = []
        for columnName in set(self.model_category_columns):
            if columnName == 'first_action':
                featureList.append(self.data_first_action_process)
            elif columnName == 'problem_id':
                featureList.append(self.data_problem_id_process)
            elif columnName == 'problem_view':
                featureList.append(self.data_problem_view_process)
            elif columnName == 'template_id':
                featureList.append(self.data_template_id_process)
            elif columnName == 'time':
                featureList.append(self.data_time_process)
            elif columnName == 'attempt_count':
                featureList.append(self.data_attempt_count_process)
            elif columnName == 'hint_count':
                featureList.append(self.data_hint_count_process)
            elif columnName == 'bottom_hint':
                featureList.append(self.data_bottom_hint_process)
            elif columnName == 'problem_view':
                featureList.append(self.data_problem_view_process)

            elif columnName in ['skill_id', 'correct']:
                pass
            else:
                raise ValueError('Check your model_category_columns configuration')

        TensorCategoryFeature = tf.concat(2, featureList)
        if self.printControl: print("==> [Tensor Shape] category features\t", TensorCategoryFeature.get_shape())
        return TensorCategoryFeature

    def getCrossFeatureAll(self):
        crossFeatureNameList = self.dp.convertCrossCoumnsToNameList(Flag=False)
        TensorCrossFeatures = self._getCrossFeature(crossFeatureNameList)
        if self.printControl: print("==> [Tensor Shape] Cross Feature whole\t", TensorCrossFeatures.get_shape())
        return TensorCrossFeatures

    def _getCrossFeature(self, crossFeatureNameList):
        if crossFeatureNameList == ['skill_id correct'] or crossFeatureNameList == ['correct skill_id']:
            crossFeatureNameList = ['skill_id correct']

        wide_length = 0
        for i, crossFeatureName in enumerate(crossFeatureNameList):  # crossFeatureName is a string'correct first_response_time'
            depthValue = int(self.dp.columns_max[crossFeatureName] + 1)
            wide_length += depthValue

            tmp_value = tf.to_int32(
                tf.slice(self.inputs, [0, 0, self.dp.columnsName_to_index[crossFeatureName]], [-1, -1, 1]))
            tmp_value_ohe = tf.to_float(
                tf.squeeze(tf.one_hot(indices=tmp_value, depth=depthValue, on_value=1.0, off_value=0.0, axis=-1)))
            if self.printControl: print("==> [Tensor Shape] Cross Feature--", crossFeatureName, " width\t", depthValue)

            if i == 0:
                TensorCrossFeatures = tmp_value_ohe
            else:
                TensorCrossFeatures = tf.concat(2, [TensorCrossFeatures, tmp_value_ohe])
        # if no cross features, the return value is null
        return TensorCrossFeatures

    def get_init_value_for_train_weights(self):
        featureslist = [self.getSkillCorrectCrossFeature(), self.getCrossFeatureAll(), self.getCategoryFeatureInputs(),
                        self.getContinuesFeatureInputs()]
        x_tmp = tf.concat(2, featureslist)
        x = tf.reshape(x_tmp, [self.batch_size * self.num_steps, -1])
        return x


if __name__ == "__main__":
    dp = code0.DatasetParameter()
    ap = code0.autoencoderParameter()

    dataset, labels = code1.load_data(dp)
    # tuple_data = code1.convert_data_labels_to_tuples(dataset, labels)

    skill_num = len(dataset['skill_id'].unique()) + 1  # 0 for unlisted skill_id
    dp.skill_num = skill_num
    dp.skill_set = list(dataset['skill_id'].unique())
    dp.columns_max, dp.columns_numb, dp.columnsName_to_index = code1.get_columns_info(dataset)
    dp.seq_width = len(dp.columnsName_to_index)

    print("columns_max\n", dp.columns_max)
    print("columns_numb\n", dp.columns_numb)
    print("columnsName_to_index\n", dp.columnsName_to_index)

    data = np.random.randint(low=0,high=2, size=())
    g =tf.Graph()
    with g.as_default():
        inputs = tf.placeholder(tf.float32, [ap.batch_size, ap.num_steps, len(dp.columnsName_to_index)])
        m = ONEHOTENCODERINPUT(ap=ap, dp=dp,inputs=inputs)

    with tf.Session(graph=g) as sess:
        m.getSkillCorrectMerge()
        m.getContinuesFeatureInputs()
        m.getCategoryFeatureInputs()
        print("-" * 60)
        m.getSkillCorrectCrossFeature()
        print("-" * 60)
        m.getCrossFeatureAll()

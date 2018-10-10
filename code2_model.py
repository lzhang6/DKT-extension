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
import numpy as np
from code0_parameter import AUTOENCODER_ACT, BASELINE, TARGETSIZE, AUTOENCODER_LABEL
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops.rnn_cell import LSTMCell, BasicRNNCell, GRUCell, DropoutWrapper
from uril_oneHotEncoder import ONEHOTENCODERINPUT


class Model(object):
    def __init__(self, is_training, config, dp):
        self._batch_size = batch_size = config.batch_size

        self._min_lr = config.min_lr
        self.hidden_size = hidden_size = config.hidden_size
        self.hidden_size_2 = hidden_size_2 = config.hidden_size_2
        self.skill_set = dp.skill_set
        self.num_steps = num_steps = config.num_steps
        self.skill_num = skill_numb = config.skill_num
        self.seq_width = seq_width = len(dp.columnsName_to_index)
        self.skill_num = skill_num = dp.skill_num

        # load data
        self.inputs = tf.placeholder(tf.float32, [batch_size, num_steps, seq_width])
        self.inputs_wide_skill_correct = tf.placeholder(tf.int32, [batch_size, num_steps])
        self._target_id = tf.placeholder(tf.int32, [None])
        self._target_correctness = target_correctness = tf.placeholder(tf.float32, [None])

        ohe = ONEHOTENCODERINPUT(config, dp, self.inputs)

        # load features
        if not BASELINE:
            if AUTOENCODER_LABEL:
                ###########################################################################################################
                featurelist = [ohe.getSkillCorrectCrossFeature(),ohe.getCrossFeatureAll()]#,ohe.getCategoryFeatureInputs()]
                                # ohe.getCategoryFeatureInputs()], # ohe.getContinuesFeatureInputs()]
                ###########################################################################################################
                tmp_v = tf.concat(2, featurelist)
                tmp_vs = tf.reshape(tmp_v, [-1, int(tmp_v.get_shape()[-1])])

                if AUTOENCODER_ACT == 'tanh':
                    transfer_function = tf.nn.tanh
                elif AUTOENCODER_ACT == 'sigmoid':
                    transfer_function = tf.nn.sigmoid

                path = './weights/' + dp.dataSetType + '/weights_' + str(tmp_vs.get_shape()[-1]) + '_' + str(
                    TARGETSIZE) + '.csv'
                autoencoderweights = tf.constant(np.loadtxt(path), dtype=tf.float32)
                path = './weights/' + dp.dataSetType + '/bias_' + str(tmp_vs.get_shape()[-1]) + '_' + str(
                    TARGETSIZE) + '.csv'
                autoencoderBias = tf.constant(np.loadtxt(path), dtype=tf.float32)
                tmp_vs = transfer_function(tf.matmul(tmp_vs, autoencoderweights) + autoencoderBias)
            else:
                ###########################################################################################################
                featurelist = [ohe.getSkillCorrectCrossFeature(), ohe.getCrossFeatureAll(), ohe.getCategoryFeatureInputs()]
                # featurelist = [ohe.getSkillCorrectCrossFeature(), ohe.getCrossFeatureAll()]
                ###########################################################################################################
                tmp_v = tf.concat(2, featurelist)
                print("==> [Tensor Shape] Final Shape\t", tmp_v.get_shape())
                tmp_vs = tf.reshape(tmp_v, [-1, int(tmp_v.get_shape()[-1])])
        else:
            tmp_v = ohe.getSkillCorrectCrossFeature()
            tmp_vs = tf.reshape(tmp_v, [-1, int(tmp_v.get_shape()[-1])])
        input_RNN = tf.reshape(tmp_vs, [batch_size, num_steps, -1])

        cell = self.getCell(is_training=is_training, dp=dp, config=config)
        self._initial_state = cell.zero_state(batch_size, tf.float32)

        outputs = []
        state = self._initial_state

        with tf.variable_scope(config.cell_type):
            for time_step in range(num_steps):
                if time_step > 0: tf.get_variable_scope().reuse_variables()
                (cell_output, state) = cell(input_RNN[:, time_step, :], state)
                outputs.append(cell_output)

        if config.num_layer == 1:
            size_rnn_out = hidden_size
        elif config.num_layer == 2:
            size_rnn_out = hidden_size_2
        else:
            raise ValueError("only support 1-2 layers, check your layer number!")

        output_RNN = tf.reshape(tf.concat(1, outputs), [-1, size_rnn_out])
        softmax_w = tf.get_variable("softmax_w", [size_rnn_out, skill_numb])
        softmax_b = tf.get_variable("softmax_b", [skill_numb])

        logits = tf.matmul(output_RNN, softmax_w) + softmax_b

        # pick up the right one
        self.logits = logits = tf.reshape(logits, [-1])
        self.selected_logits = selected_logits = tf.gather(logits, self.target_id)

        # make prediction
        self._pred = self._pred_values = tf.sigmoid(selected_logits)

        loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(selected_logits, target_correctness))
        self._cost = loss

        if not is_training:
            return

        self._lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), config.max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self.lr)

        self._train_op = optimizer.apply_gradients(zip(grads, tvars))

    def assign_lr(self, session, lr_value):
        if (lr_value > self.min_lr):
            session.run(tf.assign(self._lr, lr_value))
        else:
            session.run(tf.assign(self._lr, self.min_lr))

    def getCell(self, is_training, dp, config):
        # code for RNN
        if is_training == True:
            print("==> Construct ", config.cell_type, " graph for training")
        else:
            print("==> Construct ", config.cell_type, " graph for testing")

        if config.cell_type == "LSTM":
            if config.num_layer == 1:
                basicCell = LSTMCell(config.hidden_size, forget_bias=0.0, state_is_tuple=True)
            elif config.num_layer == 2:
                basicCell = LSTMCell(config.hidden_size, forget_bias=0.0, state_is_tuple=True)
                basicCell_2 = LSTMCell(config.hidden_size_2, forget_bias=0.0, state_is_tuple=True)
            else:
                raise ValueError("config.num_layer should be 1:2 ")
        elif config.cell_type == "RNN":
            if config.num_layer == 1:
                basicCell = BasicRNNCell(config.hidden_size)
            elif config.num_layer == 2:
                basicCell = BasicRNNCell(config.hidden_size)
                basicCell_2 = BasicRNNCell(config.hidden_size_2)
            else:
                raise ValueError("config.num_layer should be [1-3] ")
        elif config.cell_type == "GRU":
            if config.num_layer == 1:
                basicCell = GRUCell(config.hidden_size, forget_bias=0.0, state_is_tuple=True)
            elif config.num_layer == 2:
                basicCell = GRUCell(config.hidden_size, forget_bias=0.0, state_is_tuple=True)
                basicCell_2 = GRUCell(config.hidden_size_2, forget_bias=0.0, state_is_tuple=True)
            else:
                raise ValueError("only support 1-2 layers ")
        else:
            raise ValueError("cell type should be GRU,LSTM,RNN")

            # add dropout layer between hidden layers
        if is_training and config.keep_prob < 1:
            if config.num_layer == 1:
                basicCell = DropoutWrapper(basicCell, input_keep_prob=config.keep_prob,
                                           output_keep_prob=config.keep_prob)
            elif config.num_layer == 2:
                basicCell = DropoutWrapper(basicCell, input_keep_prob=config.keep_prob,
                                           output_keep_prob=config.keep_prob)
                basicCell_2 = DropoutWrapper(basicCell_2, input_keep_prob=config.keep_prob,
                                             output_keep_prob=config.keep_prob)
            else:
                pass

        if config.num_layer == 1:
            cell = rnn_cell.MultiRNNCell([basicCell], state_is_tuple=True)
        elif config.num_layer == 2:
            cell = rnn_cell.MultiRNNCell([basicCell, basicCell_2], state_is_tuple=True)

        return cell

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def min_lr(self):
        return self._min_lr

    @property
    def auc(self):
        return self._auc

    @property
    def pred(self):
        return self._pred

    @property
    def target_id(self):
        return self._target_id

    @property
    def target_correctness(self):
        return self._target_correctness

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def pred_values(self):
        return self._pred_values

    @property
    def cost(self):
        return self._cost

    @property
    def final_state(self):
        return self._final_state

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op


if __name__ == "__main__":
    pass

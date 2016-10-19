import tensorflow as tf
import math
import code0_parameter as code0
import code1_data as code1
from uril_oneHotEncoder import ONEHOTENCODERINPUT
import pyprind, sys,os
import numpy as np


class SIMPLEAUTOENCODER(object):
    def __init__(self, SAEconfig, dp):
        self.inputs = tf.placeholder(tf.float32, [SAEconfig.batch_size, SAEconfig.num_steps, SAEconfig.seq_width])
        ohe = ONEHOTENCODERINPUT(SAEconfig, dp, self.inputs)
        #self.mask = tf.placeholder(tf.float32, [SAEconfig.batch_size*SAEconfig.num_steps, SAEconfig.seq_width])
        #############################################################################################################
        featureslist = [ohe.getSkillCorrectCrossFeature(),ohe.getCrossFeatureAll()]#, ohe.getCategoryFeatureInputs()]#,
                        #ohe.getContinuesFeatureInputs()]
        #############################################################################################################
        x_tmp = tf.concat(2, featureslist)
        self.dp = dp
        self.x = x = tf.reshape(x_tmp, [SAEconfig.batch_size * SAEconfig.num_steps, -1])
        self.dimensions = dimensions = [int(x.get_shape()[-1]), code0.TARGETSIZE]
        #xp = self.mask*x
        print("n_input\t", str(dimensions[0]), "\tn_output\t", str(dimensions[1]))
        W_init_max = 4 * np.sqrt(6. / (dimensions[0] + dimensions[1]))
        W_init = tf.random_uniform(shape=dimensions, minval=-W_init_max,maxval=W_init_max)
        self.WE = WE = tf.Variable(W_init)
        self.bE = bE = tf.Variable(tf.zeros([dimensions[-1]]))
        if code0.AUTOENCODER_ACT == 'tanh':
            transfer_function = tf.nn.tanh
        elif code0.AUTOENCODER_ACT == 'sigmoid':
            transfer_function = tf.nn.sigmoid
        featureVector = transfer_function(tf.matmul(x, WE) + bE)

        self.WD = WD = tf.transpose(WE)
        #self.WD = WD = tf.Variable(tf.random_normal([self.dimensions[1],self.dimensions[0],], stddev=0.35))
        self.bD = bD = tf.Variable(tf.zeros([dimensions[0]]))
        y = transfer_function(tf.matmul(featureVector, WD) + bD)
        #self.learning_rate = tf.placeholder(tf.float32,1)
        self.cost = cost = tf.reduce_sum(tf.square(y - x))
        #self.optimizer = tf.train.GradientDescentOptimizer(SAEconfig.learning_rate).minimize(cost)
        self.optimizer = tf.train.AdamOptimizer(SAEconfig.learning_rate).minimize(cost)

        self.avgcost = tf.div(cost, tf.to_float(dimensions[0]))

    def saveWeights(self, sess):
        weigthpath = './weights/'+str(self.dp.dataSetType)+'/weights_' + str(self.dimensions[0]) + '_' + str(self.dimensions[1]) + '.csv'
        baispath = './weights/'+str(self.dp.dataSetType)+'/bias_' + str(self.dimensions[0]) + '_' + str(self.dimensions[1]) + '.csv'

        if os.path.exists(weigthpath):
            os.remove(weigthpath)
        if os.path.exists(baispath):
            os.remove(baispath)

        wt = self.WE.eval(sess)
        np.savetxt(weigthpath, wt)
        bs = self.bE.eval(sess)
        np.savetxt(baispath, bs)
        print("==> save weights to \t", os.path.dirname(weigthpath))


def run_ae_epoch(sess, model, data, TrainConfig):
    batch_number = int(len(data) / (TrainConfig.batch_size * TrainConfig.num_steps))
    learning_rate  = TrainConfig.learning_rate
    for i in pyprind.prog_percent(range(batch_number), stream=sys.stdout):
        x = np.zeros((TrainConfig.batch_size, TrainConfig.num_steps, TrainConfig.seq_width))
        kindex = i * (TrainConfig.batch_size * TrainConfig.num_steps)
        for ip in range(TrainConfig.batch_size):
            for j in range(TrainConfig.num_steps):
                x[ip, j, :] = data.iloc[kindex]
                kindex += 1
        #mask_np = np.random.binomial(1, 1 - TrainConfig.corruption_level, [TrainConfig.batch_size * TrainConfig.num_steps,TrainConfig.seq_width])
        learning_rate = learning_rate*TrainConfig.lr_decay
        if learning_rate<=TrainConfig.min_lr:
            learning_rate = TrainConfig.min_lr
        _ = sess.run(model.optimizer, feed_dict={model.inputs: x})
    avgcost = sess.run(model.avgcost, feed_dict={model.inputs: x})
    return avgcost


def trainAEWeights():
    if not code0.BASELINE:
        dp = code0.DatasetParameter()
        dataset, labels = code1.load_data(dp)

        dp.skill_num = len(dataset['skill_id'].unique()) + 1
        dp.skill_set = list(dataset['skill_id'].unique())
        dp.columns_max, dp.columns_numb, dp.columnsName_to_index = code1.get_columns_info(dataset)
        dp.seq_width = len(dp.columnsName_to_index)


        SAEconfig = code0.SAEParamsConfig()
        SAEconfig.num_steps = 30
        SAEconfig.seq_width = dp.seq_width

        g = tf.Graph()
        with g.as_default():
            model_autoencoder = SIMPLEAUTOENCODER(SAEconfig, dp)
            initializer = tf.random_uniform_initializer(-SAEconfig.init_scale, SAEconfig.init_scale)

        with tf.Session(graph=g) as sess:
            tf.initialize_all_variables().run()

            for i in range(SAEconfig.max_max_epoch):
                p = run_ae_epoch(sess, model_autoencoder, dataset, SAEconfig)
                print(str(i)+"/"+str(SAEconfig.max_max_epoch)+" epoch,avgcost ", str(p))
            model_autoencoder.saveWeights(sess)
    else:
        print("BASELINE model, don't need train weights")

if __name__ == "__main__":
    trainAEWeights()

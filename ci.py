''' Unless stated otherwise, all software is provided free of charge. 
As well, all software is provided on an "as is" basis without warranty 
of any kind, express or implied. Under no circumstances and under no legal 
theory, whether in tort, contract, or otherwise, shall Liang Zhang be liable 
to you or to any other person for any indirect, special, incidental, 
or consequential damages of any character including, without limitation, 
damages for loss of goodwill, work stoppage, computer failure or malfunction, 
or for any and all other damages or losses. If you do not agree with these terms, 
then you are advised to not use the software.'''

import pandas as pd
import numpy as np
import scipy.stats as st
import scipy as sp
import scipy.stats

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m,h

assistments_list = [
    './result/assistment2009/result_01-25-17:04.csv', #baseline
    './result/assistment2009/result_02-05-11:51.csv', #baseline + t/c
    './result/assistment2009/result_02-05-21:28.csv', #baseline + t/c [ae]
    './result/assistment2009/result_02-06-02:10.csv', #baseline + t/c + t + a + f
    './result/assistment2009/result_02-06-09:21.csv', #baseline + t/c + t + a + f [ae]
    './result/assistment2009/result_02-08-10:57.csv', #baseline + t/c + t/s + t + a + f [ae]
]

cmu_list = [
    './result/cmu_stat_f2011/result_01-24-23:23.csv', #baseline
    './result/cmu_stat_f2011/result_01-25-09:35.csv', #baseline + t/c
    './result/cmu_stat_f2011/result_01-29-17:35.csv', #baseline + t/c [ae]
    './result/cmu_stat_f2011/result_02-06-21:45.csv', #baseline + t/c + t + a + f
    './result/cmu_stat_f2011/result_02-07-08:44.csv', #baseline + t/c + t + a + f [ae]
    './result/cmu_stat_f2011/result_02-07-23:44.csv', #baseline + t/c + t/s + t + a + f [ae]
]

for name_list in [assistments_list,cmu_list]:
    print ("=="*25)
    for idx,name in enumerate(name_list):
        print ("\n","-"*5,idx,"\t",name,"-"*5,)
        data = pd.read_csv(name)
        data = data[(data['cv']!='average') & (data['type']!='train')]
        data = data[data['epoch']==8]

        aucs = data['auc']
        print ("auc mean and 95ci %2.3f\t%2.3f"%mean_confidence_interval(aucs))

        r2s = data['r2']
        print ("r2 mean and 95ci %2.3f\t%2.3f"%mean_confidence_interval(r2s))



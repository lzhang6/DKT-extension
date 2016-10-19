import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def getUserQuesNumList(dataList):
    a = list(dataList)
    target = np.empty((0, 2))
    size = len(a)
    temp = [a[0], 1]
    for i in range(1, size):
        if a[i] == a[i - 1]:
            temp[1] += 1
        else:
            target = np.vstack((target, temp))
            temp = [a[i], 1]
    return np.vstack((target, temp))

def counter(a):
    a = list(a)
    unique, counts = np.unique(a, return_counts=True)
    return unique, counts

file2009    = "/home/leon/projects/MDKT/data/assistment2009/other/processded_large.csv"
file2009new = "/home/leon/projects/MDKT/data/assistment2009/processded_large.csv"
file2014    = "/home/leon/projects/MDKT/data/assistment2014/processded_large.csv"

data2009    = pd.read_csv(file2009)
data2009new = pd.read_csv(file2009new)
data2014    = pd.read_csv(file2014)

kp = getUserQuesNumList(data2009['user_id'])
u2009,c2009 = counter(list(kp[:,1]))

_,c2009new = counter(data2009new['user_id'])
u2009new,c2009new = counter(c2009new)
_,c2014 = counter(data2014['user_id'])
u2014,c2014 = counter(c2014)

print (sum(c2009),"\t",sum(c2009new),"\t",sum(c2014))

plt.figure(1)
plt.plot(u2009,c2009,label='2009')
plt.plot(u2009new,c2009new,label='2009new')
plt.plot(u2014,c2014,label='2014')
plt.xlim([0,400])
#plt.ylim([0,1000])
plt.legend()
plt.show()

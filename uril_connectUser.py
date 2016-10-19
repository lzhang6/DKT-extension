import numpy as np
import pandas as pd
import pyprind as pp
import sys

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
            temp = [a[i], 1 ,i]
    return np.vstack((target, temp))


filename = '/home/leon/projects/MDKT/data/assistment2009/processded_large.csv'
newfile =  '/home/leon/projects/MDKT/data/assistment2009/kkk.csv'

"""
data = pd.read_csv(filename)
print("==> load data successful")
u,c = counter(data['user_id'])
UserNumberDict =dict(zip(u,c))

userQuesNumIndexList = getUserQuesNumIndexList(data['user_id'])
newdata = pd.DataFrame()

for i in pp.prog_percent(range(len(u)),stream=sys.stdout):
    for k in range(len(userQuesNumIndexList)):
        if userQuesNumIndexList[k,0] ==u[i]:
            temp = data.iloc[int(userQuesNumIndexList[k,2]):int(userQuesNumIndexList[k,2]+userQuesNumIndexList[k,1])]
            newdata = newdata.append(temp)

    #print ("i\t",i,'/',len(u),"\tuserId\t",u[i])

print (newdata)
newdata.reset_index(drop=True)
newdata.to_csv(newfile,index=False)
"""

data = pd.read_csv(newfile)
u,c = counter(data['user_id'])
print (len(getUserQuesNumIndexList(data['user_id'])))
print (len(u))

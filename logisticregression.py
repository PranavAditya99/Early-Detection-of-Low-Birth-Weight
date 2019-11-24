#Reference : My notes from Andrew Ng's ML course
#gives 70-90% accuracy - It's unreliable - probably cause data is not linearly seperable
#Gave 47% once and 95% once

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import random

class Logistic:
    def __init__(self, n, alpha):
        self.n = n
        self.alpha = alpha
        self.eps = 0.00001
    
    def sigmoid(self, x):
        return (1/(1+(np.exp(-x))))
    
    def costfunction(self,hyp,ny,m):
        #print(hyp.shape,ny.shape)
        return (1/m)*(((-ny)*np.log(hyp+self.eps)) - ((1-ny)*np.log((1-hyp)+self.eps)))
    
    def gradientdescent(self,x,y):
        self.theta = np.random.rand(x.shape[1])
        for i in range(self.n):
            ex = np.dot(x,self.theta)
            sig = self.sigmoid(ex)
            #print(sig.shape,x.T.shape)
            grad = np.dot(x.T,(sig-y))/y.size
            #print(y.size)
            #print(grad.shape)
            self.theta = self.theta - self.alpha*grad
            if(i%250000==0):
                print(f'loss:{self.costfunction(sig,y,y.size)}')
    
    def predict(self,x,thresh):
        return self.sigmoid(np.dot(x,self.theta))>thresh
            


def one_hot_encoding(data,col):  
    new_cols=list(map(str,np.unique(col)))
    
    data1=pd.DataFrame(columns=new_cols)
    d=dict()
    for i in new_cols:
        d[i]=0
    for i in col:
        d[str(i)]=1
        data1=data1.append(d,ignore_index=True)
        d[str(i)]=0
    data=data.drop([col.name],axis=1)
    for i in range(len(new_cols)):
        new_cols[i]=col.name+new_cols[i]
    data1.columns=new_cols
    bigdata = data1.join(data)
    return bigdata

random.seed(19)
path = "C:/Users/Tanay/College/3rd_year/5th_sem/Machine-Learning/Project/Data/"

data = pd.read_csv(path+'NewImputedDataset.csv')

data = one_hot_encoding(data,data.community)
#print(type(data))
data = one_hot_encoding(data,data.res)

#print(data.head())

acclist = list()
precision = list()
recall = list()
for i in range(10):
    train,test = train_test_split(data,shuffle = True, test_size = 0.2)
    train_y = train['reslt']
    train_x = train.drop(['reslt'],axis=1)
    
    test_y = test['reslt']
    test_x = test.drop(['reslt'],axis=1)
    
    train_x = train_x.values
    train_y = train_y.values
    test_x = test_x.values
    test_y = test_y.values
    train_x = np.array(train_x, dtype = 'float64')
    test_x = np.array(test_x, dtype = 'float64')
    #print(train_y)
    print(train_x)
    
    classifier = Logistic(n=500000,alpha = 0.5)
    
    classifier.gradientdescent(train_x,train_y)
    
    preds = classifier.predict(test_x,0.5)
    prediction = list()
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    
    for x in preds:
        if(x):
            prediction.append(1)
        else:
            prediction.append(0)
    
    c=0
    for a,b in zip(prediction,test_y):
        if a==b:
            c+=1
    for a,b in zip(prediction,test_y):
        if a==b==1:
            tp+=1
        elif a==b==0:
            tn+=1
        elif a==1 and b==0:
            fp+=1
        else:
            fn+=1
            
    precision.append((tp/(tp+fp)))
    recall.append((tp/(tp+fn)))
    acclist.append(c/test_y.size)      

print("Accuracy: ",sum(acclist)/len(acclist))
print("Precision: ",sum(precision)/len(precision))
print("Recall: ",sum(recall)/len(recall))

#Reference : My notes from Andrew Ng's ML course
#gives 70-90% accuracy - It's unreliable - probably cause data is not linearly seperable
#Gave 47% once and 95% once

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import random
import math


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
            #if(i%250000==0):
            #    print(f'loss:{self.costfunction(sig,y,y.size)}')
    
    def predict(self,x,thresh):
        return self.sigmoid(np.dot(x,self.theta))>thresh
            

def knn(x_train,y_train,k,pt):
    x=x_train.values.tolist()
    x=np.array(x)
    #print(x[0:5])
    pt=np.array(pt)
    
    for i in range(len(x_train)):
        x[i]=abs(x[i]-pt)
    for j in range(len(x)):
        x[j]=list(map(lambda x:float(x*x*x*x*x*x),x[j]))
    
    dist=[]
    for w in range(len(x)):
        dist.append((sum(x[w])**(1/6)))
    dictionary={"dist":dist,"y":y_train}
    d1=pd.DataFrame(dictionary)
    d1 = d1.sort_values(by ='dist' )
    d1=d1.reset_index()
    del d1['index']
    
    l=d1[0:k]['y']
    one=np.count_nonzero(l == 1)
    zero=np.count_nonzero(l == 0)
    if(one>zero):
        return(1)
    else:
        return(0)
        
        
def knn_full(x_train,y_train,k,x_test,y_test):
    y_pred=[]
    for i in range(len(x_test)):
        pt=x_test[i:i+1].values.tolist()
        pt=pt[0]
        y_pred.append( knn(x_train,y_train,k,pt))
    #print(y_pred)
    return y_pred
        

def knn_full_acc(x_train,y_train,k,x_test,y_test):
    y_pred=[]
    for i in range(len(x_test)):
        pt=x_test[i:i+1].values.tolist()
        pt=pt[0]
        y_pred.append( knn(x_train,y_train,k,pt))
    #print(y_pred)
    acc=abs(np.array(y_pred)-np.array(y_test))
    accuracy=np.count_nonzero(acc == 0)/len(y_pred)
    return accuracy
        




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

y=data.reslt
x=data.drop(["reslt"],axis=1)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


accuracy=0
pos=-1
t=[]
for i in range (1,len(x_train)+1):
    r=knn_full_acc(x_train,y_train,i,x_test,y_test)
    t.append(r)
    if (r>accuracy):
        accuracy=r
        pos=i

print("K=",pos)
print("accuracy=",accuracy)


acclist = list()
precision = list()
recall = list()
for i in range(10):
    train,test = train_test_split(data,shuffle = True, test_size = 0.2)
    
    train_y = train['reslt']
    train_x = train.drop(['reslt'],axis=1)
    ktrain_x = train_x
    ktrain_y = train_y
    
    test_y = test['reslt']
    test_x = test.drop(['reslt'],axis=1)
    ktest_x = test_x
    ktest_y = test_y
    
    train_x = train_x.values
    train_y = train_y.values
    test_x = test_x.values
    test_y = test_y.values
    train_x = np.array(train_x, dtype = 'float64')
    test_x = np.array(test_x, dtype = 'float64')
    #print(train_y)
   # print(train_x)
    
    classifier = Logistic(n=500000,alpha = 0.5)
    
    classifier.gradientdescent(train_x,train_y)
    
    predslogistic = classifier.predict(test_x,0.5)
    predictionlogistic = list()
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    
    for x in predslogistic:
        if(x):
            predictionlogistic.append(1)
        else:
            predictionlogistic.append(0)
    
    predictionknn = knn_full(ktrain_x,ktrain_y,pos,ktest_x,ktest_y)
    
    c=0
    for a,b in zip(predictionlogistic,test_y):
        if a==b:
            c+=1
    ck=0
    for a,b in zip(predictionknn,test_y):
        if a==b:
            ck+=1
    
    for pr in range(len(predictionlogistic)):
        if predictionlogistic[pr]==0:
            predictionlogistic[pr] = -1
    
    for pr in range(len(predictionknn)):
        if predictionknn[pr]==0:
            predictionknn[pr] = -1
    
    print(predictionknn,predictionlogistic)
    
    errknn = 1 - ck/test_y.size
    errlogistic = 1- c/test_y.size
    
    weightknn = 0.5*math.log((1-errknn)/errknn)
    weightlogistic = 0.5*math.log((1-errlogistic)/errlogistic)
    print(weightknn,weightlogistic)
    finalpredwt = list()
    
    for lo,kn in zip(predictionlogistic,predictionknn):
        finalpredwt.append(weightknn*kn + weightlogistic*lo )
        
    print(finalpredwt)
    
    finalpreds = list()
    for pr in finalpredwt:
        if(pr<0):
            finalpreds.append(0)
        else:
            finalpreds.append(1)
    
    for a,b in zip(finalpreds,test_y):
        if a==b==1:
            tp+=1
        elif a==b==0:
            tn+=1
        elif a==1 and b==0:
            fp+=1
        else:
            fn+=1
    
    cf=0
    for a,b in zip(finalpreds,test_y):
        if a==b:
            cf+=1
        
    precision.append((tp/(tp+fp)))
    recall.append((tp/(tp+fn)))
    acclist.append(cf/test_y.size)      

print("Accuracy: ",sum(acclist)/len(acclist))
print("Precision: ",sum(precision)/len(precision))
print("Recall: ",sum(recall)/len(recall))


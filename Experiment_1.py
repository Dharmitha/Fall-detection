# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 00:21:10 2018

@author: Admin
"""

import codecs
import csv
import numpy


from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.utils import np_utils
from sklearn.metrics import confusion_matrix,roc_auc_score,roc_curve,auc,accuracy_score
from sklearn.metrics import precision_score,recall_score,classification_report
import matplotlib.pyplot as plt




# Create a sequence classification instance
def get_sequence(n_timesteps,time):
    x = [data[index] for index in range((time*10)+1 ,(time*10)+10+1)]
    x = numpy.delete(x, (data.shape[1]-1), axis=1)
    x= numpy.array(x)
    y = [Y[index] for index in range((time*10) ,(time*10)+10)]
    y=numpy.array(y)

    # reshape input and output data to be suitable for LSTMs
    X = x.reshape(1, n_timesteps, (data.shape[1]-1))
    y = y.reshape(1, n_timesteps, y.shape[1])
    return X, y


# Load data from the csv file
csv_data = numpy.genfromtxt('Experiment1.csv', delimiter=',')
# Extract the output label and convert them to binary class matrix
output_data =[csv_data[i][-1] for i in range(1,7671)]
Y = np_utils.to_categorical(output_data)


# define problem properties
n_timesteps = 10
# define LSTM
model = Sequential()
model.add(LSTM(30,input_shape = (None, (data.shape[1]-1)),return_sequences=True))
model.add(Dropout(0.25))
model.add(Dense(6, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

print(model.summary())

# train LSTM
for epoch in range(0,149):
	# generate new random sequence
	X,y = get_sequence(n_timesteps,epoch)
	# fit model for one epoch on this sequence
	model.fit(X, y,batch_size=1, verbose=0)
    
for epoch in range(192,343):
	# generate new random sequence
	X,y = get_sequence(n_timesteps,epoch)
	# fit model for one epoch on this sequence
	model.fit(X, y,batch_size=1, verbose=0)
    
for epoch in range(385,534):
	# generate new random sequence
	X,y = get_sequence(n_timesteps,epoch)
	# fit model for one epoch on this sequence
	model.fit(X, y,batch_size=1, verbose=0)    
    
for epoch in range(576,727):
	# generate new random sequence
	X,y = get_sequence(n_timesteps,epoch)
	# fit model for one epoch on this sequence
	model.fit(X, y,batch_size=1, verbose=0)
countp=0
countn=0

ypredicted =  []
yactual = []



for d in range(150,191):
        X,y = get_sequence(n_timesteps,d)
        #yactual[] = [y[i] for i in range(len(y))]
        yhat = model.predict(X,verbose=0)[0]
        for index1 in range(10):
            i = np.where(yhat[index1] == yhat[index1].max())
            hin = i[0]
            for index2 in range(6):
                if(index2==hin):
                    yhat[index1][index2]=1
                else:
                    yhat[index1][index2]=0
        j= yhat
        k= y[0]
        for index1 in range(10):
            ypredicted.append(j[index1])
            yactual.append(k[index1])
         
            
for d in range(343,384):
        X,y = get_sequence(n_timesteps,d)
        #yactual[] = [y[i] for i in range(len(y))]
        yhat = model.predict(X,verbose=0)[0]
        for index1 in range(10):
            i = np.where(yhat[index1] == yhat[index1].max())
            hin = i[0]
            for index2 in range(6):
                if(index2==hin):
                    yhat[index1][index2]=1
                else:
                    yhat[index1][index2]=0
        j= yhat
        k= y[0]
        for index1 in range(10):
            ypredicted.append(j[index1])
            yactual.append(k[index1])
            
for d in range(534,575):
        X,y = get_sequence(n_timesteps,d)
        #yactual[] = [y[i] for i in range(len(y))]
        yhat = model.predict(X,verbose=0)[0]
        for index1 in range(10):
            i = np.where(yhat[index1] == yhat[index1].max())
            hin = i[0]
            for index2 in range(6):
                if(index2==hin):
                    yhat[index1][index2]=1
                else:
                    yhat[index1][index2]=0
        j= yhat
        k= y[0]
        for index1 in range(10):
            ypredicted.append(j[index1])
            yactual.append(k[index1])
        
for d in range(727,766):
        X,y = get_sequence(n_timesteps,d)
        #yactual[] = [y[i] for i in range(len(y))]
        yhat = model.predict(X,verbose=0)[0]
        for index1 in range(10):
            i = np.where(yhat[index1] == yhat[index1].max())
            hin = i[0]
            for index2 in range(6):
                if(index2==hin):
                    yhat[index1][index2]=1
                else:
                    yhat[index1][index2]=0
        j= yhat
        k= y[0]
        for index1 in range(10):
            ypredicted.append(j[index1])
            yactual.append(k[index1])
            
ya = []
yp = []
for index1 in range(len(ypredicted)):
            if (ypredicted[index1][0]==1 and ypredicted[index1][1]==0 and ypredicted[index1][2]==0 and ypredicted[index1][3]==0 and ypredicted[index1][4]==0 and ypredicted[index1][5]==0):
                yp.append(0)
            if (ypredicted[index1][0]==0 and ypredicted[index1][1]==1 and ypredicted[index1][2]==0 and ypredicted[index1][3]==0 and ypredicted[index1][4]==0 and ypredicted[index1][5]==0):
                yp.append(1)
            if (ypredicted[index1][0]==0 and ypredicted[index1][1]==0 and ypredicted[index1][2]==1 and ypredicted[index1][3]==0 and ypredicted[index1][4]==0 and ypredicted[index1][5]==0):
                yp.append(2)
            if (ypredicted[index1][0]==0 and ypredicted[index1][1]==0 and ypredicted[index1][2]==0 and ypredicted[index1][3]==1 and ypredicted[index1][4]==0 and ypredicted[index1][5]==0):
                yp.append(3)
            if (ypredicted[index1][0]==0 and ypredicted[index1][1]==0 and ypredicted[index1][2]==0 and ypredicted[index1][3]==0 and ypredicted[index1][4]==1 and ypredicted[index1][5]==0):
                yp.append(4)
            if (ypredicted[index1][0]==0 and ypredicted[index1][1]==0 and ypredicted[index1][2]==0 and ypredicted[index1][3]==0 and ypredicted[index1][4]==0 and ypredicted[index1][5]==1):
                yp.append(5)
for index1 in range(len(yactual)):
            if (yactual[index1][0]==1 and yactual[index1][1]==0 and yactual[index1][2]==0 and yactual[index1][3]==0 and yactual[index1][4]==0 and yactual[index1][5]==0):
                ya.append(0)
            if (yactual[index1][0]==0 and yactual[index1][1]==1 and yactual[index1][2]==0 and yactual[index1][3]==0 and yactual[index1][4]==0 and yactual[index1][5]==0):
                ya.append(1)
            if (yactual[index1][0]==0 and yactual[index1][1]==0 and yactual[index1][2]==1 and yactual[index1][3]==0 and yactual[index1][4]==0 and yactual[index1][5]==0):
                ya.append(2)
            if (yactual[index1][0]==0 and yactual[index1][1]==0 and yactual[index1][2]==0 and yactual[index1][3]==1 and yactual[index1][4]==0 and yactual[index1][5]==0):
                ya.append(3)
            if (yactual[index1][0]==0 and yactual[index1][1]==0 and yactual[index1][2]==0 and yactual[index1][3]==0 and yactual[index1][4]==1 and yactual[index1][5]==0):
                ya.append(4)   
            if (yactual[index1][0]==0 and yactual[index1][1]==0 and yactual[index1][2]==0 and yactual[index1][3]==0 and yactual[index1][4]==0 and yactual[index1][5]==1):
                ya.append(5) 
for i in range(len(ya)):
    if(ya[i] == yp[i]):
                countp = countp+1
    else:
        if(i!=len(ya)-1):
            if((ya[i+1]!=0 and yp[i]!=0) or (ya[i-1]!=0 and yp[i]!=0)):
                countp=countp+1
            else:
                countn = countn+1
        else:
            countn = countn+1

yaa=[]
ypp=[]
for i in range(len(ya)):
    if(ya[i]==yp[i]):
        yaa.append(ya[i])
        ypp.append(yp[i])
    else:
        if(ya[i]!=yp[i]):
            if((ya[i]==2 and yp[i+1]==2) or (ya[i-1]==2 and yp[i]==2) or (ya[i]==2 and yp[i-1]==2) or (ya[i+1]==2 and yp[i]==2)):
                yaa.append(2)
                ypp.append(2)
            elif((ya[i]==3 and yp[i+1]==3) or (ya[i-1]==3 and yp[i]==3) or (ya[i]==3 and yp[i-1]==3) or (ya[i+1]==3 and yp[i]==3)):
                yaa.append(3)
                ypp.append(3)
            elif((ya[i]==4 and yp[i+1]==4) or (ya[i-1]==4 and yp[i]==4) or (ya[i]==4 and yp[i-1]==4) or (ya[i+1]==4 and yp[i]==4)):
                yaa.append(4)
                ypp.append(4)
            elif((ya[i]==5 and yp[i+1]==5) or (ya[i-1]==5 and yp[i]==5) or (ya[i]==5 and yp[i-1]==5) or (ya[i+1]==5 and yp[i]==5)):
                yaa.append(5)
                ypp.append(5)
            else:
                yaa.append(ya[i])
                ypp.append(yp[i])



conf_arr = confusion_matrix(yaa, ypp)
conf_arr_1 = confusion_matrix(ya, yp)

Precision = precision_score(yaa, ypp,average='micro') 
Recall = recall_score(yaa, ypp,average='micro')
classification_report(yaa, ypp)
print(classification_report(ya, yp))
print(accuracy_score(ya,yp))


with codecs.open("output1.txt", "a", "utf-8") as my_file:
    wr = csv.writer(my_file,delimiter="\n")
    for i in range(len(ya)):
        t = ('Expected:', ya[i], 'Predicted', yp[i])
        wr.writerow(t)



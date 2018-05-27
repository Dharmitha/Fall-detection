# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 00:21:10 2018

@author: Admin
"""

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from numpy import genfromtxt
import numpy
from sklearn.utils import shuffle
import codecs
import csv
from sklearn.metrics import confusion_matrix,roc_auc_score,roc_curve,auc
from sklearn.metrics import precision_score,recall_score,classification_report
import matplotlib.pyplot as plt



# create a sequence classification instance
def get_sequence(n_timesteps,time):
    x = [data[index] for index in range((time*10)+1 ,(time*10)+1+10)]
    x = numpy.delete(x, (data.shape[1]-1), axis=1)
    x= numpy.array(x)
    y = [data[index][-1] for index in range((time*10)+1 ,(time*10)+1+10)]
    y=numpy.array(y)
    # reshape input and output data to be suitable for LSTMs
    X = x.reshape(1, n_timesteps, (data.shape[1]-1))
    y = y.reshape(1, n_timesteps, 1)
    return X, y



data = genfromtxt('Experiment6.csv', delimiter=',')

data = shuffle(data)
# define problem properties
n_timesteps = 10
# define LSTM
model = Sequential()
model.add(LSTM(30,input_shape = (None, (data.shape[1]-1)),return_sequences=True))
#model.add(TimeDistributed(Dense(1, activation='sigmoid')))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['acc'])

train_len = int(len(data)*0.7)
valid_len = int(len(data)*0.8)
test_len = len(data)-1

x1 = [data[index] for index in range(train_len+1, valid_len+1)]
x1 = numpy.delete(x1, (data.shape[1]-1), axis=1)
x1= numpy.array(x1)
y1 = [data[index][-1] for index in range(train_len+1, valid_len+1)]
y1=numpy.array(y1)
# reshape input and output data to be suitable for LSTMs
X1 = x1.reshape(1, valid_len-train_len, data.shape[1]-1)
y1 = y1.reshape(1, valid_len-train_len, 1)

# train LSTM
for epoch in range(int(train_len/10)):
	# generate new random sequence
	X,y = get_sequence(n_timesteps,0)
	# fit model for one epoch on this sequence
	model.fit(X, y,batch_size=1, verbose=0,validation_data=(X1,y1))
countp=0
countn=0
ypredicted = []
yactual = []
# evaluate LSTM
with codecs.open("output_exp4.txt", "a", "utf-8") as my_file:
    wr = csv.writer(my_file,delimiter="\n")
    for d in range(int(valid_len/10),int(test_len/10)):
        X,y = get_sequence(n_timesteps,d)
        #yactual[] = [y[i] for i in range(len(y))]
        yhat = model.predict_classes(X, verbose=0)
        j= yhat[0]
        k= y[0]
        for i in range(10):
            value = j[i][0]
            value1 =int( k[i][0])
            ypredicted.append(value)
            yactual.append(value1)
        wr.writerow('--------------------------------------------')
        for i in range(n_timesteps):
            text = ('Expected:', y[0, i], 'Predicted', yhat[0, i])
            wr.writerow(text)
            if(y[0,i] == yhat[0,i]):
                countp = countp+1
            else:
                if(i!=9):
                    if((y[0,i+1]==1 and yhat[0,i]==1) or (y[0,i-1]==1 and yhat[0,i]==1)):
                        countp=countp+1
                    else:
                        countn = countn+1
                else:
                    countn = countn+1
            
conf_arr = confusion_matrix(yactual, ypredicted)
print("Precision - ",precision_score(yactual, ypredicted))  
print("Recall - ",recall_score(yactual, ypredicted))
classification_report(yactual, ypredicted)
roc_auc_score(yactual, ypredicted)


# Confusion Matrix
plt.clf()
plt.imshow(conf_arr, interpolation='nearest', cmap=plt.cm.Wistia)
classNames = ['Not-Fall','Fall']
plt.title('Fall or Not-Fall Confusion Matrix - Balanced 10 features')
plt.ylabel('True label')
plt.xlabel('Predicted label')
tick_marks = numpy.arange(len(classNames))
plt.xticks(tick_marks, classNames, rotation=45)
plt.yticks(tick_marks, classNames)
s = [['TN','FP'], ['FN', 'TP']]
for i in range(2):
    for j in range(2):
        plt.text(j,i, str(s[i][j])+" = "+str(conf_arr[i][j]))
plt.show()

# ROC curve
fpr, tpr, threshold = roc_curve(yactual,ypredicted)
roc_auc = auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
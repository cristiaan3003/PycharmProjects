import csv
import numpy as np
from sklearn import preprocessing
from sklearn import svm


with open('featuresTest.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    lines = list(readCSV)
TODO_test=np.transpose(lines)
X_test=np.transpose(TODO_test[:-1])
Y_test=TODO_test[-1]

with open('featuresTrain.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    lines = list(readCSV)
TODO_train=np.transpose(lines)
X_train=np.transpose(TODO_train[:-1])
Y_train=TODO_train[-1]


scaler = preprocessing.MinMaxScaler(feature_range=(0,1)).fit(X_train)  # Built scaler to scale values in range [Xmin,Xmax]

# APPLY TRAINED SCALER
sX_train = scaler.transform(X_train)

sX_test = scaler.transform(X_test)

sX_train.mean(axis=0)

clf = svm.SVC(kernel='linear', gamma='auto', C=1, cache_size=500)


TRAIN = sX_train # --> Datos ESTANDARIZADOS


clf.fit(TRAIN, Y_train)


TEST = sX_test # --> Datos ESTANDARIZADOS


Y_pred = clf.predict(TEST)

#======================
# SCORE CALCULATION
#======================
score = clf.score(TEST, Y_test)

print(score)

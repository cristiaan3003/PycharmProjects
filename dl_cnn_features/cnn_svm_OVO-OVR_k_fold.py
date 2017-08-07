import numpy as np
from itertools import cycle
import matplotlib.pyplot as plt
import pandas
import pickle
import math
import time
import sensitivity_specifity
import csv
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize

from sklearn import svm
from sklearn import datasets
from sklearn import preprocessing


from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier


from sklearn import model_selection # <---- PARTICIONADO


from sklearn import metrics
from sklearn.metrics import confusion_matrix


#===================================================
#==========MAIN========================
#===================================================
if __name__ == "__main__":



    SCORE = []
    UAR = []
    F1_W = []
    PRESICION = []
    G_measure = []
    RECALL = []
    SPECIFITY = []
    k=10
    start_time = time.time()
    for i in range(10):
        print("---")
        with open("featuresTest"+str(i)+".csv") as csvfile:
            readCSV = csv.reader(csvfile, delimiter=',')
            lines = list(readCSV)
        TODO_test=np.transpose(lines)
        X_test=np.transpose(TODO_test[:-1])
        Y_test=TODO_test[-1]

        with open("featuresTrain"+str(i)+".csv") as csvfile:
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
        TRAIN = sX_train # --> Datos ESTANDARIZADOS
        TEST = sX_test # --> Datos ESTANDARIZADOS
        # Run classifier
        estimador=svm.SVC(kernel='linear',gamma='auto',cache_size=500,random_state=None)
        classifier2 = OneVsOneClassifier(estimador,-1)
        classifier2.fit(TRAIN, Y_train)#.decision_function(TEST)
        Y_pred2 = classifier2.predict(TEST)

        #======================
        # SCORE CALCULATION
        #======================
        scoreOVO=classifier2.score(TEST,Y_test)
        f1_weightedOVO=metrics.f1_score(Y_test, Y_pred2, labels=None, pos_label=1, average='weighted', sample_weight=None) #toma en cuenta el desbalance de etiqueta
        precision_scoreOVO=metrics.precision_score(Y_test, Y_pred2, labels=None, pos_label=1, average='weighted', sample_weight=None)
        recall_scoreOVO=metrics.recall_score(Y_test, Y_pred2, labels=None, pos_label=1, average='weighted', sample_weight=None)
        GOVO=math.sqrt(recall_scoreOVO*precision_scoreOVO)
        specifity=sensitivity_specifity.specificity_score(Y_test,Y_pred2,average='weighted')

        #======================
        # CM
        #======================
        cmOVO=metrics.confusion_matrix(Y_test,Y_pred2)
        MOVO = np.float64(cmOVO)
        #===============================
        # UAR CALCULATION
        #===============================

        uarOVO = np.diag(MOVO)/np.sum(MOVO,1)

        #===============================
        # SAVE MEASURES
        #===============================
        SCORE.append(scoreOVO)
        UAR.append(uarOVO)
        F1_W.append(f1_weightedOVO)
        PRESICION.append(precision_scoreOVO)
        G_measure.append(GOVO)
        RECALL.append(recall_scoreOVO)
        SPECIFITY.append(specifity)

    tiempo_total= time.time() - start_time
    #========================
    # REPORT
    #========================



    print('===========')
    print('   TEST CNN+OVO KFOLD  ')
    print('===========\n')
    score_mean=np.mean(SCORE)
    score_std=np.std(SCORE)
    uar_mean=np.mean(UAR)
    uar_std=np.std(UAR)
    f1_weighted_mean=np.mean(F1_W)
    f1_weighted_std=np.std(F1_W)
    presicion_mean=np.mean(PRESICION)
    presicion_std=np.std(PRESICION)
    G_mean=np.mean(G_measure)
    G_std=np.std(G_measure)
    recall_mean=np.mean(RECALL)
    recall_std=np.std(RECALL)
    specifity_mean=np.mean(SPECIFITY)
    specifity_std=np.std(SPECIFITY)
    print('ACCURACY(mean-std):'+str(score_mean)+" (+/-)"+str(score_std))
    print('UAR(mean-std):'+str(uar_mean)+" (+/-)"+str(uar_std))
    print('F1 WEHIGT(mean-std):'+str(f1_weighted_mean)+" (+/-)"+str(f1_weighted_std))
    print('PRESICION(mean-std):'+str(presicion_mean)+" (+/-)"+str(presicion_std))
    print('G Measure(mean-std):'+str(G_mean)+" (+/-)"+str(G_std))
    print('Recall(mean-std):'+str(recall_mean)+" (+/-)"+str(recall_std))
    print('Specifity(mean-std):'+str(specifity_mean)+" (+/-)"+str(specifity_std))
    print("Tiempo Total(min):"+str(tiempo_total/60))


import numpy as np
import matplotlib.pyplot as plt
import math
import csv
from sklearn import svm
from sklearn import preprocessing
from sklearn import metrics
import time
import sensitivity_specifity




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
    for i in range(k):
        print("---")
        with open("../Caracteristicas_Deep_learning/featuresTest"+str(i)+".csv") as csvfile:
            readCSV = csv.reader(csvfile, delimiter=',')
            lines = list(readCSV)
        TODO_test=np.transpose(lines)
        X_test=np.transpose(TODO_test[:-1])
        Y_test=TODO_test[-1]

        with open("../Caracteristicas_Deep_learning/featuresTrain"+str(i)+".csv") as csvfile:
            readCSV = csv.reader(csvfile, delimiter=',')
            lines = list(readCSV)
        TODO_train=np.transpose(lines)
        X_train=np.transpose(TODO_train[:-1])
        Y_train=TODO_train[-1]



        #----------------------------------------------
        # PREPROCESSING?? --> i.e. mean=0.0 - var=1.0
        #----------------------------------------------
        scaler = preprocessing.MinMaxScaler(feature_range=(0,1)).fit(X_train)  # Built scaler to scale values in range [Xmin,Xmax]
        # APPLY TRAINED SCALER
        sX_train = scaler.transform(X_train)

        sX_test = scaler.transform(X_test)

        sX_train.mean(axis=0)

        TRAIN = sX_train # --> Datos ESTANDARIZADOS

        TEST = sX_test # --> Datos ESTANDARIZADOS

        clf = svm.SVC(kernel='linear', gamma='auto', C=100, cache_size=500)
        #======================
        # TRAIN SVM
        #======================
        clf.fit(TRAIN,Y_train)

        #======================
        # SVM PREDICTION
        #======================
        Y_pred = clf.predict(TEST)


        #======================
        # SCORE CALCULATION
        #======================
        score = clf.score(TEST, Y_test)


        f1_weighted=metrics.f1_score(Y_test, Y_pred, labels=None, pos_label=1, average='weighted', sample_weight=None) #toma en cuenta el desbalance de etiqueta
        precision_score=metrics.precision_score(Y_test, Y_pred, labels=None, pos_label=1, average='weighted', sample_weight=None)
        recall_score=metrics.recall_score(Y_test, Y_pred, labels=None, pos_label=1, average='weighted', sample_weight=None)
        G=math.sqrt(recall_score*precision_score)
        specifity=sensitivity_specifity.specificity_score(Y_test,Y_pred,average='weighted')
        #===============================
        # CONFUSSION MATRIX CALCULATION
        #===============================
        cm=metrics.confusion_matrix(Y_test,Y_pred)
        plt.matshow(cm)
        #plt.title('Confusion matrix')
        #plt.colorbar()
        #plt.ylabel('True label')
        #plt.xlabel('Predicted label')
        #plt.show()
        M = np.float64(cm)

        #===============================
        # UAR CALCULATION
        #===============================
        uar = np.diag(M)/np.sum(M,1)


        #===============================
        # SAVE MEASURES
        #===============================
        SCORE.append(score)
        UAR.append(uar)
        F1_W.append(f1_weighted)
        PRESICION.append(precision_score)
        G_measure.append(G)
        RECALL.append(recall_score)
        SPECIFITY.append(specifity)

    tiempo_total= time.time() - start_time

    #========================
    # REPORT
    #========================



    print('===========')
    print('   TEST CNN+SVM KFOLD  ')
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

import numpy as np
import matplotlib.pyplot as plt
import pandas
import pickle
import math
import time

from sklearn import svm
from sklearn import preprocessing
import sensitivity_specifity
from sklearn import metrics
from sklearn.metrics import confusion_matrix


def read_csv_files(folder, Nfeatures=102):
    '''
    Load all the "CSV" files in a folder, and store each one in a dictionary.
    '''

    #filenames = glob.glob(rel_path + '*.csv')

    clase = dict()

    for n in range(0,24):

      clase[n] = pandas.read_csv(folder +'clase' + str(n) + '.csv', sep=',',header=None, names=['f'+str(m) for m in range(0,Nfeatures)])
#        clase[n+1] = np.array(pandas.read_csv(folder +'clase' + str(n+1) + '.csv', sep=',',header=None, names=['f'+str(m) for m in range(0,Nfeatures)]))
    return clase


#===================================================
#==========MAIN========================
#===================================================
if __name__ == "__main__":
    clase = read_csv_files('../Caracteristicas_Estandar/clases/',128)
    clases_a_usar = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]

    X = []
    Y = []

    for idx in clases_a_usar:

        x = np.array(clase[idx]).tolist()
        y = idx * np.ones((clase[idx].shape[0]))
        y = y.tolist()

        X.extend(x)
        Y.extend(y)

    #Cargo Indices del particionado K-fold previamente realizado
    filehandler = open('../Particionado_K-Fold/kfold.pki', 'rb')
    k_fold_indexs = pickle.load(filehandler)
    filehandler.close()

    X=np.array(X)
    Y=np.array(Y)

    SCORE = []
    UAR = []
    F1_W = []
    PRESICION = []
    G_measure = []
    RECALL = []
    SPECIFITY = []
    start_time = time.time()
    for fold in k_fold_indexs:
        print("---")
        #print("TRAIN:", k_fold_indexs[fold][0], "TEST:", k_fold_indexs[fold][1])
        X_train, X_test = X[k_fold_indexs[fold][0]], X[k_fold_indexs[fold][1]]
        Y_train, Y_test = Y[k_fold_indexs[fold][0]], Y[k_fold_indexs[fold][1]]



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

        clf = svm.SVC(kernel='linear', gamma='auto', C=1, cache_size=500)
        #gamma : float, optional (default=’auto’)
        #‘rbf’, ‘poly’ and ‘sigmoid’. If gamma is ‘auto’ then 1/n_features will be used instead.
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
        #plt.matshow(cm)
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
    print('   TEST  SVM KFOLD  ')
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

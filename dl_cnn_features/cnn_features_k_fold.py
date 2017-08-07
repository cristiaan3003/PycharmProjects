#Test CNN k-fold
#prueba de rendimiento de cnn completa (extracción y clasificación con el clasificador que trae
#implementado internamente tipo MPL)

import matplotlib.pyplot as plt
import math
import numpy as np
import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
import pandas
import pickle
from sklearn import metrics
import time
import sensitivity_specifity

def read_csv_files(folder, Nfeatures=2500):
    '''
    Load all the "CSV" files in a folder, and store each one in a dictionary.
    '''

    clase = dict()

    for n in range(0,24):

      clase[n] = pandas.read_csv(folder +'clase' + str(n) + '.csv', sep=',',header=None, names=['f'+str(m) for m in range(0,Nfeatures)])

    return clase




if __name__ == "__main__":
    clase = read_csv_files('clases/') #imagenes de 50x50px puestas el valor de sus pixel en archivo


    clases_a_usar = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]

    X = []
    Y = []

    for idx in clases_a_usar:

        x = np.array(clase[idx]).tolist()
        y = idx * np.ones((clase[idx].shape[0]))
        y = y.tolist()

        X.extend(x)
        Y.extend(y)

    clase=[]


    #Cargo Indices del particionado K-fold previamente realizado
    #de esta manera siempre puso los mismo datos para entrenamiento y test y puedo comparar
    filehandler = open('../Particionado_K-Fold/kfold.pki', 'rb')
    k_fold_indexs = pickle.load(filehandler)
    filehandler.close()

    #X_train = np.array(X_train)
    #X_train = X_train.astype(np.float32)

    X=np.array(X)
    #X= X.astype(np.float32)
    Y=np.array(Y)
    #Y= Y.astype(np.float32)


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
        print(fold)
        #print("TRAIN:", k_fold_indexs[fold][0], "TEST:", k_fold_indexs[fold][1])
        X_train, X_test = X[k_fold_indexs[fold][0]], X[k_fold_indexs[fold][1]]
        Y_train, Y_test = Y[k_fold_indexs[fold][0]], Y[k_fold_indexs[fold][1]]

        X_train = np.array(X_train)
        X_train = X_train.astype(np.float32)

        X_test = np.array(X_test)
        X_test = X_test.astype(np.float32)

        Y_train = np.array(Y_train)
        Y_train = Y_train.astype(np.float32)

        Y_test = np.array(Y_test)
        Y_test = Y_test.astype(np.float32)


        X_train = X_train.reshape((-1, 1, 50, 50))
        X_test = X_test.reshape((-1, 1, 50, 50))
        y_train = Y_train.astype(np.uint8)
        y_test = Y_test.astype(np.uint8)

        net1 = NeuralNet(
        layers=[('input', layers.InputLayer),
                ('conv2d1', layers.Conv2DLayer),
                ('maxpool1', layers.MaxPool2DLayer),
                ('conv2d2', layers.Conv2DLayer),
                ('maxpool2', layers.MaxPool2DLayer),
                ('dropout1', layers.DropoutLayer),
                ('dense', layers.DenseLayer),
                ('dropout2', layers.DropoutLayer),
                ('output', layers.DenseLayer),
                ],
        # input layer
        input_shape=(None, 1, 50, 50),
        # layer conv2d1
        conv2d1_num_filters=32,
        conv2d1_filter_size=(5, 5),
        conv2d1_nonlinearity=lasagne.nonlinearities.rectify,
        conv2d1_W=lasagne.init.GlorotUniform(),
        # layer maxpool1
        maxpool1_pool_size=(2, 2),
        # layer conv2d2
        conv2d2_num_filters=32,
        conv2d2_filter_size=(5, 5),
        conv2d2_nonlinearity=lasagne.nonlinearities.rectify,
        # layer maxpool2
        maxpool2_pool_size=(2, 2),
        # dropout1
        dropout1_p=0.5,
        # dense
        dense_num_units=128,
        dense_nonlinearity=lasagne.nonlinearities.rectify,
        # dropout2
        dropout2_p=0.5,
        # output
        output_nonlinearity=lasagne.nonlinearities.softmax,
        output_num_units=24,
        # optimization method params
        update=nesterov_momentum,
        update_learning_rate=0.01,
        update_momentum=0.9,
        max_epochs=30,
        verbose=1,
        )


        # Train the network
        net1.fit(X_train, y_train)


        #Prediction and Confusion Matrix
        Y_pred = net1.predict(X_test)

        #======================
        # SCORE CALCULATION
        #======================
        score = net1.score(X_test, Y_test)

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
    print('   TEST  FULL CNN ')
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

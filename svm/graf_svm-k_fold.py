import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import preprocessing
import sensitivity_specifity
from sklearn import metrics
import pandas
import pickle


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

    c = [0.1,0.5,1,5,10]
    '''clase = read_csv_files('../extraccionCaracteristicas/clases/',128)
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
    #de esta manera siempre puso los mismo datos para entrenamiento y test y puedo comparar
    filehandler = open('../Particionado_K-Fold/kfold.pki', 'rb')
    k_fold_indexs = pickle.load(filehandler)
    filehandler.close()

    X=np.array(X)
    Y=np.array(Y)
    train_prec =  []
    eval_prec = []

    for i in c:
        print(i)
        #print("TRAIN:", k_fold_indexs[fold][0], "TEST:", k_fold_indexs[fold][1])
        X_train, X_test = X[k_fold_indexs[0][0]], X[k_fold_indexs[0][1]]
        Y_train, Y_test = Y[k_fold_indexs[0][0]], Y[k_fold_indexs[0][1]]



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

        clf = svm.SVC(kernel='linear', gamma='auto', C=i, cache_size=500)
        #gamma : float, optional (default=’auto’)
        #‘rbf’, ‘poly’ and ‘sigmoid’. If gamma is ‘auto’ then 1/n_features will be used instead.
        #======================
        # TRAIN SVM
        #======================
        clf.fit(TRAIN,Y_train)
        Y_pred1 = clf.predict(TEST)
        cm1=metrics.confusion_matrix(Y_test,Y_pred1)
        M1 = np.float64(cm1)
        uar1 = np.diag(M1)/np.sum(M1,1)

        Y_pred2 = clf.predict(TRAIN)
        cm2=metrics.confusion_matrix(Y_train,Y_pred2)
        M2 = np.float64(cm2)
        uar2 = np.diag(M2)/np.sum(M2,1)

        train_prec.append( np.mean(uar2))
        eval_prec.append(np.mean(uar1))
    print(train_prec)
    print(eval_prec)'''


    train_prec=[0.87716125905688702, 0.89428635935483636, 0.9002958129278702, 0.9161302168534351, 0.92241101287254479]
    eval_prec=[0.85895992820731715, 0.87776569419900285, 0.88218164966637336, 0.8811766513677165, 0.87597417829003954]
    # graficar los resultados.
    plt.plot(c, train_prec, marker='o',color='r', label='entrenamiento')
    plt.plot(c, eval_prec, marker='x',color='b', label='evaluación')
    plt.title('Gráfico de ajuste SVM')
    plt.legend(loc=4)
    plt.axis([0.1,10,0.70,1.001])
    plt.xticks(c,c)
    #y_leyend=np.linspace(0.9,1,11)
    #plt.yticks(y_leyend, y_leyend)

    plt.ylabel('UAR')
    plt.xlabel('C')
    plt.show()

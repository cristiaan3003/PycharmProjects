import numpy as np
import pandas
import pickle
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics


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
    n_neighbors_list = list(range(1,11,1))
    '''clase = read_csv_files('../Caracteristicas_Estandar/clases/',128)
    #print(clase[0])
    #print(clase[1].shape)
    #print(clase[2].shape)
    #print(clase[23].shape)
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

    # Grafico de ajuste del árbol de decisión
    train_prec =  []
    eval_prec = []


    for nn in n_neighbors_list:
        print(nn)
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



        clf = KNeighborsClassifier(n_neighbors=nn, weights='distance',algorithm='auto', leaf_size=30,
        p=1, metric='minkowski',metric_params=None, n_jobs=-1)# Creando el modelo con 10 vecinos - manhattan_distance (l1)

        clf.fit(X_train,Y_train)
        Y_pred1 = clf.predict(X_test)
        cm1=metrics.confusion_matrix(Y_test,Y_pred1)
        M1 = np.float64(cm1)
        uar1 = np.diag(M1)/np.sum(M1,1)

        Y_pred2 = clf.predict(X_train)
        cm2=metrics.confusion_matrix(Y_train,Y_pred2)
        M2 = np.float64(cm2)
        uar2 = np.diag(M2)/np.sum(M2,1)

        eval_prec.append(np.mean(uar1))


    print(eval_prec)'''


    eval_prec=[0.84561058692523583, 0.84561058692523583, 0.85822859251780237, 0.86811731121449298, 0.86943331069148433, 0.87478165888880854, 0.87362716319914702, 0.875888048048859, 0.87615277910964762, 0.87514505059704295]

    # graficar los resultados.
    #plt.plot(n_neighbors_list, train_prec, color='r', label='entrenamiento')
    plt.plot(n_neighbors_list, eval_prec,marker='x', color='b', label='evaluación')
    plt.title('Gráfico de ajuste KNN')
    plt.legend(loc=4)
    plt.axis([1,10,0.80,1.01])
    plt.ylabel('UAR')
    plt.xlabel('Número de Vecinos')
    plt.show()

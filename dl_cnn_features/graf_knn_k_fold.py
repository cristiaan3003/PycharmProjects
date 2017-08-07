import numpy as np
import pandas
import csv
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
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

    print("---")
    with open("featuresTest0.csv") as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        lines = list(readCSV)
    TODO_test=np.transpose(lines)
    X_test=np.transpose(TODO_test[:-1])
    Y_test=TODO_test[-1]

    with open("featuresTrain0.csv") as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        lines = list(readCSV)
    TODO_train=np.transpose(lines)
    X_train=np.transpose(TODO_train[:-1])
    Y_train=TODO_train[-1]

    X=np.array(X)
    Y=np.array(Y)

    # Grafico de ajuste del árbol de decisión
    train_prec =  []
    eval_prec = []


    for nn in n_neighbors_list:
        print(nn)

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

        train_prec.append( np.mean(uar2))
        eval_prec.append(np.mean(uar1))

    print(eval_prec)'''


    #train_prec=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    eval_prec=[0.94186769381939162, 0.94186769381939162, 0.94806936119576013, 0.94706012224656122, 0.9510719863803252, 0.95216247173356372, 0.95000831504253702, 0.95241900219274422, 0.95084854063486313, 0.95031219663257482]

     # graficar los resultados.
    #plt.plot(n_neighbors_list, train_prec, color='r', label='entrenamiento')
    plt.plot(n_neighbors_list, eval_prec,marker='x', color='b', label='evaluación')
    plt.title('Grafico de ajuste CNN+KNN')
    plt.legend(loc=4)
    plt.axis([1, 10, 0.9, 1.001])
    plt.ylabel('UAR')
    plt.xlabel('Número de vecinos')
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import pandas
import csv
from sklearn import svm
from sklearn import preprocessing

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

    for i in c:
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

        clf.fit(TRAIN,Y_train)

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

    print(train_prec)
    print(eval_prec)'''


    train_prec=[0.99743062675553873, 0.99922859275307074, 0.99954852087454593, 0.99979478588883286, 0.99987688769507255]
    eval_prec=[0.93904461376318971, 0.94665908108042529, 0.94841109133727974, 0.94536067116921252, 0.94097747326195036]


     # graficar los resultados.
    plt.plot(c, train_prec, marker='o',color='r', label='entrenamiento')
    plt.plot(c, eval_prec, marker='x',color='b', label='evaluación')
    plt.title('Gráfico de ajuste CNN+SVM')
    plt.legend(loc=4)
    plt.axis([0.1,10,0.90,1.001])
    plt.xticks(c,c)
    #y_leyend=np.linspace(0.9,1,11)
    #plt.yticks(y_leyend, y_leyend)

    plt.ylabel('UAR')
    plt.xlabel('C')
    plt.show()

import numpy as np
import pandas
from sklearn import metrics
import pickle
import csv
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import linear_model
from sklearn_extensions.extreme_learning_machines.elm import ELMClassifier


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
    n_hidden_list = [100,500,1000,1500,2000,2500,3000,3500,4000]
    '''
    clase = read_csv_files('../extraccionCaracteristicas/clases/',128)

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


    for nh in n_hidden_list:
        print(nh)
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



        clf=ELMClassifier(n_hidden=nh,alpha=0.95,activation_func='multiquadric',activation_args=None,
                      user_components=None,regressor=linear_model.Ridge(),random_state=None)

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
    print(eval_prec)

    '''
    train_prec=[ 0.95096975989192767, 0.97546584402587644, 0.9788469025794525, 0.98034074242129721, 0.97979709822629746, 0.98024416184854157, 0.9810483692710239, 0.98074998887857578, 0.98036366555116228]
    eval_prec=[ 0.92742377527863351, 0.94783067750412064, 0.94924527712024276, 0.9504416873009961, 0.95281345949519081, 0.94423100096166934, 0.95066652730741297, 0.95423021271712294, 0.95159573591173485]

     # graficar los resultados.
    plt.plot(n_hidden_list, train_prec, marker='o',color='r', label='entrenamiento')
    plt.plot(n_hidden_list, eval_prec, marker='x',color='b', label='evaluación')
    plt.title('Gráfico de ajuste CNN+ELM')
    plt.legend(loc=4)
    plt.axis([100,4000,0.80,1.01])
    plt.xticks(n_hidden_list, n_hidden_list)
    y_leyend=np.linspace(0.5,1,11)
    plt.yticks(y_leyend, y_leyend)

    plt.ylabel('UAR')
    plt.xlabel('Número de Neuronas en la Capa Oculta')
    plt.show()

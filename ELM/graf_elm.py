import numpy as np
import pandas
import pickle
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import linear_model
from sklearn_extensions.extreme_learning_machines.elm import ELMClassifier
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
    n_hidden_list = [100,500,1000,1500,2000,2500,3000,3500,4000]

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

    for nh in n_hidden_list:
        print(nh)
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
    print(eval_prec)'''

    n_hidden_list = [100,500,1000,1500,2000,2500,3000,3500,4000]
    train_prec=[0.8017544865024947, 0.87057050399288094, 0.89539579859533769, 0.91421756394455667, 0.922440297992138, 0.92976313232790941, 0.93799672371513376, 0.94002612601064639, 0.94652659217247004]
    eval_prec=[0.78908229453183365, 0.85158423576850117, 0.86717423936579774, 0.88087536483199436, 0.88416489997090719, 0.89139561819717306, 0.89523008361156153, 0.89579145789890091, 0.89487568849061683]

     # graficar los resultados.
    plt.plot(n_hidden_list, train_prec,marker='o', color='r', label='entrenamiento')
    plt.plot(n_hidden_list, eval_prec,marker='x', color='b', label='evaluación')
    plt.title('Gráfico de ajuste ELM ')
    plt.legend(loc=4)
    plt.axis([100,4000,0.4,1.01])
    plt.xticks(n_hidden_list,n_hidden_list)
    #y_leyend=np.linspace(0.5,1,11)
    #plt.yticks(y_leyend, y_leyend)
    plt.ylabel('UAR')
    plt.xlabel('Número de Neuronas en la Capa Oculta')
    plt.show()

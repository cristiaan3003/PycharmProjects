import numpy as np
import pandas
import pickle
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
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

    max_deep_list=[1,3,5,10,15,20,25,30,35,40]  #numero de arboles - prof fija=15
    #max_deep_list=[1,3,5,10,15,20,25,30,35,40] #profundidad maxima - arboles fijo =30

    '''clase = read_csv_files('../Caracteristicas_Estandar/clases/',128)
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

    for deep in max_deep_list:
        print(deep)
        X_train, X_test = X[k_fold_indexs[0][0]], X[k_fold_indexs[0][1]]
        Y_train, Y_test = Y[k_fold_indexs[0][0]], Y[k_fold_indexs[0][1]]


        clf = RandomForestClassifier(n_estimators=30, criterion='gini', max_depth=deep,
        min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto',
        max_leaf_nodes=None, min_impurity_split=1e-07, bootstrap=True, oob_score=False, n_jobs=-1,
        random_state=False, verbose=0, warm_start=False, class_weight=None)
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

    #arboles
    train_prec=[0.84104218869686187, 0.94110122803702223, 0.97800885602290022, 0.99259484525761421, 0.99504996906006848, 0.99647053105902772, 0.997084650615086, 0.99772185621750398, 0.99751649211656035, 0.99768082534368607]
    eval_prec=[0.64136202274739873, 0.73972676801163095, 0.81349375783277356, 0.85674275158960478, 0.87275837958816671, 0.8762938161720436, 0.87903429803526467, 0.87911245145166961, 0.88412528371881605, 0.88435838746612838]

    #profundidad
    #train_prec=[0.34895879896335474, 0.62911096235531805, 0.7565850220705691, 0.93613018057563091, 0.99772185621750398, 0.99982381865868941, 0.99995898950131235, 0.99995898950131235, 0.99995898950131235, 0.99995898950131235]
    #eval_prec=[0.34980300577152046, 0.61783027261740342, 0.74464112892057288, 0.86598321343238904, 0.87911245145166961, 0.89053667700135042, 0.89067953284753199, 0.89691053589699477, 0.89691053589699477, 0.89691053589699477]


    # graficar los resultados.
    plt.plot(max_deep_list, train_prec,marker='o', color='r', label='entrenamiento')
    plt.plot(max_deep_list, eval_prec,marker='x', color='b', label='evaluación')
    plt.title('Gráfico de ajuste Random Forest - Arboles')
    #plt.title('Gráfico de ajuste Random Forest - Profundidad')
    plt.legend(loc=4)
    plt.axis([1,40,0.30,1.01])
    plt.xticks(max_deep_list,max_deep_list)
    #y_leyend=np.linspace(0.5,1,11)
    #plt.yticks(y_leyend, y_leyend)
    plt.ylabel('UAR')
    plt.xlabel('Número de Arboles')
    #plt.xlabel('Profundidad')
    plt.show()

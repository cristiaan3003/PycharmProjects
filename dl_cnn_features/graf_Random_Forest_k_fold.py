import numpy as np
import pandas
import csv
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
    max_deep_list = [1,3,5,10,15,20,25,30,35,40] #arboles variable
    #max_deep_list = [1,3,5,10,15,20,25,30,35,40] #profundidad

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

    for deep in max_deep_list:
        print(deep)

        clf = RandomForestClassifier(n_estimators=deep, criterion='gini', max_depth=10,
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
    print(eval_prec)

    '''
    #ACCURACY
    #prof variable 1-40([1,3,5,10,15,20,25,30,35,40]) , arboles fijo 30
    #train_prec=[0.664568268821778, 0.91080391322841348, 0.95448745214802211, 0.98774989366227139, 0.99906422798809014, 1.0, 0.99995746490854953, 0.99995746490854953, 0.99995746490854953, 0.99995746490854953]
    #eval_prec=[0.64102564102564108, 0.88289322617680832, 0.92690394182931501, 0.94871794871794868, 0.95063145809414462, 0.94910065059318793, 0.94986605434366633, 0.95063145809414462, 0.95063145809414462, 0.95063145809414462]

    #max_deep_list = [1,3,5,10,15,20,25,30,35,40]
    #prof fije = 15, numero arboles= [1,3,5,10,15,20,25,30,35,40]
    #train_prec=[0.94623564440663543, 0.98298596341982136, 0.99213100808166732, 0.99732028923862182, 0.99834113143343262, 0.99863887707358567, 0.99897915780518931, 0.99906422798809014, 0.99902169289663978,0.99902169289663978]
    #eval_prec=[0.85993111366245689, 0.90968235744355153, 0.9276693455797933, 0.93876769996172982, 0.9433601224646001, 0.94718714121699199, 0.94910065059318793, 0.95063145809414462, 0.95063145809414462,0.95063145809414462]

    #-------------------------------
    #------------------------------
    #UAR
    #prof variable 1-40([1,3,5,10,15,20,25,30,35,40]) , arboles fijo 30
    train_prec=[0.63611466425288199, 0.88008307291958332, 0.93900381726136661, 0.98608854514186906, 0.9988704414566717, 1.0, 0.99995966440787354, 0.99995966440787354, 0.99995966440787354, 0.99995966440787354]
    eval_prec=[0.62179873155881416, 0.8510551331581081, 0.90890829975521126, 0.94574346602320158, 0.94621021598515853, 0.94668182055321692, 0.94783584997938009, 0.95001478704155262, 0.95008509963164378, 0.95008509963164378]

    #max_deep_list = [1,3,5,10,15,20,25,30,35,40] , profundidad fija 10
    train_prec=[0.89744933738697341, 0.9527118481161656, 0.96902330672042891, 0.9787462639576624, 0.98222701395988521, 0.98431661311188501, 0.98520654269909691, 0.98608854514186906, 0.98727140700173732, 0.98772335998547722]
    eval_prec=[0.83466102446621626, 0.89120122369033139, 0.91633857354341741, 0.93254704040608349, 0.94144868078504729, 0.94087909106465428, 0.94499399626594982, 0.94574346602320158, 0.94612327914249006, 0.94779182155395691]



     # graficar los resultados.
    plt.plot(max_deep_list, train_prec,marker='o', color='r', label='entrenamiento')
    plt.plot(max_deep_list, eval_prec,marker='x', color='b', label='evaluación')
    plt.title('Gráfico de ajuste CNN+RandomForest - Arboles Variable')
    plt.legend(loc=4)
    plt.axis([1, 40, 0.50, 1.01])
    plt.xticks(max_deep_list, max_deep_list)
    #y_leyend=np.linspace(0.5,1,11)
    #plt.yticks(y_leyend, y_leyend)

    plt.ylabel('UAR')
    plt.xlabel('Número de Arboles en el Bosque')
    #plt.xlabel('Profundidad Maxima')
    plt.show()'''


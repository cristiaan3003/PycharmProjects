import numpy as np
import pandas
import pickle
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
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
    max_deep_list = list(range(1, 21))
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


        clf = DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=deep,
                                        min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
                                        max_features=None, random_state=0, max_leaf_nodes=None,
                                        min_impurity_split=1e-07, class_weight=None, presort=False)
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

    train_prec=[0.083212909441233149, 0.15621112383968905, 0.2789390810433664, 0.46695502870425765, 0.59541283606796414, 0.70874940752012827, 0.75290486906682441, 0.79571390201908676, 0.84067685102126644, 0.89005513085686949, 0.93246769914054095, 0.96471174761263978, 0.98661328184718611, 0.99608814136141788, 0.99869558915556811, 0.99959093256087872, 1.0, 1.0, 1.0, 1.0]
    eval_prec=[0.082903780068728519, 0.15510867808661924, 0.28190032884929828, 0.47211113141800953, 0.59383538745027653, 0.69052268282765239, 0.7292545549463707, 0.76166884267299861, 0.77167915913810792, 0.77457538853706875, 0.78691570224492102, 0.778636517685102, 0.78664446584490422, 0.78512458170624744, 0.78138954175622943, 0.78427622449264678, 0.78264883373767979, 0.78264883373767979, 0.78264883373767979, 0.78264883373767979]



    # graficar los resultados.
    plt.plot(max_deep_list, train_prec,marker='o', color='r', label='entrenamiento')
    plt.plot(max_deep_list, eval_prec,marker='x', color='b', label='evaluación')
    plt.title('Gráfico de ajuste Arbol de Decisión')
    plt.legend(loc=4)
    plt.axis([1, 20, 0, 1.01])
    plt.xticks(max_deep_list, max_deep_list)
    plt.ylabel('UAR')
    plt.xlabel('Profundidad Maxima')
    plt.show()

import numpy as np
import pandas
import csv
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

    #ACC
    #train_prec=[0.087324542747766909, 0.1690769885155253, 0.32649936197362822, 0.55806039982985967, 0.75648660144619306, 0.88000850701829014, 0.89953211399404509, 0.92228838792003398, 0.9466609953211399, 0.96737558485750741, 0.9814121650361548, 0.99042960442364947, 0.99523606975754997, 0.99812845597618038, 0.99927690344534237, 0.99982985963419824, 1.0, 1.0, 1.0,1.0]
    #eval_prec=[0.077688480673555299, 0.17145044010715652, 0.32300038270187525, 0.55530042097206278, 0.73861461921163418, 0.85227707615767312, 0.86452353616532718, 0.88059701492537312, 0.89552238805970152, 0.9008802143130501, 0.90623804056639878, 0.90662074244163793, 0.9016456180635285, 0.9039418293149637, 0.90508993494068124, 0.90279372368924604, 0.90049751243781095, 0.90049751243781095, 0.90049751243781095,0.90049751243781095]
    #UAR
    train_prec=[0.082849267142939331, 0.16223209627721921, 0.3128842088187867, 0.53544762298257764, 0.74114209681400511, 0.87611956750808384, 0.89396768861181641, 0.91922039633885466, 0.94427401145564838, 0.96591672677027407, 0.98048357434906352, 0.9904230280577373, 0.99512280511256657, 0.99800697026502661, 0.99913566205094784, 0.99972219391180717, 1.0, 1.0, 1.0, 1.0]
    eval_prec=[0.082107843137254902, 0.16139369484777474, 0.30827904953956015, 0.52691891350257336, 0.72394820500221702, 0.84956935721884275, 0.86152839237295897, 0.87495147345440138, 0.88894152221215572, 0.89850272192544212, 0.90249141510467423, 0.90184768528745229, 0.89687022973773267, 0.89894858389523458, 0.9012680449621705, 0.89891569684247929, 0.89590233056820046, 0.89590233056820046, 0.89590233056820046, 0.89590233056820046]


     # graficar los resultados.
    plt.plot(max_deep_list, train_prec,marker='o', color='r', label='entrenamiento')
    plt.plot(max_deep_list, eval_prec,marker='x', color='b', label='evaluación')
    plt.title('Gráfico de ajuste CNN+Arbol de Decisión')
    plt.legend(loc=4)
    plt.axis([1, 20, 0, 1.01])
    #leyends=",".join(str(i) for i in max_deep_list)
    #print(leyends)
    plt.xticks(max_deep_list, max_deep_list)
    y_leyend=np.linspace(0,1,11)
    plt.yticks(y_leyend, y_leyend)
    plt.ylabel('UAR')
    plt.xlabel('Profundidad Maxima')
    plt.show()



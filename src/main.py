import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import f1_score, precision_recall_curve, average_precision_score, roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import tree
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

# ---------------------------------------------------
# -------------------- APARTAT B --------------------
# ---------------------------------------------------
"""
# import some data to play with
iris = datasets.load_iris()

# Take the first two features. We could avoid this by using a two-dim dataset
X = iris.data[:, :2]
y = iris.target



"""
n_classes = 2
def curve_generator(name, probs):
    plt.figure()
    precision = {};
    recall = {};
    average_precision = {}
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_data == i, probs[:, i])
        average_precision[i] = average_precision_score(y_data == i, probs[:, i])
        plt.plot(recall[i], precision[i],
                 label='Precision-recall curve of class {} (area = {})'
                       ''.format( i, round(average_precision[i],2)))
        plt.xlabel('Recall');
        plt.ylabel('Precision');
        plt.legend()
    plt.title('{}'.format(name))
    plt.show()
    fpr = {}; tpr = {}; roc_auc = {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_data == i, probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    # Compute micro-average ROC curve and ROC area
    # Plot ROC curve
    plt.figure()
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label='ROC curve of class {} (area = {})' ''.format( i, round(roc_auc[i],2)))
    plt.legend()
    plt.title('{}'.format(name))
    plt.show()

"""


# -------------------- CORBES PRCISION-RECALL --------------------
x_t, x_v, y_t, y_v = train_test_split(X, y, train_size=0.8)

# Creem el regresor logístic  Logistic
logireg = LogisticRegression(C=2.0, fit_intercept=True, penalty='l2', tol=0.001)
# l'entrenem
logireg.fit(x_t, y_t)
probs = logireg.predict_proba(x_v)
print("Correct classification Logistic ", 0.8, "% of the data: ", logireg.score(x_v, y_v))
curve_generator('Logistic Regressor', probs)

# Creem el regresor logístic SVM
svc = svm.SVC(C=10.0, kernel='rbf', gamma=0.9, probability=True)
# l'entrenem
svc.fit(x_t, y_t)
probs = svc.predict_proba(x_v)
print("Correct classification SVM      ", 0.8, "% of the data: ", svc.score(x_v, y_v))
curve_generator('SVM', probs)

# KNN
knn = KNeighborsClassifier()
knn.fit(x_t, y_t)
probs = knn.predict_proba(x_v)
curve_generator('KNN', probs)

# LDA
lda = LinearDiscriminantAnalysis()
lda.fit(x_t, y_t)
probs = lda.predict_proba(x_v)
curve_generator('LDA', probs)

# CART
cart = DecisionTreeClassifier()
cart.fit(x_t, y_t)
probs = cart.predict_proba(x_v)
curve_generator('CART', probs)

# NB
nb = GaussianNB()
nb.fit(x_t, y_t)
probs = nb.predict_proba(x_v)
curve_generator('Naive Bayes', probs)


# -------------------- K-FOLD: ACCURACY, F1_MACRO I F1_MICRO--------------------

models = []
models.append (('Logistic Regression', LogisticRegression(solver ='lbfgs',  multi_class = 'ovr')))
models.append (('Linear Discriminant Analysis', LinearDiscriminantAnalysis()))
models.append (('K Nearest Neigbors', KNeighborsClassifier()))
models.append (('CART', DecisionTreeClassifier()))
models.append (('Support Vector Machine', SVC(gamma ='scale')))
models.append (('Guassian Naive Bayes', GaussianNB()))

seed=6
resultat=[]
scoring=['Accuracy','F1_macro','F1_micro']
import random
plt.figure()
for type_score in scoring:
    for index, (name, model) in enumerate(models):
        res_tmp = []
        for i in range(2, 20):
            K_Fold = model_selection.KFold (n_splits = i, random_state  = random.randint(0,99), shuffle=True)
            cv_results = model_selection.cross_val_score (model, X, y, cv = K_Fold, scoring = type_score.lower())
            message =  "%s:  %f  (%f)" % (name, cv_results.mean (), cv_results.std())
            print (message)
            res_tmp.append(cv_results.mean())
        resultat.append(res_tmp)
        plt.plot(range(2,20),res_tmp, label='{}'.format(name))
        plt.ylim(0.6,1)
    plt.legend()
    plt.xlabel('Folds count')
    plt.ylabel('{}'.format(type_score))
    plt.title('Model {} by K-fold'.format(type_score))
    plt.savefig("../figures/model_{}_kfoldB".format(type_score))
    plt.show()

z=3
"""

"""
def make_meshgrid(x, y, h=.02):
    #Create a mesh of points to plot in
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    #Plot the decision boundaries for a classifier.
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


def show_C_effect(C=1.0, gamma=0.7, degree=3):
    iris = datasets.load_iris()
    X = iris.data[:, :2]
    y = iris.target
    titles = ('SVC with linear kernel',
              'LinearSVC (linear kernel)',
              'SVC with RBF kernel',
              'SVC with polynomial (degree {}) kernel'.format(degree))
    # C = 1.0  # SVM regularization parameter
    models = (svm.SVC(kernel='linear', C=C),
              svm.LinearSVC(C=C, max_iter=1000000),
              svm.SVC(kernel='rbf', gamma=gamma, C=C),
              svm.SVC(kernel='poly', degree=degree, gamma='auto', C=C))
    models = (clf.fit(X, y) for clf in models)
    plt.close('all')
    fig, sub = plt.subplots(2, 2, figsize=(14, 9))
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)
    for clf, title, ax in zip(models, titles, sub.flatten()):
        plot_contours(ax, clf, xx, yy,
                      cmap=plt.cm.coolwarm, alpha=0.8)
        ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xlabel('Sepal length')
        ax.set_ylabel('Sepal width')
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(title)
    plt.suptitle("Execution parameters: C = {}, gamma = {}, degree = {}".format(C, gamma, degree))
    plt.savefig("../figures/SVC_params-C_{}-g_{}-d_{}.png".format(C, gamma, degree))
    plt.show()


cs = [0.1, 1, 10, 100, 1000]
for c in cs:
    show_C_effect(C=c)

gammas = [0.1, 1, 10, 100]
for g in gammas:
    show_C_effect(gamma=g)

degrees = [0, 1, 2, 3, 4, 5, 6]
for deg in degrees:
    show_C_effect(degree=deg)
"""

# ---------------------------------------------------
# -------------------- APARTAT A --------------------
# ---------------------------------------------------
def load_dataset(path):
    return pd.read_csv(path, header=0, delimiter=",")

wp_dataset = load_dataset("../data/water_potability.csv")

# +--------------------------+
# | VISUALITZACIÓ INFORMACIÓ |
# +--------------------------+
# Mostra els primers 5 registres dels DataFrames
def print_head(dataset):
    print("HEAD del dataset de Water Potability")
    print(dataset.head())
    print("------------------------------------")

#print_head(wp_dataset)


# Funció que mostra per consola els tipus de dades de les característiques del DataFrame.
def print_data_types():
    print("------------------------------------")
    print("Tipus de dades")
    print(wp_dataset.dtypes)
    print("------------------------------------")

#print_data_types()

# Funció que mostra la dimensionalitat del DataFrame
def df_dimensionality(dataset):
    data = dataset.values
    # separem l'atribut objectiu Y de les caracterísitques X
    x_data = data[:, :-1]  # Característiques
    y_data = data[:, -1]  # Variable objectiu (target)
    print("Dimensionalitat del DataFrame: {}:".format(dataset.shape))
    print("Dimensionalitat de les característiques (X): {}".format(x_data.shape))
    print("Dimensionalitat de la variable objectiu (Y): {}".format(y_data.shape))
    print("------------------------------------")


#df_dimensionality(wp_dataset)

def y_balance(dataset):
    ax = sns.countplot(x="Potability", data=dataset, palette={0: 'firebrick', 1: "cornflowerblue"})
    plt.suptitle("Target attribute distribution (Water potability)")
    label = ["Non Potable", "Potable"]
    ax.bar_label(container=ax.containers[0], labels=label)
    plt.xlabel('Potability')
    plt.ylabel('Number of samples')
    plt.savefig("../figures/distribucio_atribut_objectiu.png")
    plt.show()

    porc_pot = (len(dataset[dataset.Potability == 1]) / len(dataset.Potability)) * 100
    print('The percentage of waters that are potable is: {:.2f}%'.format(porc_pot))

#y_balance(wp_dataset)


# +-----------------------+
# | CORRELACIÓ D'ATRIBUTS |
# +-----------------------+
# Funció que genera la matriu de correlació de Pearson d'un DataFrame i genera el plot
def pearson_correlation(dataset):
    plt.figure()
    fig, ax = plt.subplots(figsize=(20, 10))  # figsize controla l'amplada i alçada de les cel·les de la matriu
    plt.title("Matriu de correlació de Pearson")
    sns.heatmap(dataset.corr(), annot=True, linewidths=.5, ax=ax)
    plt.savefig("../figures/pearson_correlation_matrix_.png")
    plt.show()


#pearson_correlation(wp_dataset)
def histogrames(dataset):
    plt.figure()

    plt.title("Histogrames")
    relacio = sns.pairplot(dataset)
    plt.savefig("../figures/histograms_matrix_.png")
    plt.show()


#histogrames(wp_dataset)



# +-----------------------+
# | TRACTAMENT D'ATRIBUTS |
# +-----------------------+

# Funció que substitueix els valors nuls del dataset pel valor numèric '0'.
def nan_treatment(dataset):
    print("Eliminació 'NaN' del DataFrame")
    print(dataset.isnull().sum())
    print("------------------------------------")
    dataset['ph'] = dataset['ph'].fillna(dataset.groupby(['Potability'])['ph'].transform('mean'))
    dataset['Sulfate'] = dataset['Sulfate'].fillna(dataset.groupby(['Potability'])['Sulfate'].transform('mean'))
    dataset['Trihalomethanes'] = dataset['Trihalomethanes'].fillna(dataset.groupby(['Potability'])['Trihalomethanes'].transform('mean'))
    print("------------------------------------")
    print("Després de l'eliminació 'NaN' del DataFrame")
    print(dataset.isnull().sum())
    return dataset

wp_dataset = nan_treatment(wp_dataset)

# Funció que estandarditza els valors del DataFrame, per tal de permetre fer que les diferents
# característiques siguin comparables entre elles.
def standardize_mean(dataset):
    return  MinMaxScaler().fit_transform(dataset)
    #return (dataset - dataset.mean(0)) / dataset.std(0)

dataset_norm = standardize_mean(wp_dataset)

#pearson_correlation(dataset_norm)

#roc mide como de separados estan las clases Falsos positives/falsos negativos y  ROC sube rápida,
#precision recall decae
"""
Leave one Out es hacer KFold con N.
KFold separa uno en test 
    Fa todas las combinaciones de Ks elementos el conjunto de entrenamiento y luego hace media y desviación estandard
Random  y grid permite poner un badget , lesforc si fiquem el mateix dels dos sigui similar
"""
wp_data = dataset_norm
x_data = wp_data[:, :-1]  # Característiques
y_data = wp_data[:, -1]  # Variable objectiu (target)
x_t, x_v, y_t, y_v = train_test_split(x_data, y_data, train_size=0.7)
#x_t=x_v=x_data
#y_t=y_v=y_data

def SVM():


    svc = svm.SVC(C=10.0, kernel='rbf', gamma=0.9, probability=True) #tol=0.001

    # l'entrenem
    svc.fit(x_t, y_t)
    probs = svc.predict_proba(x_v)

    print("Correct classification SVM      ", 0.8, "% of the data: ", svc.score(x_v, y_v))

#SVM()

def Logistic_Regressor():
    logireg = LogisticRegression(C=2.0, fit_intercept=True, penalty='l2', tol=0.001)
    # l'entrenem
    logireg.fit(x_t, y_t)
    probs = logireg.predict_proba(x_v)
    print("Correct classification Logistic Regression      ", 0.8, "% of the data: ", logireg.score(x_v, y_v))


#Logistic_Regressor()

def Naive_Bayes(): #var_smoothing default=1e-9
    nb = GaussianNB()
    # l'entrenem
    nb.fit(x_t, y_t)
    probs = nb.predict_proba(x_v)
    print("Correct classification Naive Bayes     ", 0.8, "% of the data: ", nb.score(x_v, y_v))

#Naive_Bayes()


def Linear_Discriminant(): #solver default=’svd’ ,shrinkage default=None, n_components default=None, tol default=1.0e-4
    lda = LinearDiscriminantAnalysis()
    # l'entrenem
    lda.fit(x_t, y_t)
    probs = lda.predict_proba(x_v)
    print("Correct classification Linear_Discriminant     ", 0.8, "% of the data: ", lda.score(x_v, y_v))

#Linear_Discriminant()


def Decision_Tree(): #criterion='gini', max_depth=None, min_samples_leaf=1,  random_state=None,
    cart = DecisionTreeClassifier()
    # l'entrenem
    #cart.fit(x_t, y_t)
    scores = model_selection.cross_val_predict(cart, x_data, y_data, cv=4, method='predict_proba')
    #probs = cart.predict_proba(x_v)
    curve_generator('Decision Tree', scores)
    plt.figure()

    #tree.plot_tree(cart)
    plt.savefig("../figures/arbre.png")
    plt.show()

    print("Correct classification Decision_Tree     ", 0.8, "% of the data: ", cart.score(x_v, y_v))

Decision_Tree()

def KNN(): #n_neighbors=5,  weights='uniform', algorithm='auto', metric='minkowski',
    knn = KNeighborsClassifier()
    # l'entrenem
    knn.fit(x_t, y_t)
    probs = knn.predict_proba(x_v)
    print("Correct classification KNN     ", 0.8, "% of the data: ", knn.score(x_v, y_v))

#KNN()

"""
models = []
models.append(('SVM', svm.SVC(C=1.0, kernel='rbf', gamma=0.7, probability=True)))
models.append (('Logistic Regression', LogisticRegression(C=2.0, fit_intercept=True, penalty='l2', tol=0.001)))
models.append (('Guassian Naive Bayes', GaussianNB()))
models.append (('Linear Discriminant Analysis', LinearDiscriminantAnalysis()))
models.append (('CART', DecisionTreeClassifier()))
models.append (('K Nearest Neigbors', KNeighborsClassifier()))


i_index=[2,3,4,6,10,20,40,60]

for index, (name, model) in enumerate(models):

        for i in i_index:
            K_Fold = model_selection.KFold (n_splits = i, shuffle=True)
            cv_results = model_selection.cross_val_score (model, x_data, y_data, cv = K_Fold, scoring = "accuracy")
            message =  "%s (%f):  %f  (%f)" % (name, i,cv_results.mean(), cv_results.std())
            print (message)




"""

from sklearn import datasets
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import StackingClassifier
"""
cvs=[2,5,10,20,40,60]

for cv_triat in cvs:
    clf = AdaBoostClassifier(n_estimators=150)
    scores = model_selection.cross_val_score(clf, x_data, y_data, cv=cv_triat)
    print("Accuracy: %f (+/- %0.2f) [%s(%f)]" % (scores.mean(), scores.std(),"Ada Boost",cv_triat))

    clf = RandomForestClassifier( n_estimators=218, random_state=178,n_jobs=-1)
#clf.fit(x_t, y_t)
#probs = clf.predict_proba(x_v)
#print("Correct classification Random Forest     ", 0.7, "% of the data: ", clf.score(x_v, y_v))
    scores = model_selection.cross_val_score(clf, x_data, y_data, cv=cv_triat)
    print("Accuracy: %f (+/- %0.2f) [%s(%f)]" % (scores.mean(), scores.std(),"Random Forest",cv_triat))

    clf = HistGradientBoostingClassifier(max_iter=100)
    scores = model_selection.cross_val_score(clf, x_data, y_data, cv=cv_triat)
    print("Accuracy: %f (+/- %0.2f) [%s(%f)]" % (scores.mean(), scores.std(),"HistGradBoost",cv_triat))

    clf = ExtraTreesClassifier(n_estimators=100, max_depth=None, min_samples_split=2, random_state=0)
    scores = model_selection.cross_val_score(clf, x_data, y_data, cv=cv_triat)
    print("Accuracy: %f (+/- %0.2f) [%s(%f)]" % (scores.mean(), scores.std(),"ExtraTrees",cv_triat))



    bagging = BaggingClassifier(DecisionTreeClassifier(), max_samples=0.9, max_features=0.9)
    scores = model_selection.cross_val_score(bagging, x_data, y_data, cv=cv_triat)
    print("Accuracy: %f (+/- %0.2f) [%s(%f)]" % (scores.mean(), scores.std(),"Bagging",cv_triat))




    estimator = [('rf', RandomForestClassifier(n_estimators=100, random_state=24)),('hgb', HistGradientBoostingClassifier(max_iter=100)), ('CART', DecisionTreeClassifier())]
    clf = StackingClassifier(estimators=estimator, final_estimator=GaussianNB())
#clf.fit(x_t, y_t).score(x_v, y_v)
    scores = cross_val_score(clf, x_data, y_data, scoring='accuracy', cv=cv_triat)
    print("Accuracy: %f (+/- %0.2f) [%s(%f)]" % (scores.mean(), scores.std(), "Stacking", cv_triat))



    eclf = VotingClassifier(estimators=[('bag',
                                 BaggingClassifier(base_estimator=DecisionTreeClassifier(),
                                                    max_features=0.9,
                                                    max_samples=0.9)),
                                ('rf',
                               RandomForestClassifier(n_estimators=300,
                                                      random_state=1)),
                               ('hgb',
                                 HistGradientBoostingClassifier(max_iter=50)),
                                  ('ADA', AdaBoostClassifier(n_estimators=150))],
                     weights=[2, 2, 2, 1])





    scores = cross_val_score(eclf, x_data, y_data, scoring='accuracy', cv=cv_triat)
    print("Accuracy: %f (+/- %0.2f) [%s(%f)]" % (scores.mean(), scores.std(),"Voting1",cv_triat))
    
    params = {'hgb__max_iter': [50, 200], 'rf__n_estimators': [20, 300], 'rf__random_state': [1, 200], 'ADA__n_estimators': [20, 150], 'bag__n_estimators': [20, 150]}
    grid = GridSearchCV(estimator=eclf, param_grid=params, cv=40)

    grid = grid.fit(x_data, y_data)

    best_estimator = grid.best_estimator_
    print(best_estimator)
    




    eclf = VotingClassifier(estimators=[
                                 ('rf',
                                   RandomForestClassifier(n_estimators=218,
                                                        random_state=178)),
                                ('hgb',
                                HistGradientBoostingClassifier(max_iter=100))], voting='hard')

    scores = cross_val_score(eclf, x_data, y_data, scoring='accuracy', cv=cv_triat)
    print("Accuracy: %f (+/- %0.2f) [%s(%f)]" % (scores.mean(), scores.std(),"Voting2",cv_triat))
    

#for clf, label in zip([ clf2, clf3, eclf], ['Random Forest', 'HistGradientBoosting', 'Ensemble']):
    #scores = cross_val_score(clf, x_data, y_data, scoring='accuracy', cv=40)
    #print("Accuracy: %f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))

#scores = model_selection.cross_val_predict(eclf, x_data, y_data, cv=40, method='predict_proba')
#curve_generator('Ensemble: Random Forest i HistGradientBoosting', scores)

#clf = RandomForestClassifier(n_estimators=1000, max_depth=None, random_state=24)
#scores = model_selection.cross_val_predict(clf, x_data, y_data, cv=40, method='predict_proba')
#curve_generator('Random Forest', scores)

#recordar canviar y_v per y_data en la funcio curve_generator

"""
"""
clf = RandomForestClassifier( n_estimators=218, random_state=178,n_jobs=-1)
scores = model_selection.cross_val_predict(clf, x_data, y_data, cv=40, method='predict_proba')
curve_generator('Random Forest', scores)


eclf = VotingClassifier(estimators=[('bag',
                                     BaggingClassifier(base_estimator=DecisionTreeClassifier(),
                                                       max_features=0.9,
                                                       max_samples=0.9)),
                                    ('rf',
                                     RandomForestClassifier(n_estimators=300,
                                                            random_state=1)),
                                    ('hgb',
                                     HistGradientBoostingClassifier(max_iter=50)),
                                    ('ADA', AdaBoostClassifier(n_estimators=150))],
                        weights=[2, 2, 2, 1], voting='soft')
scores = model_selection.cross_val_predict(eclf, x_data, y_data, cv=60, method='predict_proba')
curve_generator('Voting: bag+rf+hgb+ADA', scores)
"""

#clf.fit(x_t, y_t)
#probs = clf.predict_proba(x_v)
#print("Correct classification Random Forest     ", 0.7, "% of the data: ", clf.score(x_v, y_v))

# +-----------------------+
# | HIPERPARAMETRES       |
# +-----------------------+
"""
import time

start = time.time()


clf = RandomForestClassifier(n_estimators=150, max_depth=None, random_state=24)




#params = { 'n_estimators': [20, 60, 100, 150, 200,300], 'random_state': [1,24,50,75, 200], 'max_depth': [1,6,16,32]}
params = { 'n_estimators': [20,60,100,150,200,300,1000], 'random_state': [1,24,36,50,64,75, 200],'max_depth': [1,6,16,32]}
grid = GridSearchCV(estimator=clf, param_grid=params, n_jobs=-1)

grid = grid.fit(x_data, y_data)
best_estimator = grid.best_estimator_
print(best_estimator)

end = time.time()
"""

"""
import time
from scipy.stats import rv_discrete
p=np.arange(1,1000)
s=np.arange(1,1000)
d=np.arange(1,32)
start = time.time()


clf = RandomForestClassifier()
#params = { 'n_estimators': [20, 60, 100, 150, 200,300], 'random_state': [1,24,50,75, 200], 'max_depth': [1,6,16,32]}
params = { 'n_estimators': p, 'random_state': s,'max_depth': d}
randomS = RandomizedSearchCV(estimator=clf, param_distributions=params, n_jobs=-1, n_iter=1000)

randomS = randomS.fit(x_data, y_data)
best_estimator = randomS.best_estimator_
print(best_estimator)

end = time.time()"""

#print('time: '+str(end - start))

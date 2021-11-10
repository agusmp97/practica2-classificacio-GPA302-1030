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

# ---------------------------------------------------
# -------------------- APARTAT B --------------------
# ---------------------------------------------------
"""
# import some data to play with
iris = datasets.load_iris()

# Take the first two features. We could avoid this by using a two-dim dataset
X = iris.data[:, :2]
y = iris.target

n_classes = 3


def curve_generator(name, probs):
    plt.figure()
    precision = {};
    recall = {};
    average_precision = {}
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_v == i, probs[:, i])
        average_precision[i] = average_precision_score(y_v == i, probs[:, i])
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
        fpr[i], tpr[i], _ = roc_curve(y_v == i, probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    # Compute micro-average ROC curve and ROC area
    # Plot ROC curve
    plt.figure()
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label='ROC curve of class {} (area = {})' ''.format( i, round(roc_auc[i],2)))
    plt.legend()
    plt.title('{}'.format(name))
    plt.show()




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
# Mostra els primers 5 registres dels DataFrames dels jugadors
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
    x_data = data[:, :-1]  # Característiques del jugador
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
    plt.show()

    porc_pot = (len(dataset[dataset.Potability == 1]) / len(dataset.Potability)) * 100
    print('The percentage of waters that are potable is: {:.2f}%'.format(porc_pot))

y_balance(wp_dataset)


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
    return (dataset - dataset.mean(0)) / dataset.std(0)

dataset_norm = standardize_mean(wp_dataset)

pearson_correlation(dataset_norm)
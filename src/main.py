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


from sklearn.metrics import f1_score, precision_recall_curve, average_precision_score, roc_curve, auc


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



#For amb K-fold
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

        z = 3

    # plt.legend(loc='upper right')
    plt.legend()
    plt.xlabel('Folds count')
    plt.ylabel('{}'.format(type_score))
    plt.title('Model {} by K-fold'.format(type_score))
    plt.savefig("../figures/model_{}_kfoldB".format(type_score))
    plt.show()

z=3


#
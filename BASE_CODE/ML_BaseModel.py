
from pandas import read_csv
import numpy
from sklearn.pipeline import FeatureUnion
from sklearn.feature_selection import SelectKBest
from matplotlib import pyplot

from sklearn.model_selection import KFold,train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier  , AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
# load dataset
filename = 'pima-indians-diabetes.csv'
seed = 7
numpy.random.seed(seed)
# load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",", skiprows=1)

X = dataset[:,0:8]
Y = dataset[:,8]

models = []
models.append(('LR',   LogisticRegression()))
models.append(('LDA',  LinearDiscriminantAnalysis()))
models.append(('KNN',  KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)))
models.append(('CART', DecisionTreeClassifier(criterion='entropy', random_state=0)))
models.append(('RF',   RandomForestClassifier(criterion="entropy",n_estimators= 10,max_features=3)))
models.append(('Ada',  AdaBoostClassifier(algorithm='SAMME',n_estimators= 10)))
models.append(('GBT',  GradientBoostingClassifier(n_estimators= 10)))
models.append(('NB',   GaussianNB()))
models.append(('SVM',  SVC(kernel='linear', random_state=0)))
# evaluate each model in turn
results = []
names = []
scoring = 'accuracy'
for name, model in models:

    features = []
    features.append(('pca', PCA(n_components=3)))
    features.append(('select_best', SelectKBest(k=6)))
    feature_union = FeatureUnion(features)

    estimators = []
    estimators.append(('feature_union', feature_union))
    estimators.append(('standardize', StandardScaler()))
    estimators.append((name, model))

    Pipeline_model = Pipeline(estimators)

    kfold = KFold(n_splits=10, random_state=7, shuffle=True)

    cv_results = cross_val_score(Pipeline_model, X, Y, cv=kfold, scoring=scoring)

    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
# boxplot algorithm comparison
fig = pyplot.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()

#__________________________________________________
#Make predictions on validation dataset
# X_train, X_validation, y_train, y_validation = train_test_split(X, Y, test_size=0.2, random_state=42)
# model = LogisticRegression()
# model.fit(X_train, y_train)
# predictions = model.predict(X_validation)
# print("Accuracy:", accuracy_score(y_validation, predictions))
# print(confusion_matrix(y_validation, predictions))
# print(classification_report(y_validation, predictions))
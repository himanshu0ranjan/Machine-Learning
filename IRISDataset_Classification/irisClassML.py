#!/usr/bin/env python
"""
irisClassML is Created on October 28 2017 3:12 PM using IDE PyCharm Community Edition

@author: Himanshu Ranjan
@description: This is the Hello World program for Machine Learning on the Iris Dataset
@change tag: None
"""
# Load libraries
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
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
from sklearn.preprocessing import LabelEncoder

# Load Iris dataset directly from the UCI Machine Learning Repositry using Pandas
# url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

# Below code is used to load the dataset from the local directory on the system
url = "C:\\Users\\Heman\\PythonWorkspace\\Self Exercises\\ML Exercises\\firstMLproj_IRISdataset\\iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataframe = pandas.read_csv(url, names=names)

# 1. Find out the dimension of the dataframe using shape
print(dataframe.shape)

# 2. Let's see the data say for 25 rows
print(dataframe.head(25))

# 3. Summarize the attributes of data. Mainly this includes the count, mean, the min
# and max values as well as some percentiles.
print(dataframe.describe())

# 4. Class Distribution( Number of instances (rows) that belong to each class
print(dataframe.groupby('class').size())

# Data Visualization 
# 5. Univariate plots to better understand each attribute (box and whisker plots)
dataframe.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()

# 6. Histograms of each input variable
dataframe.hist()
plt.show()

# 7. Multivariate Plots (Scatter plot matrix)
scatter_matrix(dataframe)
plt.show()

# 8. Creating a validation dataset by
# Split-out validation dataset
dataset = dataframe.values
X = dataset[:,0:4]
y = dataset[:,4]
validation_size = 0.20
seed = 7
X_train, X_validation, y_train, y_validation = model_selection.train_test_split(X, y, test_size=validation_size, random_state=seed)

# 9. Test options and evaluation metric
seed = 7
scoring = 'accuracy'


# 10. Spot Check Algorithms
models = list()
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
print(models)

# 11. Evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# 12. Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

# 13. Make predictions on validation dataset using KNN Algorithm
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
predictions = knn.predict(X_validation)
print('Accuracy Score of KNN Algorithm:', accuracy_score(y_validation, predictions))
print('Confusion Matrix of KNN Algorithm:\n', confusion_matrix(y_validation, predictions))
print('Classification Report of KNN Algorithm:\n', classification_report(y_validation, predictions))
# 15. Plot the prediction vs true values for KNN algorithm
# first encode the string data of the class(target values) before using them in scatter
labelEncod = LabelEncoder()
y_true_KNN = labelEncod.fit_transform(y_validation)
y_pred_KNN = labelEncod.fit_transform(predictions)
plt.scatter(y_true_KNN, y_pred_KNN)
plt.xlabel('True Values as in Train Dataset')
plt.ylabel('Predictions Based on KNN Algorithm')
plt.show()

# 14. Make predictions on validation dataset using SVM Algorithm
svm = SVC()
svm.fit(X_train, y_train)
predictions = svm.predict(X_validation)
print('Accuracy Score of SVC Algorithm:', accuracy_score(y_validation, predictions))
print('Confusion Matrix of SVC Algorithm:\n', confusion_matrix(y_validation, predictions))
print('Classification Report of SVC Algorithm:\n', classification_report(y_validation, predictions))
# 15. Plot the prediction vs true values for SVC Algorithm
y_true_SVC = labelEncod.fit_transform(y_validation)
y_pred_SVC = labelEncod.fit_transform(predictions)
plt.scatter(y_true_SVC, y_pred_SVC)
plt.xlabel('True Values as in Train Dataset')
plt.ylabel('Predictions Based on SVC Algorithm')
plt.show()
# Conclusion : The best accuracy came for SVC algorithm

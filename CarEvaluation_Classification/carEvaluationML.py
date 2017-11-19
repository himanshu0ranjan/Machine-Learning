# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 12:40:24 2017

@author      : Himanshu Ranjan
@description : This is a Car Evaluation Prediction Program for Machine Learning on 
               UCI Car Evaluation Dataset for Accenture AI Hackathon Oct 17.
"""

# Load libraries
import pandas 
import numpy
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot as plt

# Start of the program

# Load Car Evaluation dataset directly from the UCI Public repository using panda
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"
names = ['BuyingPrice', 'MaintenanceCost', 'Doors', 'Persons', 'Lug_Boot', 'Safety', 'CarAcceptability']
dataframe = pandas.read_csv(url, names=names)

# 1. Find out the dimension of the dataset using shape
print(dataframe.shape)
# 2. Let's see the first 25 rows of the data
print(dataframe.head(25))
# 3. Summarize the attributes of data
print(dataframe.describe())
# 4. Class Distribution
print(dataframe.groupby('CarAcceptability').size())

# 5. Let's see the initial datatypes of the data
print('Initial dtypes:', dataframe.dtypes)
# 6. Check the data for null or missing values
print(dataframe.isnull().sum())

# 7. Modify the columns which have string data type values using label encoder
colModify = ['BuyingPrice', 'MaintenanceCost', 'Doors', 'Persons', 'Lug_Boot', 'Safety', 'CarAcceptability']
labelEncod = LabelEncoder()
for i in colModify:
    dataframe[i] = labelEncod.fit_transform(dataframe[i]) 
    dataframe.dtypes

# 8. Print the data types after encoding
print(dataframe.dtypes)
# 9. Look at first 25 rows of the dataframe after the label encoding 
print(dataframe.head(25))

# 10. Data Visualization 
# 10a. Univariate Plots (box and whisker plots)
dataframe.plot(kind='box', subplots=True, layout=(2,4), sharex=False, sharey=False)
plt.show()

# 10b. Histograms of each input variable
dataframe.hist()
plt.show()

# 10c. Multivariate Plots (Scatter plot matrix)
scatter_matrix(dataframe)
plt.show()

# 10. Take out the dataset in num array
dataset = dataframe.values

# 11. Split data into X and Y and create validation data
X = dataset[:,0:6]
y = dataset[:,6]
validation_size = 0.10
seed = 6
X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=validation_size, random_state=seed)

# 12. Test options and evaluation metric
seed = 6
scoring = 'accuracy'

# 13. Spot Check Algorithms
models = list()
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
models.append(('RC', RandomForestClassifier()))

# 14. evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = KFold(n_splits=10, random_state=seed)
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# 15. Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

# 16. Fit a Decision tree Model
dtc = DecisionTreeClassifier()    
model = dtc.fit(X_train, y_train)
predictions = dtc.predict(X_validation)
print('Accuracy Score:', accuracy_score(y_validation, predictions))
print('Confusion Matrix:', confusion_matrix(y_validation, predictions))
print('Classification report:', classification_report(y_validation, predictions))
plt.scatter(y_validation, predictions)
plt.xlabel('True Values')
plt.ylabel('Predictions')

# 17. Fit a Random forest classifier Model
rfc = RandomForestClassifier()    
model = rfc.fit(X_train, y_train)
predictions = rfc.predict(X_validation)
print('Accuracy Score:', accuracy_score(y_validation, predictions))
print('Confusion Matrix:', confusion_matrix(y_validation, predictions))
print('Classification report:', classification_report(y_validation, predictions))
plt.scatter(y_validation, predictions)
plt.xlabel('True Values')
plt.ylabel('Predictions')

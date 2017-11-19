# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 16:12:17 2017 using IDE PyCharm Community Edition

@author      : Himanshu Ranjan
@description : This is a Loan Prediction Program for Machine Learning on 
               Loan Prediction dataset given by Analytics Vidhya Hackathon
@change tag  : None
"""

# Load libraries
import pandas
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from xgboost import XGBClassifier
from matplotlib import pyplot as plt


# Start of the program
# Load Loan Prediction Training dataset from the current working directory using pandas
url = ".\\trainLoanPrediction.csv"
dataframe = pandas.read_csv(url, header=0, index_col=0)  # The training dataset contains the header and Loan id as RowID

# 1. Find out the dimension of the dataframe using shape
print(dataframe.shape)

# 2. Let's see the data say for 25 rows
print(dataframe.head(25))

# 3. Check if there are any missing values in any column. Also find the count for the same
print(dataframe.isnull().sum())

# 4. Drop all the rows which are having any missing values in any of the columns
dataframe = dataframe.dropna(axis=0, how='any')

# 5. Check if there are still any missing values in any column.
print(dataframe.isnull().sum())

# 6. Modify the columns which have string data type values using label encoder leaving out Loan_ID
# column since it's just a row identifier
colModify = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']
labelEncod = LabelEncoder()  # This is optional as we already have done this above in case of Loan_Status
for i in colModify:
    dataframe[i] = labelEncod.fit_transform(dataframe[i])

# 6a. Let's checkout the datatypes of the each column now after encoding
print(dataframe.dtypes)

# 7. Summarize the attributes of data. Mainly this includes the count, mean, the min
# and max values as well as some percentiles.
print(dataframe.describe())

# 8. Class Distribution( Number of instances (rows) that belong to each class
print(dataframe.groupby('Loan_Status').size())

# Data Visualization
# 9. Draw Univariate plots to better understand each attribute (box and whisker plots)
dataframe.plot(kind='box', subplots=True, layout=(3, 5), sharex=False, sharey=False)
plt.show()

# 10. Histograms of each input variable
dataframe.hist()
plt.show()

# 11. Draw the density plot fr each attribute
dataframe.plot(kind='density', subplots=True, layout=(3, 5), sharex=False, sharey=False)
plt.draw()

# 12. Multivariate Plots (Scatter plot matrix)
scatter_matrix(dataframe)
plt.show()

# 13. Creating a validation dataset using dataframe. Also removing header row and row identifier Loan_ID
# Split-out validation dataset in num array
dataset = dataframe.values
# Split data into X and y and create validation data
X = dataset[:, :11]  # removed rhe row identifier from the dataset's num array
y = dataset[:, 11]
validation_size = 0.20
seed = 7
X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=validation_size, random_state=seed)

# 14. Test options and evaluation metric
seed = 7
scoring = 'accuracy'

# 15. Spot Check Algorithms
models = list()
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
models.append(('XGBC', XGBClassifier()))

# 16. Evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = KFold(n_splits=10, random_state=seed)
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# 17. Fit the Logistics Regression model
lr = LogisticRegression()
model = lr.fit(X_train, y_train)
predictions = lr.predict(X_validation)
print('LR Model Score(Mean Accuracy of Test Data): %.2f%%' % (model.score(X_validation, y_validation) * 100.0))
print('Accuracy Score of LR Algorithm: %.2f%%' % (accuracy_score(y_validation, predictions) * 100.0))
print('Confusion Matrix of LR Algorithm:\n', confusion_matrix(y_validation, predictions))
print('Classification Report of LR Algorithm:\n', classification_report(y_validation, predictions))

# 17a. Plot the LR model
plt.scatter(y_validation, predictions)
plt.xlabel('LR True Values')
plt.ylabel('LR Predictions')
plt.show()

# 17b. Test cross validation for LR algorithm
seed = 7
scoring = 'accuracy'

kfold = KFold(n_splits=10, random_state=seed)
scores = cross_val_score(model, X, y, cv=kfold)
print('LR Cross-Validated Scores:', scores)
predictions = cross_val_predict(model, X, y, cv=kfold)
print('Cross Validated LR Model Score: %.2f%%' % (model.score(X, y) * 100.0))
print('Cross Validated Accuracy Score of LR Algorithm: %.2f%%' % (accuracy_score(y, predictions) * 100.0))
print('Cross Validated Confusion Matrix of LR Algorithm:\n', confusion_matrix(y, predictions))
print('Cross Validated Classification Report of LR Algorithm:\n', classification_report(y, predictions))

# 17c. Plot the line / model
plt.scatter(y, predictions)
plt.xlabel('LR CV True Values')
plt.ylabel('LR CV Predictions')
plt.show()

# 18. Fit the GaussianNB() model
nb = GaussianNB()
model = nb.fit(X_train, y_train)
predictions = nb.predict(X_validation)
print('NB Model Score(Mean Accuracy of Test Data): %.2f%%' % (model.score(X_validation, y_validation) * 100.0))
print('Accuracy Score of NB Algorithm: %.2f%%' % (accuracy_score(y_validation, predictions) * 100.0))
print('Confusion Matrix of NB Algorithm:\n', confusion_matrix(y_validation, predictions))
print('Classification Report of NB Algorithm:\n', classification_report(y_validation, predictions))

# 18a. Plot the line / model
plt.scatter(y_validation, predictions)
plt.xlabel('NB True Values')
plt.ylabel('NB Predictions')
plt.show()

# 18b. Test cross validation for LR and evaluation metric
seed = 7
scoring = 'accuracy'

kfold = KFold(n_splits=10, random_state=seed)
scores = cross_val_score(model, X, y, cv=kfold)
print('Cross-Validated Scores:', scores)
predictions = cross_val_predict(model, X, y, cv=kfold)
print('Cross Validated NB Model Score: %.2f%%' % (model.score(X, y) * 100.0))
print('Cross Validated Accuracy Score of NB Algorithm: %.2f%%' % (accuracy_score(y, predictions) * 100.0))
print('Cross Validated Confusion Matrix of NB Algorithm:\n', confusion_matrix(y, predictions))
print('Cross Validated Classification Report of NB Algorithm:\n', classification_report(y, predictions))

# 18c. Plot the line / model
plt.scatter(y, predictions)
plt.xlabel('NB CV True Values')
plt.ylabel('NB CV Predictions')
plt.show()

# 19. Fit an Extra Boost Classifier model to the data
xgbc = XGBClassifier()
model = xgbc.fit(X_train, y_train)
predictions = xgbc.predict(X_validation)
predictions = [round(value) for value in predictions]
print('XGBC Model Score(Mean Accuracy of Test Data): %.2f%%' % (model.score(X_validation, y_validation) * 100.0))
print('Accuracy Score of XGBC Algorithm: %.2f%%' % (accuracy_score(y_validation, predictions) * 100.0))
print('Confusion Matrix of XGBC Algorithm:\n', confusion_matrix(y_validation, predictions))
print('Classification Report of XGBC Algorithm:\n', classification_report(y_validation, predictions))

# 19a. Plot the XGBC model
plt.scatter(y_validation, predictions)
plt.xlabel('XGBC True Values')
plt.ylabel('XGBC Predictions')
plt.show()

# 19b. Test cross validation for XGBC algorithm
seed = 7
scoring = 'accuracy'

kfold = KFold(n_splits=10, random_state=seed)
scores = cross_val_score(model, X, y, cv=kfold)
print('XGBC Cross-Validated Scores:', scores)
predictions = cross_val_predict(model, X, y, cv=kfold)
print('Cross Validated XGBC Model Score: %.2f%%' % (model.score(X, y) * 100.0))
print('Cross Validated Accuracy Score of XGBC Algorithm: %.2f%%' % (accuracy_score(y, predictions) * 100.0))
print('Cross Validated Confusion Matrix of XGBC Algorithm:\n', confusion_matrix(y, predictions))
print('Cross Validated Classification Report of XGBC Algorithm:\n', classification_report(y, predictions))

# 19c. Plot the line / model
plt.scatter(y, predictions)
plt.xlabel('XGBC CV True Values')
plt.ylabel('XGBC CV Predictions')
plt.show()

# 20. Fit the LDA model
lda = LinearDiscriminantAnalysis()
model = lda.fit(X_train, y_train)
predictions = nb.predict(X_validation)
print('LDA Model Score(Mean Accuracy of Test Data): %.2f%%' % (model.score(X_validation, y_validation) * 100.0))
print('Accuracy Score of LDA Algorithm: %.2f%%' % (accuracy_score(y_validation, predictions) * 100.0))
print('Confusion Matrix of LDA Algorithm:\n', confusion_matrix(y_validation, predictions))
print('Classification Report of LDA Algorithm:\n', classification_report(y_validation, predictions))

# 21a. Plot the line / model
plt.scatter(y_validation, predictions)
plt.xlabel('LDA True Values')
plt.ylabel('LDA Predictions')
plt.show()

# 21b. Test cross validation for LR and evaluation metric
seed = 7
scoring = 'accuracy'

kfold = KFold(n_splits=10, random_state=seed)
scores = cross_val_score(model, X, y, cv=kfold)
print('Cross-Validated Scores:', scores)
predictions = cross_val_predict(model, X, y, cv=kfold)
print('Cross Validated LDA Model Score: %.2f%%' % (model.score(X, y) * 100.0))
print('Cross Validated Accuracy Score of LDA Algorithm: %.2f%%' % (accuracy_score(y, predictions) * 100.0))
print('Cross Validated Confusion Matrix of LDA Algorithm:\n', confusion_matrix(y, predictions))
print('Cross Validated Classification Report of LDA Algorithm:\n', classification_report(y, predictions))

# 21c. Plot the line / model
plt.scatter(y, predictions)
plt.xlabel('LDA CV True Values')
plt.ylabel('LDA CV Predictions')
plt.show()

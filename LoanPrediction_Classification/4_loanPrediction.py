# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 16:12:17 2017 using IDE PyCharm Community Edition

@author      : Himanshu Ranjan
@description : This is a Loan Prediction Program for Machine Learning on 
               Loan Prediction dataset given by Analytics Vidhya Hackathon
@change tag  : None
"""

# Load libraries
import numpy
import pandas
from pandas.plotting import scatter_matrix
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier, \
    RandomForestClassifier
from sklearn.feature_selection import RFE
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

# 4. Impute missing values as per your strategy. I have manly looked in to the data and done imputing based on that
# In first example 1_loanPrediction.py we had just dropped all the rows which are having missing values
# We clearly saw the accuracy of that model. Now let's see what happens when we impute the missing values
# 4a. Credit_History column
# before imputing it let's check the Credit_History vs Loan_Status plot
dataframe.boxplot(column="Credit_History", by="Loan_Status")
plt.show()

dataframe.hist(column="Credit_History", by="Loan_Status")
plt.show()

# As per the plot impute the missing values in the Credit_History column
# according to the Loan_Status after encoding the loan status column using label encoder
# Basically first convert the loan status column of values Y/N to 0/1 and then populate the missing values in
# Credit_History column
labelEncod = LabelEncoder()
dataframe['Loan_Status'] = labelEncod.fit_transform(dataframe['Loan_Status'])
dataframe['Credit_History'].fillna(dataframe['Loan_Status'], inplace=True)

# 4b. Impute the Gender, Married, Dependents and Self_Employed column as per the below strategy.
# Strategy has been developed after looking at the data in csv file
for i in range(len(dataframe)):
    # Handle Gender missing values
    if dataframe['Loan_Amount_Term'][i] >= 360.0:
        dataframe['Gender'].fillna('Male', inplace=True)
    else:
        dataframe['Gender'].fillna('Female', inplace=True)
    # Handle Married column missing values
    if dataframe['Gender'][i] == 'Female' and dataframe['Loan_Status'][i] == 'N':
        dataframe['Married'].fillna('No', inplace=True)
    else:
        dataframe['Married'].fillna('Yes', inplace=True)

# 4c. Fill missing values in the Dependents column with maximum frequent values
dataframe['Dependents'].fillna(dataframe['Dependents'].dropna().max(), inplace=True)

# 4d. Fill missing values in the Self_Employed column with maximum frequent values
dataframe['Self_Employed'].fillna(dataframe['Self_Employed'].dropna().max(), inplace=True)

# 4e. Fill the missing values in LoanAmount with mean of the present values
dataframe['LoanAmount'].fillna(dataframe['LoanAmount'].mean(), inplace=True)

# 4f. Fill the missing values in Loan_Amount_Term with most frequent values
dataframe['Loan_Amount_Term'].fillna(dataframe['Loan_Amount_Term'].max(), inplace=True)

# 5. Check if there are still any missing values in any column.
print(dataframe.isnull().sum())

# Let's apply some feature engineering on the data
# 6. Bining the LoanAmount, ApplicantIncome and CoapplicantIncome for clear visualization and prediction
df1 = dataframe.loc[:, 'Gender':'Property_Area']  # Split up the dataframe in to df1 and df2
df2 = dataframe.loc[:, 'Loan_Status']

# 6a. Bining the LoanAmount
bins1 = numpy.array([90, 140, 190])
df1['Binned_Loan_Amount'] = numpy.digitize(df1['LoanAmount'], bins1)

# 6b. Bining the ApplicantIncome and CoapplicantIncome
bins2 = numpy.array([10000, 20000])
df1['ApplicantIncome'] = numpy.digitize(df1['ApplicantIncome'], bins2)
df1['CoapplicantIncome'] = numpy.digitize(df1['CoapplicantIncome'], bins2)

# 6c. Bining the Loan_Amount_Term
bins4 = numpy.array([180, 240, 360])
df1['Loan_Amount_Term'] = numpy.digitize(df1['Loan_Amount_Term'], bins4)

# 7. Create a set of dummy variables from the values in Gender variable
dummy_gender = pandas.get_dummies(dataframe['Gender'], prefix='Gender')

# 7a. Join the new dummy variables and binned columns to the main dataframe
dataframe = pandas.concat([df1, dummy_gender], axis=1)
dataframe = pandas.concat([dataframe, df2], axis=1)

# 8. Modify the columns which have string data type values using label encoder leaving out Loan_ID
# column since it's just a row identifier
# Please note that the Loan_Status column has been removed from here because that has been already encoded above
# under section 4a.
colModify = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
labelEncod = LabelEncoder()  # This is optional as we already have done this above in case of Loan_Status
for i in colModify:
    dataframe[i] = labelEncod.fit_transform(dataframe[i])

# 8a. Let's checkout the datatypes of the each column now after encoding
print(dataframe.dtypes)

# 9. Summarize the attributes of data. Mainly this includes the count, mean, the min
# and max values as well as some percentiles.
print(dataframe.describe())

# 10. Class Distribution( Number of instances (rows) that belong to each class
print(dataframe.groupby('Loan_Status').size())

# Data Visualization
# 11. Draw Univariate plots to better understand each attribute (box and whisker plots)
dataframe.plot(kind='box', subplots=True, layout=(4, 5), sharex=False, sharey=False)
plt.show()

# 12. Histograms of each input variable
dataframe.hist()
plt.show()

# 13. Draw the density plot fr each attribute
dataframe.plot(kind='density', subplots=True, layout=(4, 5), sharex=False, sharey=False)
plt.draw()

# 14. Multivariate Plots (Scatter plot matrix)
scatter_matrix(dataframe)
plt.show()

# 15. Creating a validation dataset using dataframe. Also removing header row and row identifier Loan_ID
# Split-out validation dataset in num array
dataset = dataframe.values
# Split data into X and y and create validation data
X = dataset[:, :14]  # removed rhe row identifier from the dataset's num array
y = dataset[:, 14]  # Five columns increased due to feature engineering
validation_size = 0.20
seed = 7
X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=validation_size, random_state=seed)

# 16. Test options and evaluation metric
seed = 7
scoring = 'accuracy'

# 17. Try the individual ensembling with GradientBoostingClassifier
num_trees = 100
max_features = 8
kfold = KFold(n_splits=10, random_state=seed)
gb = GradientBoostingClassifier(n_estimators=num_trees, random_state=seed, max_features=max_features)
model = gb.fit(X_train, y_train)
scores = cross_val_score(model, X, y, cv=kfold)
predictions = cross_val_predict(model, X, y, cv=kfold)
print('GB Cross-Validated Scores:', scores)
print('Cross Validated GB Model Score: %.2f%%' % (model.score(X, y) * 100.0))
print('Cross Validated Accuracy Score of GB Algorithm: %.2f%%' % (accuracy_score(y, predictions) * 100.0))
print('Cross Validated Confusion Matrix of GB Algorithm:\n', confusion_matrix(y, predictions))
print('Cross Validated Classification Report of GB Algorithm:\n', classification_report(y, predictions))

# 18. Try the individual ensembling with AdaBoostClassifier
num_trees = 100
kfold = KFold(n_splits=10, random_state=seed)
ab = AdaBoostClassifier(n_estimators=num_trees, random_state=seed)
model = ab.fit(X_train, y_train)
scores = cross_val_score(model, X, y, cv=kfold)
predictions = cross_val_predict(model, X, y, cv=kfold)
print('ADB Cross-Validated Scores:', scores)
print('Cross Validated ADB Model Score: %.2f%%' % (model.score(X, y) * 100.0))
print('Cross Validated Accuracy Score of ADB Algorithm: %.2f%%' % (accuracy_score(y, predictions) * 100.0))
print('Cross Validated Confusion Matrix of ADB Algorithm:\n', confusion_matrix(y, predictions))
print('Cross Validated Classification Report of ADB Algorithm:\n', classification_report(y, predictions))

# 19. Try the individual ensembling with ExtraTreesClassifier
num_trees = 100
kfold = KFold(n_splits=10, random_state=seed)
etc = ExtraTreesClassifier(n_estimators=num_trees, random_state=seed)
model = etc.fit(X_train, y_train)
scores = cross_val_score(model, X, y, cv=kfold)
predictions = cross_val_predict(model, X, y, cv=kfold)
print('ETC Cross-Validated Scores:', scores)
print('Cross Validated ETC Model Score: %.2f%%' % (model.score(X, y) * 100.0))
print('Cross Validated Accuracy Score of ETC Algorithm: %.2f%%' % (accuracy_score(y, predictions) * 100.0))
print('Cross Validated Confusion Matrix of ETC Algorithm:\n', confusion_matrix(y, predictions))
print('Cross Validated Classification Report of ETC Algorithm:\n', classification_report(y, predictions))

# 20. Try the individual ensembling with ExtraTreesClassifier
num_trees = 100
kfold = KFold(n_splits=10, random_state=seed)
rfc = RandomForestClassifier(n_estimators=num_trees, random_state=seed)
model = rfc.fit(X_train, y_train)
scores = cross_val_score(model, X, y, cv=kfold)
predictions = cross_val_predict(model, X, y, cv=kfold)
print('RFC Cross-Validated Scores:', scores)
print('Cross Validated RFC Model Score: %.2f%%' % (model.score(X, y) * 100.0))
print('Cross Validated Accuracy Score of RFC Algorithm: %.2f%%' % (accuracy_score(y, predictions) * 100.0))
print('Cross Validated Confusion Matrix of RFC Algorithm:\n', confusion_matrix(y, predictions))
print('Cross Validated Classification Report of RFC Algorithm:\n', classification_report(y, predictions))


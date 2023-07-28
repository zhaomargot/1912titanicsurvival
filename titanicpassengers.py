# Margot Zhao

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# display all columns
pd.set_option("display.max_columns", None)

# (1) read dataset into a dataframe
titanic = pd.read_csv("Titanic.csv")
print("Data:")
print(titanic)

# (2) determine the target variable: "survived"
# set the rest as feature variables

# (3) drop factors that are not likely to be relevant for logistic regression
# drop passenger column
del titanic["Passenger"]

# (4) make sure there are no missing values
print("Number of Missing Values:")
print(titanic.isnull().sum())
print()

# (5) plot count plots of remaining factors
# countplot for class
sns.countplot(data=titanic, x="Class")
plt.show()
# countplot for sex
sns.countplot(x="Sex", data=titanic)
plt.show()
# countplot for age
sns.countplot(x="Age", data=titanic)
plt.show()
# countplot for survivbed
sns.countplot(x="Survived", data=titanic)
plt.show()

# (6) convert all categorical variables into dummy variables
titanic2 = pd.get_dummies(titanic, columns=["Class", "Sex", "Age"])
print(titanic2)

# (7) partition data into train and test sets (75/25)
X = titanic2[["Class_1st", "Class_2nd", "Class_3rd", "Class_Crew", "Sex_Female", "Sex_Male", "Age_Adult", "Age_Child"]]
Y = titanic2["Survived"]

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=2021)

# (8) fit training data to a logistic regression model
# import class
from sklearn.linear_model import LogisticRegression

# instantiate the model
logreg = LogisticRegression()

# fit the model with data
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)

# (9) display accuracy, precision, recall of predictions for survivability
# accuracy
from sklearn import metrics
print("Accuracy:\n", metrics.accuracy_score(Y_test, Y_pred))
print()
# precision and recall
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred))
print()

# (10) display confusion matrix with labels Y / N
from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(logreg, X_test, Y_test, display_labels=["Yes", "No"])
plt.show()

# (11) predicted value for survivability of a female adult passenger travelling 2nd class
# create array representing above scenario
x = np.array([[0, 1, 0, 0, 1, 0, 1, 0]])
# print prediction result
print("Result: ")
print(logreg.predict(x.reshape(1, -1)))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score


#importing dataset
glass = pd.read_csv('C:/Users/Harjinder/Downloads/glass.csv')
#print(glass.shape)
print(glass.head())
print(glass.info())
print(glass.describe())

#splitting data into training and testing
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
print("x train")
print(x_train)
print("x test")
print(x_test)
print("y train")
print(y_train)
print("y test")
print(y_test)

#feature scaling
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

#logistic regression
classifier1 = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')
classifier1.fit(x_train,y_train)
y_pred1 = classifier1.predict(x_test)
print(classification_report(y_test,y_pred1))
print(accuracy_score(y_test,y_pred1))
cm=confusion_matrix(y_test,y_pred1)

#decision tree classifier
classifier2 = DecisionTreeClassifier(criterion='entropy', random_state=0)
classifier2.fit(x_train,y_train)
y_pred2 = classifier2.predict(x_test)
print("decision tree")
print(classification_report(y_test,y_pred2))
print(accuracy_score(y_test,y_pred2))
cm2=confusion_matrix(y_test,y_pred2)

#random forest classifier
classifier3 = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
classifier3.fit(x_train,y_train)
y_pred3 = classifier3.predict(x_test)
print("random forest")
print(classification_report(y_test,y_pred3))
print(accuracy_score(y_test,y_pred3))
cm3=confusion_matrix(y_test,y_pred3)

#naive bayes
classifier4 = GaussianNB()
classifier4.fit(x_train,y_train)
y_pred4 = classifier4.predict(x_test)
print("Naive Bayes")
print(classification_report(y_test,y_pred4))
print(accuracy_score(y_test,y_pred4))
cm4=confusion_matrix(y_test,y_pred4)

#svc
classifier5 = SVC()
classifier5.fit(x_train,y_train)
y_pred5 = classifier5.predict(x_test)
print("svc")
print(classification_report(y_test,y_pred5))
print(accuracy_score(y_test,y_pred5))
cm5=confusion_matrix(y_test,y_pred5)

#KNN
classifier6=KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
classifier6.fit(x_train,y_train)
y_pred6=classifier6.predict(x_test)
print("KNN")
print(classification_report(y_test,y_pred6))
print(accuracy_score(y_test,y_pred6))
cm6=confusion_matrix(y_test,y_pred6)
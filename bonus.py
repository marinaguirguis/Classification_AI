####el data w el testing ####
import pandas as pd
from sklearn import model_selection

header = ['Label','REGION-CENTROID-COL','REGION-CENTROID-ROW','REGION-PIXEL-COUNT',
         'SHORT-LINE-DENSITY-5','SHORT-LINE-DENSITY-2','VEDGE-MEAN','VEDGE-SD,HEDGE-MEAN',
         'HEDGE-SD','INTENSITY-MEAN','RAWRED-MEAN','RAWBLUE-MEAN','RAWGREEN-MEAN','EXRED-MEAN',
         'EXBLUE-MEAN','EXGREEN-MEAN','VALUE-MEAN','SATURATION-MEAN','HUE-MEAN']

test = pd.read_csv("segmentation.test.txt", names = header)
data = pd.read_csv("segmentation.data.txt", names = header)
# 3lshan ashil el 7agat eli fo2 eli homa 1 w 2 fa ha5lih size 3
test = test[3:]
data = data[3:]
dataset = pd.concat([test,data]) #3alshan y concatinat el test w el data
val = dataset.values
Y= val[:,0]  
X = val[:,1:] 
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.5, random_state=1) #to split 50% training 50% testing

##### classification #######
### 1)naive baiyes ###
from sklearn.naive_bayes import GaussianNB

GaussianNB = GaussianNB()
GaussianNB.fit(X_train, Y_train.ravel())
print("\n1)naive baiyes :")
print('Accuracy on TRAINING set: {:.4f}'.format(GaussianNB.score(X_train, Y_train)))
print('Accuracy on TEST set: {:.4f}'.format(GaussianNB.score(X_test, Y_test)))

### 2)Decision tree ###
from sklearn.tree import DecisionTreeClassifier

DecisionTree = DecisionTreeClassifier(criterion="entropy")
DecisionTree.fit(X_train, Y_train)
print("\n2)Decision Tree :")
print('Accuracy on TRAINING set: {:.4f}'.format(DecisionTree.score(X_train, Y_train)))
print('Accuracy on TEST set: {:.4f}'.format(DecisionTree.score(X_test, Y_test)))

### 3)K nearest neightbor###
from sklearn.neighbors import KNeighborsClassifier

#N = 3
KNN = KNeighborsClassifier(n_neighbors=3)
KNN.fit(X_train, Y_train.ravel())
print("\n3)K nearest neighbors ( N = 3) : ")
print('Accuracy on training set: {:.4f}'.format(KNN.score(X_train, Y_train)))
print('Accuracy on test set: {:.4f}'.format(KNN.score(X_test, Y_test)))
#N = 10
KNN = KNeighborsClassifier(n_neighbors=10)
KNN.fit(X_train, Y_train.ravel())
print("\n4)K nearest neighbors ( N = 10) : ")
print('Accuracy on training set: {:.4f}'.format(KNN.score(X_train, Y_train)))
print('Accuracy on test set: {:.4f}'.format(KNN.score(X_test, Y_test)))
#N = 20
KNN = KNeighborsClassifier(n_neighbors=20)
KNN.fit(X_train, Y_train.ravel())
print("\n5)K nearest neighbors ( N = 20) : ")
print('Accuracy on training set: {:.4f}'.format(KNN.score(X_train, Y_train)))
print('Accuracy on test set: {:.4f}'.format(KNN.score(X_test, Y_test)))

### Preprocessing : scaling ###
from sklearn.preprocessing import MinMaxScaler 

X = X.astype('float_')
scaling = MinMaxScaler(feature_range=(-1, 1))
scaling.fit(X)
X = scaling.transform(X)
print ('\n After scaling : ')
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.5, random_state=1) #to split 50% training 50% testing
##### classification #######
### 1)naive baiyes ###
from sklearn.naive_bayes import GaussianNB

GaussianNB = GaussianNB()
GaussianNB.fit(X_train, Y_train.ravel())
print("\n1)naive baiyes :")
print('Accuracy on TRAINING set: {:.4f}'.format(GaussianNB.score(X_train, Y_train)))
print('Accuracy on TEST set: {:.4f}'.format(GaussianNB.score(X_test, Y_test)))

### 2)Decision tree ###
from sklearn.tree import DecisionTreeClassifier

DecisionTree = DecisionTreeClassifier(criterion="entropy")
DecisionTree.fit(X_train, Y_train)
print("\n2)Decision Tree :")
print('Accuracy on TRAINING set: {:.4f}'.format(DecisionTree.score(X_train, Y_train)))
print('Accuracy on TEST set: {:.4f}'.format(DecisionTree.score(X_test, Y_test)))

### 3)K nearest neightbor###
from sklearn.neighbors import KNeighborsClassifier

#N = 3
KNN = KNeighborsClassifier(n_neighbors=3)
KNN.fit(X_train, Y_train.ravel())
print("\n3)K nearest neighbors ( N = 3) : ")
print('Accuracy on training set: {:.4f}'.format(KNN.score(X_train, Y_train)))
print('Accuracy on test set: {:.4f}'.format(KNN.score(X_test, Y_test)))
#N = 10
KNN = KNeighborsClassifier(n_neighbors=10)
KNN.fit(X_train, Y_train.ravel())
print("\n4)K nearest neighbors ( N = 10) : ")
print('Accuracy on training set: {:.4f}'.format(KNN.score(X_train, Y_train)))
print('Accuracy on test set: {:.4f}'.format(KNN.score(X_test, Y_test)))
#N = 20
KNN = KNeighborsClassifier(n_neighbors=20)
KNN.fit(X_train, Y_train.ravel())
print("\n5)K nearest neighbors ( N = 20) : ")
print('Accuracy on training set: {:.4f}'.format(KNN.score(X_train, Y_train)))
print('Accuracy on test set: {:.4f}'.format(KNN.score(X_test, Y_test)))
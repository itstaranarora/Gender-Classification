from sklearn import discriminant_analysis
from sklearn import tree
from sklearn import neighbors
from sklearn.metrics import accuracy_score

#Data and labels [height, weight, shoe size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40], [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]
Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female', 'female', 'male', 'male']

#Classifiers
clf1 = discriminant_analysis.QuadraticDiscriminantAnalysis()
clf2 = tree.DecisionTreeClassifier()
clf3 = neighbors.KNeighborsClassifier()

#Train Model
clf1 = clf1.fit(X,Y)
clf2 = clf2.fit(X,Y)
clf3 = clf3.fit(X,Y)

_X=[[184,84,44],[198,92,48],[183,83,44],[166,47,36],[170,60,38],[172,64,39],[182,80,42],[180,80,43]]
_Y=['male','male','male','female','female','female','male','male']

#Prediction
prediction1 = clf1.predict(_X)
prediction2 = clf2.predict(_X)
prediction3 = clf3.predict(_X)

#Result
r1 = accuracy_score(_Y,prediction1)
r2 = accuracy_score(_Y,prediction2)
r3 = accuracy_score(_Y,prediction3)

#Printing Best Result
if r1>r2 and r1>r3:
    print("QuadraticDiscriminantAnalysis :",r1)
elif r2>r1 and r2>r3:
    print("DecisionTreeClassifier :",r2)
else:
    print("KNeighborsClassifier :",r3)
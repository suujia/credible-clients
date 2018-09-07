## Run instructions

$ docker build --name credibleimage credible-clients (whole repository)
or run it using any other image you have that contains python, numpy, scipy, scikit learn

$ docker run -ti --name learning -v ~/Desktop/Software/credible-clients:/credible-clients credibleimage

$ ls  
--> you will see credible-clients folder 

$ cd credible-clients

$ python model.py


## K Nearest
```
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()
```
Results: 
```
Accuracy:  75.583%
Precision: 36.707%
Recall:    18.852%
```

## Decision Tree 
```
from sklearn import tree
model = tree.DecisionTreeClassifier()
```
Results:
```
Accuracy:  73.400%
Precision: 40.805%
Recall:    42.388%
```
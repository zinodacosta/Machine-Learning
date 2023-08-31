import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split 


iris = datasets.load_iris()
irisData = iris['data']
irisLabels = iris['target']

SDscore = (irisData - np.mean(irisData, axis=0)) / np.std(irisData, axis=0)

data_train, data_test, label_train, label_test = train_test_split(irisData, irisLabels, test_size = 0.3, random_state = 45)



print(data_train)
print(data_test)
print(label_train)
print(label_test)


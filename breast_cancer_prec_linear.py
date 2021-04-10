import numpy as np
# Standard scientific Python imports
# sum each support vector*alpha
import matplotlib.pyplot as plt
import struct
import json

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer


digits = datasets.load_breast_cancer()

# Create a classifier: a support vector classifier
clf = svm.SVC(kernel = 'linear')


X_train, X_test, y_train, y_test = train_test_split(
    digits.data, digits.target, test_size=0.5, shuffle=False)


clf.fit(X_train, y_train)
spv = clf.support_vectors_ 
print(spv)
d = clf.dual_coef_
w = clf.coef_
b = clf.intercept_

print(np.shape(d))
print(d)
print(np.shape(b))
print(b)

count = 0
count_alpha = 0
coef = []
test = []
for x in d[count]:
    coef.append(x)
for x in X_test[count]:
    test.append(x)    

count = 0
sum = 0
for x in spv:
    x = x*coef[count]
    count = count+1
    sum = sum + x

print(np.shape(sum))
print(sum)



full_set = []

count = 0
count2 = 0
for x in sum:
    data_set =  {"_instr_No.": count, 
                "addr": hex(count), 
                "data": x, 
                "mode": "1",
                "isfloat": "0"}
    full_set.append(data_set)
    #json_dump = json.dumps(data_set, indent=4)
    #print(json_dump)
    #print(",")
    count = count+1
    count2 = count2 + 1


for x in X_test[0]:
    count2 = 0
    data_set =  {"_instr_No.": count, 
                "addr": hex(count), 
                "data": x, 
                "mode": "1",
                "isfloat": "0"}
    
    full_set.append(data_set)
    #json_dump = json.dumps(data_set, indent=4)
    #print(json_dump)
    #print(",")
    count = count+1
    count2 = count2 + 1

for x in X_test[1]:
    count2 = 0
    data_set =  {"_instr_No.": count, 
                "addr": hex(count), 
                "data": x, 
                "mode": "1",
                "isfloat": "0"}
    
    full_set.append(data_set)
    #json_dump = json.dumps(data_set, indent=4)
    #print(json_dump)
    #print(",")
    count = count+1
    count2 = count2 + 1

for x in X_test[2]:
    count2 = 0
    data_set =  {"_instr_No.": count, 
                "addr": hex(count), 
                "data": x, 
                "mode": "1",
                "isfloat": "0"}
    
    full_set.append(data_set)
    #json_dump = json.dumps(data_set, indent=4)
    #print(json_dump)
    #print(",")
    count = count+1
    count2 = count2 + 1


json_dump = json.dumps(full_set, indent=4)
print(json_dump)

decision = clf.decision_function(X_test)
print(decision)



# Predict the value of the digit on the test subset
predicted = clf.predict(X_test)
#print(predicted)

print(f"Classification report for classifier {clf}:\n"
      f"{metrics.classification_report(y_test, predicted)}\n")
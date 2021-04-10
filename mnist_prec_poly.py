

import numpy as np

np.set_printoptions(threshold=np.inf)
# Standard scientific Python imports
import matplotlib.pyplot as plt
import struct
import json

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer


skdigits = datasets.load_digits()


n_samples = len(skdigits.images)
data = skdigits.images.reshape((n_samples, -1))

newdata = []
newtargets = []
count = 0
for x in skdigits.target:
    if(x == 0 or x == 1):
        newtargets.append(x)
        newdata.append(data[count])
    count = count+1

newdata = np.array(newdata)
newtargets = np.array(newtargets)
print(newdata.shape)
print(newtargets.shape)


# Create a classifier: a support vector classifier
clf = svm.SVC(kernel = 'poly', degree=2, coef0=1)


X_train, X_test, y_train, y_test = train_test_split(
    newdata, newtargets, test_size=0.7, shuffle=False)


clf.fit(X_train, y_train)
spv = clf.support_vectors_ 
print(spv)
d = clf.dual_coef_
b = clf.intercept_

gamma = 1 / (64 * X_train.var()) 
print(gamma)

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
sumMatrix = 0
for x in spv:
    x = x*gamma
    x = np.insert(x, 0, 1)
    x = x.reshape(65, 1)
    supportMatrix = x*np.transpose(x)
    supportMatrix = supportMatrix*coef[count]
    #print(supportMatrix)
    count = count+1
    sumMatrix = sumMatrix + supportMatrix
    #print(sumMatrix)


print(np.shape(sumMatrix))
x_tet = np.insert(test, 0, 1)
#x_tet = X_test[0]
out = x_tet.T.dot(sumMatrix)
print(out)
outdot = out.dot(x_tet)
print(outdot)

#print(out)
final = outdot + b
print(final)




# tv = np.reshape(tv, (64, 1))
# print(tv)
# sv0 = sv0.reshape(64, 1)
#print(sv0)z
# supportMatrix = sv0*np.transpose(sv0)
# print(supportMatrix)


full_set = [] 

count = 0
count2 = 0
for x in sumMatrix:
    count2 = 0
    for v in x:
        data_set =  {"_instr_No.": count, 
                    "addr": hex(count), 
                    "data": v, 
                    "mode": "1",
                    "isfloat": "0"}
        full_set.append(data_set)
        #json_dump = json.dumps(data_set, indent=4)
        #print(json_dump)
        #print(",")
        count = count+1
        count2 = count2 + 1


for x in x_tet:
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

x_tetone = np.insert(X_test[1], 0, 1)
for x in x_tetone:
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

x_tettwo = np.insert(X_test[2], 0, 1)
for x in x_tettwo:
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
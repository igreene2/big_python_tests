import numpy as np
# Standard scientific Python imports
import matplotlib.pyplot as plt
import struct
import json

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer


skdigits = datasets.load_breast_cancer()


n_samples = len(skdigits.images)
data = skdigits.images.reshape((n_samples, -1))




# Create a classifier: a support vector classifier
clf = svm.SVC(kernel = 'poly', degree=4, coef0=1)

X_train, X_test, y_train, y_test = train_test_split(
    newdata, newtargets, test_size=0.7, shuffle=False)


clf.fit(X_train, y_train)
gamma = 1 / (64 * X_train.var()) 
print(gamma)
spv = clf.support_vectors_ 
print(spv)
d = clf.dual_coef_
b = clf.intercept_

c = 1/gamma
print(c)


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

print(coef)
print(test)


full_set = []
for v in spv:
    count2 = 0
    for x in v:
        data_set =  {"_instr_No.": count, 
                    "addr": hex(count), 
                    "data": v[count2], 
                    "mode": "1",
                    "isfloat": "0"}
        full_set.append(data_set)
        #json_dump = json.dumps(data_set, indent=4)
        #print(json_dump)
        #print(",")
        count = count+1
        count2 = count2 + 1
    # remember to take gamma to the DEGREE!!!!
    data_set =  {"_instr_No.": count, 
                    "addr": hex(count), 
                    "data": coef[count_alpha]*(gamma*gamma*gamma*gamma), 
                    "mode": "1",
                    "isfloat": "0"}
    full_set.append(data_set)
    count_alpha = count_alpha + 1
  
    #print(",")
    count = count + 1


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
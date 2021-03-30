import numpy as np
# Standard scientific Python imports
import matplotlib.pyplot as plt
import struct
import json

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

def float_to_bin(num):
    return bin(struct.unpack('!I', struct.pack('!f', num))[0])[2:].zfill(32)

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
clf = svm.SVC(kernel = 'linear')


X_train, X_test, y_train, y_test = train_test_split(
    newdata, newtargets, test_size=0.7, shuffle=True)


clf.fit(X_train, y_train)
spv = clf.support_vectors_ 
print(spv)
d = clf.dual_coef_
print(np.shape(d))
print(d)


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
                    "data": hex(int(v[count2])), 
                    "mode": "1"}
        full_set.append(data_set)
        #json_dump = json.dumps(data_set, indent=4)
        #print(json_dump)
        #print(",")
        count = count+1
        count2 = count2 + 1
    data_set =  {"_instr_No.": count, 
                    "addr": hex(count), 
                    "data": hex(int(str(float_to_bin(coef[count_alpha])), 2)), 
                    "mode": "1"}
    full_set.append(data_set)
    count_alpha = count_alpha + 1
  
    #print(",")
    count = count + 1


for x in test:
    count2 = 0
    data_set =  {"_instr_No.": count, 
                "addr": hex(count), 
                "data": hex(int(x)), 
                "mode": "1"}
    full_set.append(data_set)
    #json_dump = json.dumps(data_set, indent=4)
    #print(json_dump)
    #print(",")
    count = count+1
    count2 = count2 + 1


json_dump = json.dumps(full_set, indent=4)
print(json_dump)

decision = clf.decision_function(X_test)
#print(decision)
#for x in decision:  
    #print(x)


# Predict the value of the digit on the test subset
predicted = clf.predict(X_test)
#print(predicted)


#print(f"Classification report for classifier {clf}:\n"
     # f"{metrics.classification_report(y_test, predicted)}\n")
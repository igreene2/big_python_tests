import numpy as np
import json
# Standard scientific Python imports
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
digits = load_breast_cancer()



#digits = np.load('MNISTcwtrain1000.npy')
#d = np.load('MNISTcwtest100.npy')
#data = datasets.load_digits()

#print(d.shape)
#for x in d:
 #   print(x)
#print(digits)
#print(d)


#_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
#for ax, image, label in zip(axes, digits.images, digits.target):
 #   ax.set_axis_off()
  #  ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
   # ax.set_title('Training: %i' % label)

# flatten the images
#n_samples = len(digits.images)
#data = digits.images.reshape((n_samples, -1))


# Create a classifier: a support vector classifier
clf = svm.SVC(kernel = 'linear')


# WHAT ARE X AND Y IN THIS CASE OF THE DATA I HAVE
# TRY TO USE MNIST WITH YIS NOTEBOOK AND SEE IF THAT IS BETTER

# Split data into 50% train and 50% test subsets
X_train, X_test, y_train, y_test = train_test_split(
    digits.data, digits.target, test_size=0.5, shuffle=False)

print(np.shape(X_train))
print(np.shape(X_test))


clf.fit(X_train, y_train)
spv = clf.support_vectors_ 
d = clf.dual_coef_
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

# for x in spv:
#     print(x)

# for x in spv[0]:
#     print(x)

# print(coef)
# print(test)
# print(X_test[1])
# print(X_test[2])

# print(spv[0].dot(X_test[0]))
sum = 0
count = 0
for x in spv:
    dot = round(x.dot(X_test[1]))
    print(dot)
    dot = round(dot*coef[count])
    print(dot)
    sum = sum + dot
    print(sum)
    count = count + 1
sum = sum + b
print(sum)

count = 0
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
    data_set =  {"_instr_No.": count, 
                    "addr": hex(count), 
                    "data": coef[count_alpha], 
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
print(np.shape(decision))
print(decision)


# Predict the value of the digit on the test subset
predicted = clf.predict(X_test)
print(predicted)




print(f"Classification report for classifier {clf}:\n"
      f"{metrics.classification_report(y_test, predicted)}\n")
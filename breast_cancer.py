import numpy as np
# Standard scientific Python imports
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
digits = load_breast_cancer()

for x in digits.data:
    print(x)

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

# Learn the digits on the train subset
clf.fit(X_train, y_train)
spv = clf.support_vectors_ 
print(np.shape(spv))
w = clf.coef_
print(np.shape(w))
print(w)
d = clf.dual_coef_
print(np.shape(d))

decision = clf.decision_function(X_test)
print(np.shape(decision))
#for x in decision:  
    #print(x)


# Predict the value of the digit on the test subset
predicted = clf.predict(X_test)
print(np.shape(predicted))
print(predicted)



print(f"Classification report for classifier {clf}:\n"
      f"{metrics.classification_report(y_test, predicted)}\n")
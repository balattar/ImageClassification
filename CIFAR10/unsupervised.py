# KNN
import numpy as np
import pickle
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

def unpickle(file):
   with open(file, 'rb') as fo:
       dict = pickle.load(fo, encoding='bytes')
   return dict

#Load data
path = 'F:\Penn State Study\Spring 2020\EE552\Term project\SVM\data\cifar-10-batches-py/'
str = ['1', '2', '3', '4', '5' ]
file = 'data_batch_'

for i in range(5):
   data_part = unpickle(path+file+str[i])
   if i == 0:
       temp_data = data_part[b'data']
       temp_label = data_part[b'labels']
   else:
       temp_data = np.concatenate((temp_data,data_part[b'data']), axis=0)
       temp_label = temp_label+data_part[b'labels']

train = temp_data
y_train = temp_label

# Select 5000 for training due to the running time
k_samples_train = np.linspace(1,49990,5000).astype(int)
train_sample = train[k_samples_train]
y_train_sample = np.array(y_train)[k_samples_train]
y_train_sample = y_train_sample.tolist()

data_part_test = unpickle(path+'test_batch')
test = data_part_test[b'data']
y_test = data_part_test[b'labels']

# Select 1000 for test due to the running time
k_samples_test = np.linspace(1,9990,1000).astype(int)
test_sample = test[k_samples_test]
y_test_sample = np.array(y_test)[k_samples_test]
y_test_sample = y_test_sample.tolist()


# KNN Classifier
k_neighbors = np.linspace(1,19,10).astype(int)

accuracy_test_knn = []
accuracy_train_knn = []
for k in k_neighbors:
   knn = KNeighborsClassifier(k)
   knn = knn.fit(train_sample,y_train_sample)
   y_pred_test =  knn.predict(test_sample)
   y_pred_train = knn.predict(train_sample)
   accuracy_test_knn.append(metrics.accuracy_score(y_test_sample,y_pred_test))
   accuracy_train_knn.append(metrics.accuracy_score(y_train_sample,y_pred_train))

# plot knn accuracy
fig,knn_plt = plt.subplots()
knn_plt.plot(k_neighbors,accuracy_train_knn,'-b',label = 'Training')
knn_plt.plot(k_neighbors,accuracy_test_knn,'--o',label = 'Test')
#plt.xscale('log')
plt.title('Training and Test Data Accuracy for KNN')
plt.xlabel('K Nearest Neighbors')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.show()
print(np.max(accuracy_test_knn))

# PCA+KNN
import numpy as np
import pickle
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

def unpickle(file):
   with open(file, 'rb') as fo:
       dict = pickle.load(fo, encoding='bytes')
   return dict

#Load data
path = 'F:\Penn State Study\Spring 2020\EE552\Term project\SVM\data\cifar-10-batches-py/'
str = ['1', '2', '3', '4', '5' ]
file = 'data_batch_'

for i in range(5):
   data_part = unpickle(path+file+str[i])
   if i == 0:
       temp_data = data_part[b'data']
       temp_label = data_part[b'labels']
   else:
       temp_data = np.concatenate((temp_data,data_part[b'data']), axis=0)
       temp_label = temp_label+data_part[b'labels']

train = temp_data
y_train = temp_label

data_part_test = unpickle(path+'test_batch')
test = data_part_test[b'data']
y_test = data_part_test[b'labels']

# Reduce Dimensions with PCA 0.9
pcaT = PCA(0.9)
train_sample = pcaT.fit_transform(train)
test_sample = pcaT.transform(test)

# KNN
knn = KNeighborsClassifier(1)
knn = knn.fit(train_sample,y_train)
y_pred_test = knn.predict(test_sample)
y_pred_train = knn.predict(train_sample)

#Training accuracy
correct = 0
total = len(y_train)
for i in range(len(y_train)):
   if y_pred_train[i] == y_train[i]:
       correct = correct + 1
print('Accuracy train: %d %%' % (
   100 * correct / total))

#Test accuracy
correct = 0
total = len(y_test)
for i in range(len(y_test)):
   if y_pred_test[i] == y_test[i]:
       correct = correct + 1
print('Accuracy test: %d %%' % (
   100 * correct / total))

#Confusion matrix
print(sklearn.metrics.confusion_matrix(y_test, y_pred_test))
#Classification report
print(sklearn.metrics.classification_report(y_test, y_pred_test))

# HOG+KNN
import numpy as np
import pickle
from sklearn import metrics
from skimage import feature as ft
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

def unpickle(file):
   with open(file, 'rb') as fo:
       dict = pickle.load(fo, encoding='bytes')
   return dict

#Load data
path = 'F:\Penn State Study\Spring 2020\EE552\Term project\SVM\data\cifar-10-batches-py/'
str = ['1', '2', '3', '4', '5' ]
file = 'data_batch_'

for i in range(5):
   data_part = unpickle(path+file+str[i])
   if i == 0:
       temp_data = data_part[b'data']
       temp_label = data_part[b'labels']
   else:
       temp_data = np.concatenate((temp_data,data_part[b'data']), axis=0)
       temp_label = temp_label+data_part[b'labels']

train = temp_data
y_train = temp_label

# Select 5000 for training due to the running time
k_samples_train = np.linspace(1,49990,5000).astype(int)
train_sample = train[k_samples_train]
y_train_sample = np.array(y_train)[k_samples_train]
y_train_sample = y_train_sample.tolist()

X_train = [None]*5000
for i in range(5000):
   img = train_sample[i]
   img = img.reshape([3, 32, 32])
   img = img.transpose([1, 2, 0])
   X_train[i] = img

X_train_array = np.array(X_train)

data_part_test = unpickle(path+'test_batch')
test = data_part_test[b'data']
y_test = data_part_test[b'labels']

# Select 1000 for test due to the running time
k_samples_test = np.linspace(1,9990,1000).astype(int)
test_sample = test[k_samples_test]
y_test_sample = np.array(y_test)[k_samples_test]
y_test_sample = y_test_sample.tolist()

X_test = [None]*1000
for i in range(1000):
   img = test_sample[i]
   img = img.reshape([3, 32, 32])
   img = img.transpose([1, 2, 0])
   X_test[i] = img

X_test_array = np.array(X_test)

# Define HOG
def hog(size,input):
   hog_features = np.zeros([size, 1764])
   for i in range(size):
       hog_features[i,:], hog_image = ft.hog(input[i],
                                             orientations=9,
                                             pixels_per_cell=(4,4),
                                             cells_per_block=(2,2),
                                             block_norm='L2',
                                             visualize=True,
                                             transform_sqrt=True,
                                             feature_vector=True)
   return hog_features

X_train_array_hog = hog(5000,X_train_array)
X_test_array_hog = hog(1000,X_test_array)

# KNN Classifier
k_neighbors = np.linspace(1,19,10).astype(int)

accuracy_test_knn = []
accuracy_train_knn = []
for k in k_neighbors:
   knn = KNeighborsClassifier(k)
   knn = knn.fit(X_train_array_hog,y_train_sample)
   y_pred_test =  knn.predict(X_test_array_hog)
   y_pred_train = knn.predict(X_train_array_hog)
   accuracy_test_knn.append(metrics.accuracy_score(y_test_sample,y_pred_test))
   accuracy_train_knn.append(metrics.accuracy_score(y_train_sample,y_pred_train))

# plot knn accuracy

fig,knn_plt = plt.subplots()
knn_plt.plot(k_neighbors,accuracy_train_knn,'-b',label = 'Training')
knn_plt.plot(k_neighbors,accuracy_test_knn,'--o',label = 'Test')
#plt.xscale('log')
plt.title('Training and Test Data Accuracy for KNN')
plt.xlabel('K Nearest Neighbors')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.show()
print(np.max(accuracy_test_knn))

# PCA+SVM
import numpy as np
from sklearn.decomposition import PCA
import pickle
import matplotlib.pyplot as plt
from sklearn import svm

def unpickle(file):
   with open(file, 'rb') as fo:
       dict = pickle.load(fo, encoding='bytes')
   return dict

#Load data
path = 'F:\Penn State Study\Spring 2020\EE552\Term project\SVM\data\cifar-10-batches-py/'
str = ['1', '2', '3', '4', '5' ]
file = 'data_batch_'

for i in range(5):
   data_part = unpickle(path+file+str[i])
   if i == 0:
       temp_data = data_part[b'data']
       temp_label = data_part[b'labels']
   else:
       temp_data = np.concatenate((temp_data,data_part[b'data']), axis=0)
       temp_label = temp_label+data_part[b'labels']

train = temp_data
y_train = temp_label

data_part_test = unpickle(path+'test_batch')
test = data_part_test[b'data']
y_test = data_part_test[b'labels']

##########
# #Uncomment this part for PCA+SVM, comment this part for SVM only
# #Apply PCA and find best parameter
# pca = np.linspace(0.1, 0.95, 11)
# train_pca = []
# for p in pca:
#     p = (p)
#     print(p)
#     pcaT = PCA(p)
#     train_proj = pcaT.fit_transform(train)
#     train_pca.append(train_proj)
#
# numComp = []
# for f in train_pca:
#     print(f.shape)
#     numComp.append(len(f[1]))
#
# # Plot PCA
# pca = pca * 100
# plt.plot(numComp,pca)
# plt.title('PCA Analysis')
# plt.xlabel('# of Principal Components')
# plt.ylabel('% of Overall Variabilty')
# plt.show()
#
# #Apply PCA and Reduce Dimensions with PCA 0.9
# pcaT = PCA(0.95)
# train = pcaT.fit_transform(train)
# test = pcaT.transform(test)
##########


# SVM Classifier
SVM = svm.LinearSVC(C=0.1)
SVM.fit(train, y_train)
y_train_pred = SVM.predict(train)
y_test_pred = SVM.predict(test)

#Training accuracy
correct = 0
total = len(y_train)
for i in range(len(y_train)):
   if y_train_pred[i] == y_train[i]:
       correct = correct + 1
print('Accuracy train: %d %%' % (
   100 * correct / total))

#Test accuracy
correct = 0
total = len(y_test)
for i in range(len(y_test)):
   if y_test_pred[i] == y_test[i]:
       correct = correct + 1
print('Accuracy test: %d %%' % (
   100 * correct / total))

C.5 HOG+SVM
import numpy as np
import pickle
import sklearn
from skimage import feature as ft
from sklearn.metrics import confusion_matrix
from sklearn import svm
import matplotlib.pyplot as plt
from PIL import Image

def unpickle(file):
   with open(file, 'rb') as fo:
       dict = pickle.load(fo, encoding='bytes')
   return dict

#Load data
path = 'F:\Penn State Study\Spring 2020\EE552\Term project\SVM\data\cifar-10-batches-py/'
str = ['1', '2', '3', '4', '5' ]
file = 'data_batch_'

for i in range(5):
   data_part = unpickle(path+file+str[i])
   if i == 0:
       temp_data = data_part[b'data']
       temp_label = data_part[b'labels']
   else:
       temp_data = np.concatenate((temp_data,data_part[b'data']), axis=0)
       temp_label = temp_label+data_part[b'labels']

train = temp_data
y_train = temp_label

X_train = [None]*50000
for i in range(50000):
   img = train[i]
   img = img.reshape([3, 32, 32])
   img = img.transpose([1, 2, 0])
   # img = Image.fromarray(img)  # array to image
   # img = img.convert("L")    # change color image to grayscale
   # img = np.array(img)     # image to array
   X_train[i] = img

X_train_array = np.array(X_train)

data_part_test = unpickle(path+'test_batch')
test = data_part_test[b'data']
y_test = data_part_test[b'labels']

X_test = [None]*10000
for i in range(10000):
   img = test[i]
   img = img.reshape([3, 32, 32])
   img = img.transpose([1, 2, 0])
   # img = Image.fromarray(img)  # array to image
   # img = img.convert("L")    # change color image to grayscale
   # img = np.array(img)     # image to array
   X_test[i] = img

X_test_array = np.array(X_test)

# Define HOG
def hog(size,input):
   hog_features = np.zeros([size, 1764])
   for i in range(size):
       hog_features[i,:], hog_image = ft.hog(input[i],
                                             orientations=9,
                                             pixels_per_cell=(4,4),
                                             cells_per_block=(2,2),
                                             block_norm='L2',
                                             visualize=True,
                                             transform_sqrt=True,
                                             feature_vector=True)
   return hog_features

X_train_array_hog = hog(50000,X_train_array)
X_test_array_hog = hog(10000,X_test_array)

# SVM Classifier training
SVM = svm.LinearSVC(C=0.1)
SVM.fit(X_train_array_hog, y_train)

# SVM Classifier predict
y_train_pred = SVM.predict(X_train_array_hog)
y_test_pred = SVM.predict(X_test_array_hog)

#Training accuracy
correct = 0
total = len(y_train)
for i in range(len(y_train)):
   if y_train_pred[i] == y_train[i]:
       correct = correct + 1
print('Accuracy train: %d %%' % (
   100 * correct / total))

#Test accuracy
correct = 0
total = len(y_test)
for i in range(len(y_test)):
   if y_test_pred[i] == y_test[i]:
       correct = correct + 1
print('Accuracy test: %d %%' % (
   100 * correct / total))

#Confusion matrix
print(sklearn.metrics.confusion_matrix(y_test, y_test_pred))
#Classification report
print(sklearn.metrics.classification_report(y_test, y_test_pred))

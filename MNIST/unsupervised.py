# Import MNIST Dataset
from sklearn import datasets, model_selection,metrics
X_raw, y = datasets.fetch_openml('mnist_784', version=1, return_X_y=True) # 28 x 28 ( 0 - 255)
 
 
# Subsample first n data points
n_sub = 5000
X_raw_sub = X_raw[0:n_sub]
y_sub = y[0:n_sub]
 
# Test on Raw pixel data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_raw_sub,y_sub)
 
# RAW data KNN classifier
from sklearn.neighbors import KNeighborsClassifier
import time 
import numpy as np
 
k_neighbors = [1]# [1, 2,3,4,5,6,7,8,9,10]
    
accuracy_test_knn = []
accuracy_train_knn = []
tstart = time.time()
for k in k_neighbors:
    knn = KNeighborsClassifier(k)
    knn = knn.fit(X_train,y_train)
    y_pred_test =  knn.predict(X_test)
    y_pred_train = knn.predict(X_train)
    accuracy_test_knn.append(metrics.accuracy_score(y_test,y_pred_test))
    accuracy_train_knn.append(metrics.accuracy_score(y_train,y_pred_train))
    telapsed = time.time() - tstart
    print(telapsed)
print(np.max(accuracy_test_knn))
 
 
from sklearn.model_selection import GridSearchCV 
from sklearn.svm import SVC
 
svm = SVC(gamma = 1, C = 0.1 , kernel ='linear')
svm.fit(X_train,y_train)
y_pred_test =  svm.predict(X_test)
accuracy_svm = metrics.accuracy_score(y_test,y_pred_test)
print(accuracy_svm)
 
 
 
# HOG Descriptor
 
from skimage.feature import hog
import time
n_samples = len(X_raw_sub)
 
 
# 8 ori 4x4 
# 8 ori 8x4 (knn 95) (svm  )
hog_features = np.zeros((n_samples,784))
tstart = time.time()
for n in range(0,n_samples):
    fn = X_raw_sub[n]
    fn = np.reshape(fn,(28,28))
    fn = fn.astype(np.uint8)
    feature_hog, imhog = hog(fn, orientations=8, pixels_per_cell=(4,2),
                    cells_per_block=(1, 1), visualize=True, multichannel=False)
    feature_hog = feature_hog.flatten()
    #print(feature_hog)
    #print(type(feature_hog))
    #print(len(feature_hog))
    hog_features[n] = feature_hog
    telapsed = time.time() - tstart
    print(telapsed)
    
#hog_features = np.asarray(hog_features)
 
# Split Data 
 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(hog_features,y_sub)
 
# HOG KNN classifier
from sklearn.neighbors import KNeighborsClassifier
 
knn = KNeighborsClassifier(5)
knn = knn.fit(X_train,y_train)
 
k_neighbors = [1]#, 2,3,4,5,6,7,8,9,10]
    
accuracy_test_knn = []
accuracy_train_knn = []
tstart = time.time()
for k in k_neighbors:
    knn = KNeighborsClassifier(k)
    knn = knn.fit(X_train,y_train)
    y_pred_test =  knn.predict(X_test)
    y_pred_train = knn.predict(X_train)
    accuracy_test_knn.append(metrics.accuracy_score(y_test,y_pred_test))
    accuracy_train_knn.append(metrics.accuracy_score(y_train,y_pred_train))
    telapsed = time.time() - tstart
    print(telapsed)
 
print(np.max(accuracy_test_knn))
 
 
from sklearn.model_selection import GridSearchCV 
from sklearn.svm import SVC
 
svm = SVC(gamma = 1, C = 0.1 , kernel ='linear')
svm.fit(X_train,y_train)
y_pred_test =  svm.predict(X_test)
accuracy_svm = metrics.accuracy_score(y_test,y_pred_test)
print(accuracy_svm)
 
 
# In[77]:
 
 
# PCA 
from sklearn.decomposition import PCA
pca = PCA(0.95)
proj = pca.fit_transform(X_raw_sub)
numComp = len(proj[1])
print(numComp)
 
 
# Split Data 
 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(proj,y_sub)
 
 
# pca KNN classifier
from sklearn.neighbors import KNeighborsClassifier
 
knn = KNeighborsClassifier(5)
knn = knn.fit(X_train,y_train)
 
k_neighbors = [1]#, 2,3,4,5,6,7,8,9,10]
    
accuracy_test_knn = []
accuracy_train_knn = []
tstart = time.time()
for k in k_neighbors:
    knn = KNeighborsClassifier(k)
    knn = knn.fit(X_train,y_train)
    y_pred_test =  knn.predict(X_test)
    y_pred_train = knn.predict(X_train)
    accuracy_test_knn.append(metrics.accuracy_score(y_test,y_pred_test))
    accuracy_train_knn.append(metrics.accuracy_score(y_train,y_pred_train))
    telapsed = time.time() - tstart
    print(telapsed)
 
print(np.max(accuracy_test_knn))
 
 
from sklearn.model_selection import GridSearchCV 
from sklearn.svm import SVC
 
svm = SVC(gamma = 1, C = 0.1 , kernel ='linear')
svm.fit(X_train,y_train)
y_pred_test =  svm.predict(X_test)
accuracy_svm = metrics.accuracy_score(y_test,y_pred_test)
print(accuracy_svm)

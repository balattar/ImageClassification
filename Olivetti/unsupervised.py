
from sklearn import datasets , metrics
import numpy as np
 
# Import face data
# Pixel data is already gray scaled and normalized
faces,person = datasets.fetch_olivetti_faces( return_X_y=True)
faces = (faces*255).astype(int)
 
# Perfrom PCA 
from sklearn.decomposition import PCA
pca = np.linspace(0.1,0.95,11)
faces_pca = []
for p in pca:
    p = (p)
    print(p)
    pcaT = PCA(p)
    faces_proj = pcaT.fit_transform(faces)
    faces_pca.append(faces_proj)
    
numComp = []
for f in faces_pca:
    print(f.shape)
    numComp.append(len(f[1]))
numComp2 = 100*np.array(numComp)/len(faces[1])
pca = 100*pca
 
# Plot PCA 
 
import matplotlib.pyplot as plt
plt.plot(numComp,pca)
#plt.title('PCA Analysis')
plt.xlabel('# of Principal Components')
plt.ylabel('% of Overall Variability')
plt.ylim(0,100)
plt.xlim(0,130)
plt.show
plt.savefig('PCA')
 
 
# Reduce Dimensions with PCA 0.9
pcaT = PCA(0.95)
faces_proj = pcaT.fit_transform(faces)
print(len(faces_proj[1]))
 
# Split Data 
 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(faces_proj,person)
 
 
print(len(X_train)/(len(X_train)+len(X_test)))
 
 
# In[68]:
 
 
from sklearn.neighbors import KNeighborsClassifier
k_neighbors = np.linspace(1,40,40).astype(int)
 
accuracy_test_knn = []
accuracy_train_knn = []
for k in k_neighbors:
    knn = KNeighborsClassifier(k)
    knn = knn.fit(X_train,y_train)
    y_pred_test =  knn.predict(X_test)
    y_pred_train = knn.predict(X_train)
    accuracy_test_knn.append(metrics.accuracy_score(y_test,y_pred_test))
    accuracy_train_knn.append(metrics.accuracy_score(y_train,y_pred_train))
 
accuracy_test_knn = np.array(accuracy_test_knn)*100
accuracy_train_knn = np.array(accuracy_train_knn)*100
 
# plot knn accuracy
 
import matplotlib.pyplot as plt
fig,knn_plt = plt.subplots()
knn_plt.plot(k_neighbors,accuracy_train_knn,'-b',label = 'Training')
knn_plt.plot(k_neighbors,accuracy_test_knn,'--o',label = 'Test')
#plt.xscale('log')
#plt.title('Training and Test Data Accuracy for KNN')
plt.xlabel('K Nearest Neighbors')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.show()
print(np.max(accuracy_test_knn))
fig.savefig('knnpcaolivtetti')
 
# Search for best SVM hyperparameters kernal, C , gamma
 
from sklearn.model_selection import GridSearchCV 
from sklearn.svm import SVC
 
hp_grid = {'C': [0.1, 1, 10, 100, 1000],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['linear','poly','rbf']}  
grid = GridSearchCV(SVC(), hp_grid, refit = True, verbose = 4) 
 
grid.fit(X_train,y_train)
 
print(grid.best_params_)
print(grid.best_score_)
 
# HOG Descriptor
 
from skimage.feature import hog
n_samples = len(faces)
 
 
# 8 ori 4x4 
# 8 ori 8x4 (knn 95) (svm  )
hog_features = np.zeros((n_samples,1024))
for n in range(0,n_samples):
    fn = faces[n]
    fn = np.reshape(fn,(64,64))
    fn = fn.astype(np.uint8)
    feature_hog, imhog = hog(fn, orientations=8, pixels_per_cell=(8, 4),
                    cells_per_block=(1, 1), visualize=True, multichannel=False)
    feature_hog = feature_hog.flatten()
    #print(feature_hog)
    #print(type(feature_hog))
    #print(len(feature_hog))
    hog_features[n] = feature_hog
    
#hog_features = np.asarray(hog_features)
 
 
# Split Data 
 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(hog_features,person)
 
# HOG KNN classifier
from sklearn.neighbors import KNeighborsClassifier
 
knn = KNeighborsClassifier(5)
knn = knn.fit(X_train,y_train)
 
k_neighbors = [1, 2,3,4, 5,6,7,8,9,10]
    
accuracy_test_knn = []
accuracy_train_knn = []
for k in k_neighbors:
    knn = KNeighborsClassifier(k)
    knn = knn.fit(X_train,y_train)
    y_pred_test =  knn.predict(X_test)
    y_pred_train = knn.predict(X_train)
    accuracy_test_knn.append(metrics.accuracy_score(y_test,y_pred_test))
    accuracy_train_knn.append(metrics.accuracy_score(y_train,y_pred_train))
    
# plot knn accuracy
 
import matplotlib.pyplot as plt
fig,knn_plt = plt.subplots()
knn_plt.plot(k_neighbors,accuracy_train_knn,'-b',label = 'Training')
knn_plt.plot(k_neighbors,accuracy_test_knn,'--o',label = 'Test')
#plt.xscale('log')
#plt.title('Training and Test Data Accuracy for KNN')
plt.xlabel('K Nearest Neighbors')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.show()
print(np.max(accuracy_test_knn))
fig.savefig('knnhog')
 
# Search for best SVM hyperparameters kernal, C , gamma
 
from sklearn.model_selection import GridSearchCV 
from sklearn.svm import SVC
 
hp_grid = {'C': [0.1, 1, 10, 100, 1000],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['linear','poly','rbf','sigmoid']}  
grid = GridSearchCV(SVC(), hp_grid, refit = True, verbose = 1) 
 
grid.fit(X_train,y_train)
print(grid.best_params_)
print(grid.best_score_)
 
# Test on Raw pixel data
X_train, X_test, y_train, y_test = train_test_split(faces,person)
 
# RAW data KNN classifier
 
knn = KNeighborsClassifier(5)
knn = knn.fit(X_train,y_train)
 
k_neighbors = [1, 3,4,5,6,7,8,9,10]
    
accuracy_test_knn = []
accuracy_train_knn = []
for k in k_neighbors:
    knn = KNeighborsClassifier(k)
    knn = knn.fit(X_train,y_train)
    y_pred_test =  knn.predict(X_test)
    y_pred_train = knn.predict(X_train)
    accuracy_test_knn.append(metrics.accuracy_score(y_test,y_pred_test))
    accuracy_train_knn.append(metrics.accuracy_score(y_train,y_pred_train))
    
# plot knn accuracy
accuracy_train_knn = np.array(accuracy_train_knn)*100
accuracy_test_knn = np.array(accuracy_test_knn)*100
 
import matplotlib.pyplot as plt
fig,knn_plt = plt.subplots()
knn_plt.plot(k_neighbors,accuracy_train_knn,'-b',label = 'Training')
knn_plt.plot(k_neighbors,accuracy_test_knn,'--o',label = 'Test')
#plt.xscale('log')
#plt.title('Training and Test Data Accuracy for KNN')
plt.xlabel('K Nearest Neighbors')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.show()
print(np.max(accuracy_test_knn))
fig.savefig('knnraw')
 
 
# Search for best SVM hyperparameters kernal, C , gamma
 
from sklearn.model_selection import GridSearchCV 
from sklearn.svm import SVC
 
hp_grid = {'C': [0.1, 1, 10, 100, 1000],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['linear','poly','rbf']}  
grid = GridSearchCV(SVC(), hp_grid, refit = True, verbose = 3) 
 
grid.fit(X_train,y_train)
print(grid.best_params_)
print(grid.best_score_)

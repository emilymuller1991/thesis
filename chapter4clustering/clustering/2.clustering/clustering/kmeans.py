from sklearn.cluster import KMeans
from scipy.stats import mode
import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class accuracy():
    def __init__(self,n_clusters):
        self.n_clusters = n_clusters
        #self.test_x = test_x
        #self.y_test = y_test
        #self.model = model
    def fit(self, test_x, test_y):
        fitted = KMeans(n_clusters=self.n_clusters, random_state=0).fit(test_x)
        return self.accuracy(fitted, test_y)

    def fit_raw(self, test_x, test_y):
        fitted = KMeans(n_clusters=self.n_clusters, random_state=0).fit(test_x)
        return self.accuracy(fitted, test_y)

    def fit_z(self, test_x, test_y, model=None):
        mean, logvar = model.encode(test_x)
        z = model.reparameterize(mean, logvar)
        fitted = KMeans(n_clusters=self.n_clusters, random_state=0).fit(z)
        return self.accuracy(fitted, test_y)

    def fit_ms(self, test_x, test_y, model=None):
        mean, logvar = model.encode(test_x)
        y = tf.concat([mean, logvar], 1)
        #z = model.reparameterize(mean, logvar)
        fitted = KMeans(n_clusters=self.n_clusters, random_state=0).fit(y)
        return self.accuracy(fitted, test_y)
    
    def accuracy(self, fitted, test_y):
        real_pred = np.zeros_like(fitted.labels_)
        for cat in range(10):
            idx = fitted.labels_ == cat
            lab = test_y[idx]
            if len(lab) == 0:
                continue
            real_pred[fitted.labels_ == cat] = mode(lab).mode[0]
        return np.mean(real_pred == test_y)
	
# class Mammals:
#     def __init__(self):
#         ''' Constructor for this class. '''
#         # Create some member animals
#         self.members = ['Tiger', 'Elephant', 'Wild Cat']
 
 
#     def printMembers(self):
#         print('Printing members of the Mammals class')
#         for member in self.members:
#             print('\t%s ' % member)

# # Cluster Accuracy
# def kmeans(test_x, y_test, model=None):
#     if model is None:
#         kmeans = KMeans(n_clusters=10, random_state=0).fit(test_x)
#     else:
#         mean, logvar = model.encode(test_x)
#         z = model.reparameterize(mean, logvar)
#         kmeans = KMeans(n_clusters=10, random_state=0).fit(z)
#     real_pred = np.zeros_like(kmeans.labels_)
#     for cat in range(10):
#         idx = kmeans.labels_ == cat
#         lab = y_test[idx]
#         if len(lab) == 0:
#             continue
#         real_pred[kmeans.labels_ == cat] = mode(lab).mode[0]
#     return np.mean(real_pred == y_test)
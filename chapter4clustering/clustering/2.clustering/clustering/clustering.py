from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.cluster import SpectralClustering
from sklearn.cluster import DBSCAN
#from sklearn.cluster import OPTICS, cluster_optics_dbscan
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabaz_score
from sklearn.metrics import silhouette_score
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn import mixture
from scipy.stats import mode
import numpy as np
#import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class accuracy():
    def __init__(self,n_clusters=None, labels=None, scorer='accuracy'):
        #self.scorer = 'accuracy'
        self.scorers = {'SS': make_scorer(silhouette_score)} #, 'CSH': make_scorer(calinski_harabaz_score)}
        self.n_clusters = n_clusters
        self.labels = labels
        #self.test_x = test_x
        #self.y_test = y_test
        #self.model = model

    def kmeans(self, X):
        #X = StandardScaler().fit_transform(X)
        ### user defined parameters
        fitted = KMeans(n_clusters=self.n_clusters, random_state=0).fit(X)
        # for each permutation of parameters performed by GridCVsearch, now we want to calculate Score from other functions
        scs = silhouette_score(X, fitted.labels_)
        chs = calinski_harabaz_score(X, fitted.labels_)
        dbs = davies_bouldin_score(X, fitted.labels_)
        return scs, chs, dbs, fitted.labels_, fitted.transform(X)**2

    def aprop(self, X):
        ### user defined parameters
        parameters = {'preference':[-50], 'damping':[0.5, 0.6, 0.7, 0.8, 0.9]}
        AP = AffinityPropagation()
        clf = GridSearchCV(AP, parameters, 'accuracy', cv=[(slice(None), slice(None))])
        fitted = clf.fit(X, self.labels)
        # for each permutation of parameters performed by GridCVsearch, now we want to calculate Score from other functions
        scs = []
        chs = []
        cbs = []
        for p in fitted.cv_results_['params']:
            AP = AffinityPropagation(damping = p['damping'], preference=p['preference']).fit(X)
            scs.append(silhouette_score(X, AP.labels_))
            chs.append(calinski_harabaz_score(X, AP.labels_))
            cbs.append(davies_bouldin_score(X, AP.labels_))
        return scs, chs, cbs, fitted

    # def mean_shift(self, X, bin_seeding=True, n_samples=500, quantile=0.2):
        # bandwidth = estimate_bandwidth(X, quantile=quantile, n_samples=n_samples)
        # ms = MeanShift(bandwidth=bandwidth, bin_seeding=bin_seeding)
        # fitted = ms.fit(X)
        # if self.labels is not None:
        #     return self.accuracy(fitted.labels_, self.labels), fitted.labels_
        # return 0, fitted

    def spectral(self, X, assign_labels="discretize"):
        fitted = SpectralClustering(self.n_clusters, assign_labels=assign_labels, random_state=0).fit(X)
        if self.labels is not None:
            return self.accuracy(fitted.labels_, self.labels), fitted.labels_
        return 0, fitted

    def dbscan(self, X, eps, min_samples):
        #X = StandardScaler().fit_transform(X)
        fitted = DBSCAN(eps=eps, min_samples = min_samples).fit(X)
        scs = silhouette_score(X, fitted.labels_)
        chs = calinski_harabaz_score(X, fitted.labels_)
        dbs = davies_bouldin_score(X, fitted.labels_)
        return scs, chs, dbs, fitted.labels_

    def optics(self, X, min_samples=10, xi=.1, min_cluster_size=.1):
        clust = OPTICS(min_samples=min_samples, xi=xi, min_cluster_size=min_cluster_size)
        fitted = clust.fit(X)
        if self.labels is not None:
            return self.accuracy(fitted.labels_, self.labels), fitted.labels_
        return 0, fitted

    def EM(self, X, covariance_type='full'):
        fitted = mixture.GaussianMixture(self.n_clusters, covariance_type=covariance_type).fit(X)
        labels = fitted.predict(X)
        if self.labels is not None:
            return self.accuracy(labels, self.labels), labels
        return 0, fitted      
    
    def accuracy(self, fitted, test_y):
        real_pred = np.zeros_like(fitted)
        for cat in range(3):
            idx = fitted == cat
            lab = test_y[idx]
            if len(lab) == 0:
                continue
            real_pred[fitted == cat] = mode(lab).mode[0]
        return np.mean(real_pred == test_y)

    # def kmeans(self, test_x):
    #     fitted = KMeans(n_clusters=self.n_clusters, random_state=0).fit(test_x)
    #     if self.labels is not None:
    #         return self.accuracy(fitted.labels_, self.labels), fitted.labels_
    #     return 0, fitted.labels_

    # def aprop(self, X, pref=-50):
    #     parameters = {'preference':[-50,50], 'damping':[0.5, 0.7]}
    #     AP = AffinityPropagation()
    #     clf = GridSearchCV(AP, parameters, 'accuracy')
    #     fitted = clf.fit(X, self.labels)
    #     if self.labels is not None:
    #         return self.accuracy(fitted.labels_, self.labels), fitted.labels_
    #     return 0, fitted
	
#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
import copy

### data preprocessing
d_data = pd.read_csv('dataset_diabetes/diabetic_data.csv')
d_data[d_data['medical_specialty']=='?']

notnulld_data1 = d_data.drop(['weight','payer_code','medical_specialty'], axis = 1)
notnulld_data1 = notnulld_data1[notnulld_data1 != '?']
notnulld_data1 = notnulld_data1.dropna()

cleaned_data1 = notnulld_data1.drop(['encounter_id','patient_nbr','race', 'gender'], axis = 1)
cleaned_data1[['diag_1','diag_2','diag_3']] = cleaned_data1[['diag_1','diag_2','diag_3']].apply(lambda x: pd.to_numeric(x, errors='coerce')).dropna()

cats1 = cleaned_data1.iloc[:,15:]
cats1['age'] = cleaned_data1['age']

cleaned_data1 = cleaned_data1.drop(list(cats1.columns), axis=1)

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
enc = LabelEncoder()
sc = MinMaxScaler()

cats1 = cats1.apply(enc.fit_transform)
cleaned_data1[['num_lab_procedures','num_medications','number_diagnoses']] = sc.fit_transform(cleaned_data1[['num_lab_procedures','num_medications','number_diagnoses']])
cleaned_data1[['diag_1','diag_2','diag_3']] = sc.fit_transform(cleaned_data1[['diag_1','diag_2','diag_3']])
cleaned_data1.dropna(inplace=True)

final_data1 = pd.concat((cleaned_data1,cats1), axis = 1)
final_data1 = final_data1.dropna()

columns1 = list(final_data1.columns)
readmitted = final_data1['readmitted']

final_data1 = final_data1.drop(['readmitted'], axis = 1)
final_data_scaled1 = sc.fit_transform(final_data1)

### Gaussian Mixture Model for Kmeans++
from sklearn.metrics import silhouette_score as sc

class kmeans_plusplus:
    
    def __init__(self,k):
        self.data = None
        self.k = k
        self.T = None
        self.max_iter = 20
        
    def initialize_centroid(self):
        centroids = []
        min_ = np.min(self.data, axis = 0)
        max_ = np.max(self.data, axis = 0)
        centroids.append(np.random.uniform(min_, max_))
        for i in range(1,self.k):
            dist = []
            for point in self.data:
                d = 9999
                for j in range(len(centroids)):
                    current_cdist = np.sqrt(np.sum((point - centroids[j])**2))
                    d = min(d, current_cdist)
                    
                dist.append(d)
            
            dist = np.array(dist)
            centroids.append(self.data[np.argmax(dist), :])
        return centroids
        
    def assign_clusters(self,centroids):
        clusters = [[] for i in range(self.k)]
        
        for i, point in enumerate(self.data):
            centroid_num = np.argmin(np.sqrt(np.sum((point - centroids)**2)))
            
            clusters[centroid_num].append(i)
            
        return clusters
    
    def update_centroids(self,clusters):
        new_centroids = np.zeros((self.k, self.data.shape[1]))
        
        for i,cluster in enumerate(clusters):
            new_centroid = np.mean(self.data[cluster], axis = 0)
            
            new_centroids[i] = new_centroid
            
        return new_centroids
    
    def calculate(self,centroids, previous_centroids):
        diff = np.sum(np.subtract(previous_centroids, centroids))
        
        return diff
    
    def fit(self,data,T):
        self.data = data
        self.T = T
        
        current_centroids = self.initialize_centroid()
        
        iter = 0
        while iter < self.max_iter:
            clusters = self.assign_clusters(current_centroids)
            previous_centroids = current_centroids
            current_centroids = self.update_centroids(clusters)
            diff = self.calculate(current_centroids, previous_centroids)
            
            if diff < self.T:
                final_clusters = np.zeros((self.data.shape[0]))
                for i,cluster in enumerate(clusters):
                    for j in cluster:
                        final_clusters[j] = i
                
                return final_clusters
            iter += 1
        
        final_clusters = np.zeros((self.data.shape[0]))
        for i,cluster in enumerate(clusters):
            for j in cluster:
                final_clusters[j] = i
        return final_clusters
    
class GMMplus:
    
    def __init__(self, data, k, eps):
        self.data = data
        self.k = k
        self.eps = eps
        self.maxiter = 10
        self.mu = None
        self.sigma = None
        self.prior = None
        self.w = None
        self.d = data.shape[0]
        self.f = data.shape[1]
        
    def begin(self):
        mu = np.zeros((self.k,self.f),dtype = float)
        sigma = np.zeros((self.k,self.f,self.f), dtype = float)
        prior = []
        for i in range(self.k):
            mu[i] = self.data[i]
            sigma[i] = np.identity(self.f)
            prior.append(1/self.k)   
        prior = np.asarray(prior, dtype = float)
        return mu, sigma, prior
        
    def expectation(self):
        w = np.zeros((self.k,self.d), dtype = float)
        
        for i in range(self.k):
            w[i] = (st.multivariate_normal.pdf(self.data, mean = self.mu[i], cov = self.sigma[i], allow_singular=True)) * prior[i]
            w = w/np.sum(w, axis = 0)
            
        return w
    
    def maximize(self, w):
        for i in range(self.k):
            self.mu[i] = ((np.sum(self.data * w.T[:,i][:,np.newaxis], axis = 0))/np.sum(w, axis = 1)[i])
            
            centered = (self.data - self.mu[i])* w.T[:,i][:,np.newaxis]
            t = (self.data - self.mu[i]).T
            self.sigma[i] = np.dot(t, centered)/ np.sum(w, axis = 1)[i]
            
            self.prior = np.sum(w, axis = 1)/ self.d
            
            
    def em(self):
        self.mu, self.sigma, self.prior = self.begin()
        prev_mu = copy.deepcopy(self.mu)
        self.w = self.expectation()
        self.maximize(self.w)
            
    def fit(self):
        km = kmeansplusplus(self.k)
        clusters = km.fit(self.data, 0.01)
        
        q = np.zeros((self.d, self.k))
        q[np.arange(self.d).astype(int), clusters.astype(int)] = 1
        
        self.em()
        
        for i in range(self.d):
            for j in range(self.k):
                prob = st.multivariate_normal.pdf(x = self.data[i], mean = self.mu[j], cov = self.sigma[j], allow_singular=True)
                temp = np.sum(self.w, axis = 0) / self.d
                q[i,j] = prob * temp[j]
            q[i] /= np.sum(q[i])
        
        final_clusters = []
        for i in range(self.d):
            final_clusters.append(np.argmax(q[i]))
        
        return final_clusters
    
### silhoutte score
total_errors = []
for k in range(2,6):
    errors = []
    for i in range(20):
        gmm = GMMplus(pca_data, k, 0.01)
        preds = gmm.fit()
        errors.append(sc(pca_data, preds))
    total_errors.append(errors)
    
### plotting the score
errors_gm_plus= pd.DataFrame(total_errors)
errors_gm_plus = errors_gm_plus.transpose()
errors_gm_plus.columns = [2,3,4,5]

plt.figure(figsize=(10,10))
errors_gm_plus.boxplot()
plt.ylabel('Score')
plt.xlabel('Cluster Count')
plt.show()

### Davies-Bouldin score
from sklearn.metrics import davies_bouldin_score as db

total_errors = []
for k in range(2,6):
    errors = []
    for i in range(20):
        gmm = GMMplus(pca_data, k, 0.01)
        preds = gmm.fit()
        errors.append(db(pca_data, preds))
    total_errors.append(errors)
    
### plotting the score
errors_gm_plus_db = pd.DataFrame(total_errors)
errors_gm_plus_db = errors_gm_plus_db.transpose()
errors_gm_plus_db.columns = [2,3,4,5]

plt.figure(figsize=(10,10))
errors_gm_plus_db.boxplot()
plt.ylabel('Score')
plt.xlabel('Cluster Count')
plt.show()


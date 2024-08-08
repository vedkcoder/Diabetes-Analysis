#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
import copy

class kmeans:
    
    def __init__(self,k):
        self.data = None
        self.k = k
        self.T = None
        self.max_iter = 20
        
    def initialize_centroid(self):
        min_ = np.min(self.data, axis = 0)
        max_ = np.max(self.data, axis = 0)
        centroids = [np.random.uniform(min_, max_) for _ in range(self.k)]
            
        return centroids
        
    def assign_clusters(self,centroids):
        clusters = [[] for i in range(self.k)]
        
        for i, point in enumerate(self.data):
            centroid_num = np.argmin(np.sqrt(np.sum((point - centroids)**2, axis = 1)))
            
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
    
class GMM:
    
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
        for i in range(self.maxiter):
            prev_mu = copy.deepcopy(self.mu)
            self.w = self.expectation()
            self.maximize(self.w)
            diff = 0
            for j in range(self.k):
                diff += np.sqrt(np.sum(np.square(np.abs(self.mu[j] - prev_mu[j]))))
                
            if diff < self.eps:
                return
                
    def fit(self):
        km = kmeans(self.k)
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
    


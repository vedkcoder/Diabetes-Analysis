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

### Poisson Mixture Model
import numpy as np
from scipy.stats import poisson
from scipy.optimize import minimize

def pmm(k, data):
    def log_likelihood(theta, data):
    k = len(theta) // 2
    p = theta[:k]
    lambdas = theta[k:]
    ll = np.log(np.array([p[i] * poisson.pmf(data, lambdas[i]) for i in range(k)]).sum(axis=0)).sum()
    return -ll

    theta0 = np.concatenate([np.ones(k) / k, [6, 12]])
    result = minimize(log_likelihood, theta0, args=(final_data_scaled1,))
    p_est, lambdas_est = result.x[:k], result.x[k:]

    pmfs = np.array([p_est[i] * poisson.pmf(final_data_scaled1, lambdas_est[i]) for i in range(k)])
    labels = np.argmax(pmfs, axis=0)

    return labels

### Silhouette score for PMM
total_errors = []
for k in range(2,6):
    print('For k : ', k)
    errors = []
    for i in range(20):
        preds = pmm(k,final_data_scaled1)
        errors.append(sc(final_data_scaled1, preds))
    total_errors.append(errors)
    
### plotting the score
errors_pm = pd.DataFrame(total_errors)
errors_pm = errors_pm.transpose()
errors_pm.columns = [2,3,4,5]

plt.figure(figsize=(10,10))
errors_pm.boxplot()
plt.ylabel('Score')
plt.xlabel('Cluster Count')
plt.show()

### Davis-Bouldin Score
from sklearn.metrics import davies_bouldin_score as db

total_errors = []
for k in range(2,6):
    errors = []
    for i in range(20):
        preds = pmm(k,final_data_scaled1)
        errors.append(db(final_data_scaled1, preds))
    total_errors.append(errors)
    
### plotting the score
errors_pm_db = pd.DataFrame(total_errors)
errors_pm_db = errors_pm_db.transpose()
errors_pm_db.columns = [2,3,4,5]

plt.figure(figsize=(10,10))
errors_pm_db.boxplot()
plt.ylabel('Score')
plt.xlabel('Cluster Count')
plt.show()


### Multinomial Mixture Model
import numpy as np
from scipy.stats import multinomial
from scipy.optimize import minimize

def mmm(k, data):

    def log_likelihood(theta, data):
        k = len(theta) // 4
        p = theta[:k]
        mus = theta[k:3*k].reshape(-1, 3)
        sigmas = theta[3*k:].reshape(-1, 3, 3)
        ll = np.log(np.array([p[i] * multinomial.pmf(data, n=np.sum(data), p=mus[i]) for i in range(k)]).sum(axis=0)).sum()
        return -ll

    theta0 = np.concatenate([np.ones(k) / k, np.random.rand(k * 3), np.tile(np.eye(3).flatten(), k)])
    result = minimize(log_likelihood, theta0, args=(final_data_scaled1,))
    p_est, mus_est, sigmas_est = result.x[:k], result.x[k:3*k].reshape(-1, 3), result.x[3*k:].reshape(-1, 3, 3)

    pmfs = np.array([p_est[i] * multinomial.pmf(final_data_scaled1, n=np.sum(final_data_scaled1, axis=1), p=mus_est[i]) for i in range(k)])

    labels = np.argmax(pmfs, axis=0)
    
    return labels

### Silhouette score for MMM
total_errors = []
for k in range(2,6):
    errors = []
    for i in range(20):
        preds = mmm(k,final_data_scaled1)
        errors.append(sc(final_data_scaled1, preds))
    total_errors.append(errors)
    
### plotting the score
errors_mm = pd.DataFrame(total_errors)
errors_mm = errors_mm.transpose()
errors_mm.columns = [2,3,4,5]

plt.figure(figsize=(10,10))
errors_mm.boxplot()
plt.ylabel('Score')
plt.xlabel('Cluster Count')
plt.show()

### Davis-Bouldin Score
from sklearn.metrics import davies_bouldin_score as db

total_errors = []
for k in range(2,6):
    print('For k : ', k)
    errors = []
    for i in range(20):
        preds = mmm(k,final_data_scaled1)
        errors.append(db(final_data_scaled1, preds))
    total_errors.append(errors)
    
### plotting the score
errors_mm_db = pd.DataFrame(total_errors)
errors_mm_db = errors_mm_db.transpose()
errors_mm_db.columns = [2,3,4,5]

plt.figure(figsize=(10,10))
errors_mm_db.boxplot()
plt.ylabel('Score')
plt.xlabel('Cluster Count')
plt.show()


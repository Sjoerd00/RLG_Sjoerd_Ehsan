# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 20:19:02 2023

@author: Steve
"""
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load your data into a pandas dataframe
df = pd.read_csv("RLdata\\zozo_Context_80items.csv")

# Define the list of columns to be converted to ordinal format
columns = ['user_feature_0', 'user_feature_1', 'user_feature_2', 'user_feature_3']

# Convert each column to ordinal format
for column in columns:
    df[column] = pd.Categorical(df[column])
    df[column] = df[column].cat.codes

# Select the 4 columns you want to cluster on
data = df[columns]

# Fit the k-means model to the data and compute the inertia on the fitted model
SSE = []
for cluster in range(1,20):
    kmeans = KMeans(n_clusters = cluster)
    kmeans.fit(data)
    SSE.append(kmeans.inertia_)

# Convert the results into a dataframe and plotting them
frame = pd.DataFrame({'Cluster':range(1,20), 'SSE':SSE})
plt.figure(figsize=(12,6))
plt.plot(frame['Cluster'], frame['SSE'], marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')

#%% set clusters
for cs in [3, 4, 10]:
    kmeans = KMeans(n_clusters = cs)
    kmeans.fit(data)
    df[f'class_{cs}'] = kmeans.predict(data)

df.to_csv('dataProcessed1.csv')

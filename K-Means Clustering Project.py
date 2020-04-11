#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns
sns.set()


# In[5]:


data = pd.read_csv('C:\\Users\\Asus\\Desktop\\MachineLearning-Coursera\\3.01. Country clusters.csv')
data


# In[6]:


plt.scatter(data['Longitude'], data['Latitude'])
plt.xlim(-180,180)
plt.ylim(-90,90)
plt.show


# In[7]:


x = data.iloc[:,1:3]


# In[8]:


x


# In[16]:


kmeans = KMeans(3)


# In[17]:


kmeans.fit(x)


# In[18]:


identified_clusters = kmeans.fit_predict(x)
identified_clusters


# In[19]:


data_with_clusters = data.copy()
data_with_clusters['Cluster'] = identified_clusters
data_with_clusters


# In[20]:


plt.scatter(data['Longitude'], data['Latitude'], c= data_with_clusters['Cluster'], cmap='rainbow' )
plt.xlim(-180,180)
plt.ylim(-90,90)
plt.show


# In[21]:


data_mapped = data.copy()
data_mapped['Language'] = data_mapped['Language'].map({'English':0, 'French':1, 'German':2})
data_mapped


# In[28]:


x = data_mapped.iloc[:,1:4]
x


# In[29]:


kmeans = KMeans(3)


# In[30]:


kmeans.fit(x)


# In[31]:


identified_clusters = kmeans.fit_predict(x)
identified_clusters


# In[32]:


data_with_clusters = data.copy()
data_with_clusters['Cluster'] = identified_clusters
data_with_clusters


# In[33]:


plt.scatter(data['Longitude'], data['Latitude'], c= data_with_clusters['Cluster'], cmap='rainbow' )
plt.xlim(-180,180)
plt.ylim(-90,90)
plt.show


# ## WCSS-within cluster sum of squares

# In[34]:


kmeans.inertia_


# In[35]:


wcss = []

for i in range(1,7):
    kmeans = KMeans(i)
    kmeans.fit(x)
    wcss_iter = kmeans.inertia_
    wcss.append(wcss_iter)


# In[36]:


wcss


# ## The Elbow Method

# In[38]:


number_clusters = range(1,7)
plt.plot(number_clusters, wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')


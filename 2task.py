
# coding: utf-8

# In[1]:


# Importing libraries
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt  
get_ipython().magic('matplotlib inline')


# In[2]:


data = pd.read_csv("xclara_min.csv")
data.head() 


# In[3]:


df = pd.DataFrame(np.random.randn(100, 2))
msk = np.random.rand(len(df)) < 0.8
train = df[msk]
test = df[~msk]
X = np.array(train)
plt.scatter(X[:,0],X[:,1], label='True Position')  


# In[4]:


kmeans = KMeans(n_clusters=3, random_state=0).fit(train)
centrs = kmeans.cluster_centers_
X = np.array(train)
plt.scatter(X[:,0],X[:,1], c=kmeans.labels_, cmap='rainbow')  
plt.scatter(centrs[:,0],centrs[:,1], c='yellow', cmap='rainbow')  
X1 = np.array(test)
plt.scatter(X1[:,0],X1[:,1], label='True Position', c = 'black')


# In[5]:


def distance(instance1, instance2):
    instance1 = np.array(instance1) 
    instance2 = np.array(instance2) 
    return np.linalg.norm(instance1 - instance2)


# In[6]:


# Euclidean Distance Caculator
def dist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)


# In[10]:


dists = []

def getNearestDot(X, item, clusters):
    min = 100
    itr = 0
    for iterr in range(len(X)):
        trydist = distance(X[iterr], item)
        if trydist < min:
            min = trydist
            itr = iterr
    min2 = 100
    itr2 = 0
    for iterr in range(len(X)):
        trydist = distance(X[iterr], item)
        if iterr != itr:
            if trydist < min2:
                min2 = trydist
                itr2 = iterr
    min3 = 100
    itr3 = 0
    for iterr in range(len(X)):
        trydist = distance(X[iterr], item)
        if iterr != itr and iterr != itr2:
            if trydist < min2:
                min3 = trydist
                itr3 = iterr  
    if clusters[itr2] == clusters[itr3]:
        return [min2, itr2, clusters[itr2]]
    else:
        return [min, itr, clusters[itr]]

def recalcCnts(C, X, labels):
    k = 0
    cen = []
    while k < 3:
        C_old = np.zeros(C.shape)
        error = dist(C, C_old, None)
        while error != 0:
            C_old = C
            points = [X[j] for j in range(len(X)) if labels[j] == k]
            C = np.mean(points, axis=0)
            error = dist(C, C_old, None)
            cen.append(C)
        k += 1
    return cen


# In[11]:


# global trains
# global labs
# global centrs2
# item = [1.323, 2.13314]
# item = np.array([item])
# min, itr, cent = getNearestDot(trains, item, labs)
# cent = np.array([cent])
# trains = np.concatenate((trains, item))
# labs = np.concatenate((labs, cent))
# print(centrs2)
# centr = recalcCnts(centrs2, trains, labs)
# #print()

# #print(centr)
# centrs2 = np.concatenate((centrs2, centr))
# #print()


# In[12]:


trains = X
labs = kmeans.labels_
centrs2 = centrs
for item in X1:
    global trains
    global labs
    global centrs2
    item = np.array([item])
    min, itr, cent = getNearestDot(trains, item, labs)
    cent = np.array([cent])
    trains = np.concatenate((trains, item))
    labs = np.concatenate((labs, cent))
    print(centrs2)
    print()
    print(np.array(recalcCnts(centrs2, trains, labs)))
    print()
    centrs2 = np.array(recalcCnts(centrs2, trains, labs))


# In[13]:


plt.scatter(trains[:,0],trains[:,1], c=labs, cmap='rainbow')  
plt.scatter(centrs2[:,0],centrs2[:,1], c='yellow', cmap='rainbow') # 


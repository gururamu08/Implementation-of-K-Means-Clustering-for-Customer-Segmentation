# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
### Choose the number of clusters (K): 
     Decide how many clusters you want to identify in your data. This is a hyperparameter that you need to set in advance.

 ### Initialize cluster centroids: 
     Randomly select K data points from your dataset as the initial centroids of the clusters.

 ### Assign data points to clusters: 
    Calculate the distance between each data point and each centroid. Assign each data point to the cluster with the closest centroid. This step is typically  done using Euclidean distance, but other distance metrics can also be used.

  ### Update cluster centroids: 
      Recalculate the centroid of each cluster by taking the mean of all the data points assigned to that cluster.

   ### Repeat steps 3 and 4: 
      Iterate steps 3 and 4 until convergence. Convergence occurs when the assignments of data points to clusters no longer change or change very minimally.

  ### Evaluate the clustering results: 
      Once convergence is reached, evaluate the quality of the clustering results. This can be done using various metrics such as the within-cluster sum of squares (WCSS), silhouette coefficient, or domain-specific evaluation criteria.

 ### Select the best clustering solution: 
      If the evaluation metrics allow for it, you can compare the results of multiple clustering runs with different K values and select the one that best suits your requirements

## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: GURUMOORTHI R
RegisterNumber:  212222230042
*/
```
```
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("/content/Dataset-20230524.zip")
data

data.info()

data.isnull().sum()

from sklearn.cluster import KMeans
wcss = []

for i in range(1,11):
    kmeans = KMeans(n_clusters = i,init = "k-means++")
    kmeans.fit(data.iloc[:,3:])
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss)
plt.xlabel("No. of Clusters")
plt.ylabel("wcss")
plt.title("Elbow Method")

km = KMeans(n_clusters = 5)
km.fit(data.iloc[:,3:])

y_pred = km.predict(data.iloc[:,3:])
y_pred

data["cluster"] = y_pred
df0 = data[data["cluster"]==0]
df1 = data[data["cluster"]==1]
df2 = data[data["cluster"]==2]
df3 = data[data["cluster"]==3]
df4 = data[data["cluster"]==4]
plt.scatter(df0["Annual Income (k$)"],df0["Spending Score (1-100)"],c="red",label="cluster0")
plt.scatter(df1["Annual Income (k$)"],df1["Spending Score (1-100)"],c="yellow",label="cluster1")
plt.scatter(df2["Annual Income (k$)"],df2["Spending Score (1-100)"],c="pink",label="cluster2")
plt.scatter(df3["Annual Income (k$)"],df3["Spending Score (1-100)"],c="green",label="cluster3")
plt.scatter(df4["Annual Income (k$)"],df4["Spending Score (1-100)"],c="purple",label="cluster4")
plt.legend()
plt.title("Customer Segments")

```

## Output:
### DATA.HEAD():

![image](https://github.com/gururamu08/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/118707009/379b8b04-444e-4af3-932e-d4899e8eec94)


### DATA.info():

![image](https://github.com/gururamu08/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/118707009/800d4d96-b959-4ca8-b78d-9af9ba982c61)


### NULL VALUES:

![image](https://github.com/gururamu08/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/118707009/9f692bac-2e62-4f69-8e25-fa004145a591)


### ELBOW GRAPH:

![image](https://github.com/gururamu08/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/118707009/c57eab3f-efb6-4d7b-8df8-c969e1ab2bff)


### CLUSTER FORMATION:

![image](https://github.com/gururamu08/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/118707009/86078321-7a54-4d86-8832-f58281db9893)

### PREDICICTED VALUE:

![image](https://github.com/gururamu08/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/118707009/936d36b3-27e8-4cfa-911d-3962fe01f086)


### FINAL GRAPH(D/O):

![image](https://github.com/gururamu08/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/118707009/9d5acede-968b-4d47-8269-887c9f02c6ea)



## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.

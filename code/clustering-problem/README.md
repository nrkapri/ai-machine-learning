# Cluserting 

## K-means Clustering 

#### Pseudocode:

* Step 1: choose number of K clusters

* Step 2: Select at Random k points , the centroids 

* Step 3: Assign each data point to the closest centroid --> that forms K clusters

* Step 4: Compute and place the new centroid of each cluster

* Step 5: Reassign each data point to the new closest centroid .
         if any reassignment took place to to Sttep 4 , otherwise Finish.
         
         

![Image](https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/K-means_convergence.gif/440px-K-means_convergence.gif)

#### Using the elbow method to find the optimal number of clusters

```python
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

#Training
kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)
```


## Hierarchical Clustering

#### Agglomerative: 
This is a "bottom-up" approach: each observation starts in its own cluster, and pairs of clusters are merged as one moves up the hierarchy.
#### Divisive:
This is a "top-down" approach: all observations start in one cluster, and splits are performed recursively as one moves down the hierarchy.

#### Agglomerative Pseudocode:

* Step1:  make each datapoint a single cluster  --> forms N clusters

* Step2: take two closest data points and make them one cluster -->  that forms N-1  clusters

* Step3: take two closest clusters and make them one cluster --> forms N-2 clusters

* Step4: repeat Step 3 untill there is only one cluster 


In above steps the results are kept in memory which creates a dendogram .

Data points:

![Image](https://upload.wikimedia.org/wikipedia/commons/thumb/b/b5/Clusters.svg/250px-Clusters.svg.png)


Dendogram generated:

![Image ](https://upload.wikimedia.org/wikipedia/commons/thumb/a/ad/Hierarchical_clustering_simple_diagram.svg/418px-Hierarchical_clustering_simple_diagram.svg.png)


Using the dendrogram to find the optimal number of clusters

```python
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()
```

Training the Hierarchical Clustering model 

```python
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(X)
```



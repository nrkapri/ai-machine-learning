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


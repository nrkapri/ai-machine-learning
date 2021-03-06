# Classification

## Logistic regression
Logistic regression is a statistical model that in its basic form uses a logistic function to model a binary dependent variable, although many more complex extensions exist. 

![Logistic Function ](https://wikimedia.org/api/rest_v1/media/math/render/svg/9e26947596d387d045be3baeb72c11270a065665)

![Image](https://upload.wikimedia.org/wikipedia/commons/6/6d/Exam_pass_logistic_curve.jpeg)

```python
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)
```

## KNN Classification
A case is classified by a majority vote of its neighbors, with the case being assigned to the class most common amongst its K nearest neighbors measured by a distance function. If K = 1, then the case is simply assigned to the class of its nearest neighbor. 
![Image](https://upload.wikimedia.org/wikipedia/commons/thumb/e/e7/KnnClassification.svg/440px-KnnClassification.svg.png)
![Pseudocode](https://www.researchgate.net/profile/Jung_Keun_Hyun/publication/260397165/figure/fig7/AS:214259620421658@1428094882662/Pseudocode-for-KNN-classification.png)

```python
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)
```

## SVM 
An SVM model is a representation of the examples as points in space, mapped so that the examples of the separate categories are divided by a clear gap that is as wide as possible. New examples are then mapped into that same space and predicted to belong to a category based on the side of the gap on which they fall.

![Image](https://upload.wikimedia.org/wikipedia/commons/thumb/b/b5/Svm_separating_hyperplanes_%28SVG%29.svg/440px-Svm_separating_hyperplanes_%28SVG%29.svg.png)

H1 does not separate the classes. H2 does, but only with a small margin. H3 separates them with the maximal margin.

![Image](https://upload.wikimedia.org/wikipedia/commons/thumb/7/72/SVM_margin.png/600px-SVM_margin.png)

Maximum-margin hyperplane and margins for an SVM trained with samples from two classes. Samples on the margin are called the support vectors.

```python
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)
```

## Kernel SVM
SVM algorithm is good for linearly seperable dataset. But for non-linearly seperable dataset we need Kernel SVM.

Mapping to a higher Dimension: For nonlinearly seperable dataset can be transformed into linearly seperable dataset by use of mapping function. But mapping to higher dimension requires high computing power. 

![Image](https://miro.medium.com/max/1400/1*zWzeMGyCc7KvGD9X8lwlnQ.png)

Kernel Trick :  The “trick” is that kernel methods represent the data only through a set of pairwise similarity comparisons between the original data observations x (with the original coordinates in the lower dimensional space), instead of explicitly applying the transformations ϕ(x) and representing the data by these transformed coordinates in the higher dimensional feature space.

In kernel methods, the data set X is represented by an n x n kernel matrix of pairwise similarity comparisons where the entries (i, j) are defined by the kernel function: k(xi, xj). This kernel function has a special mathematical property. The kernel function acts as a modified dot product. We have:

![Image](https://miro.medium.com/max/1400/1*4hVAPL2cSycg0fYz3MZoYw.png)


```python 
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)
```


## Naive Bayes Classifier 

#### Bayes Theorem:  
describes the probability of an event, based on prior knowledge of conditions that might be related to the event.

![image](https://wikimedia.org/api/rest_v1/media/math/render/svg/87c061fe1c7430a5201eef3fa50f9d00eac78810)

where A and B are events and P(B)!= 0.

P(A|B) is a conditional probability: the likelihood of event A occurring given that B is true.
P(B|A) is also a conditional probability: the likelihood of event B occurring given that A is true.
P(A) and P(B) are the probabilities of observing A and B respectively

Example :
Drug testing:
A particular test for whether someone has been using cannabis is 90% sensitive and 80% specific, meaning it leads to 90% true "positive" results (meaning, "Yes he used cannabis") for cannabis users and 80% true negative results for non-users-- but also generates 20% false positives for non-users. Assuming 5% of people actually do use cannabis, what is the probability that a random person who tests positive is really a cannabis user?
Let {\displaystyle P({\text{User}}\mid {\text{Positive}})}{\displaystyle P({\text{User}}\mid {\text{Positive}})} mean "the probability that someone is a cannabis user given that he tests positive". Then we can write:
![Image](https://wikimedia.org/api/rest_v1/media/math/render/svg/88fc386e383ff18231f9be3c1d17e2d8ca3aa49a)

*[Naive Bayes Classifier:](https://en.wikipedia.org/wiki/Naive_Bayes_classifier#Probabilistic_model)

![Image](https://wikimedia.org/api/rest_v1/media/math/render/svg/d0d9f596ba491384422716b01dbe74472060d0d7)


```python
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)
```


## Decision Tree Classification 

A tree is built by splitting the source set, constituting the root node of the tree, into subsets—which constitute the successor children. The splitting is based on a set of splitting rules based on classification features.[4] This process is repeated on each derived subset in a recursive manner called recursive partitioning. The recursion is completed when the subset at a node has all the same values of the target variable, or when splitting no longer adds value to the predictions.

![Image](https://upload.wikimedia.org/wikipedia/commons/2/25/Cart_tree_kyphosis.png)

```python
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)
```

## Random Forest Classification 
Random forests or random decision forests are an ensemble learning method for classification, regression and other tasks that operate by constructing a multitude of decision trees at training time and outputting the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees. Random decision forests correct for decision trees' habit of overfitting to their training set.

![Image](https://miro.medium.com/max/1148/0*a8KgF1IINziv7KIQ.png)

```python 
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)
```



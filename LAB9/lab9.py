import numpy as np
from collections import defaultdict
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

class KNeighborsClassifier1(object):
    def __init__(self, n_neighbors=1, weights='uniform', p=2):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.p = p

    def fit(self, X, y):
        self.X = X
        self.y = y
        return self

    def _distance(self, data1, data2):
        """1: Manhattan, 2: Euclidean"""
        if self.p == 1:
            return sum(abs(data1 - data2))
        elif self.p == 2:
            return np.sqrt(sum((data1 - data2)**2))
        raise ValueError("p not recognized: should be 1 or 2")

    def _compute_weights(self, distances):
        if self.weights == 'uniform':
            return [(1, y) for d, y in distances]
        elif self.weights == 'distance':
            matches = [(1, y) for d, y in distances if d == 0]
            return matches if matches else [(1/d, y) for d, y in distances]
        raise ValueError("weights not recognized: should be 'uniform' or 'distance'")

    def _predict_one(self, test):
        distances = sorted((self._distance(x, test), y) for x, y in zip(self.X, self.y))
        weights = self._compute_weights(distances[:self.n_neighbors])
        weights_by_class = defaultdict(list)
        for d, c in weights:
            weights_by_class[c].append(d)
        return max((sum(val), key) for key, val in weights_by_class.items())[1]

    def predict(self, X):
        return [self._predict_one(x) for x in X]

    def score(self, X, y):
        return sum(1 for p, t in zip(self.predict(X), y) if p == t) / len(y)



iris = datasets.load_iris()
X_train, X_temp, y_train, y_temp = \
    train_test_split(iris.data, iris.target, test_size=.4)
X_validation, X_test, y_validation, y_test = \
    train_test_split(X_temp, y_temp, test_size=.5)

neighbor = KNeighborsClassifier1().fit(X_train, y_train)

print("błąd średniokwadratowy")
print(neighbor.score(X_train, y_train))
print(neighbor.score(X_validation, y_validation))
print(neighbor.score(X_test, y_test))

#2 knn as classifier
print("classifier")

X, y = datasets.make_classification(
    n_samples=100,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_repeated=0,
    random_state=3
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .4)
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
classifier = KNeighborsClassifier(n_neighbors = 1)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

result = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(result)
result1 = classification_report(y_test, y_pred)
print("Classification Report:",)
print (result1)
result2 = accuracy_score(y_test,y_pred)
print("Accuracy:",result2)

pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)
print(X_reduced)

plt.scatter(X_reduced[:,0], X_reduced[:,1], c=y, edgecolor='k')

plt.show()

iris = datasets.load_iris()
z = iris.data
Y = iris.target
pca = PCA(n_components=3)
z_reduced = pca.fit_transform(z)
print(z_reduced)

plt.scatter(z_reduced[:,0], z_reduced[:,1], c=Y, edgecolor='k')

plt.show()

#knn as regressor
print("regressor")

knnr = KNeighborsRegressor(n_neighbors = 10)
knnr.fit(X, y)

print ("The MSE is:",format(np.power(y-knnr.predict(X),2).mean()))

boston = datasets.load_boston()




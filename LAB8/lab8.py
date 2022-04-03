from sklearn import datasets
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
import scipy.stats
from sklearn.metrics import jaccard_score


def find_perm(clusters, Y_real, Y_pred):
    perm=[]
    for i in range(clusters):
        idx = Y_pred == i
        new_label = scipy.stats.mode(Y_real[idx])[0][0]
        perm.append(new_label)
    return [perm[label] for label in Y_pred]


iris = datasets.load_iris()
X = iris.data
Y = iris.target

model = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
model = model.fit(X)
#print(model.labels_)

s = find_perm(2, Y, model.labels_)
#print(s)
Y_pred = model.labels_
j = jaccard_score(Y, Y_pred, average=None)
#print(j)



#2D
iris = datasets.load_iris()
X = iris.data
Y = iris.target
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)
print(X_reduced)

plt.scatter(X_reduced[:,0], X_reduced[:,1], c=Y, edgecolor='k')

plt.show()

#DIGITS
digits = datasets.load_digits()
X = digits.data
Y = digits.target

model = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
model = model.fit(X)
print(model.labels_)

s = find_perm(2, Y, model.labels_)
print(s)
Y_pred = model.labels_
j = jaccard_score(Y, Y_pred, average=None)
print(j)


#2D
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)
print(X_reduced)

plt.scatter(X_reduced[:,0], X_reduced[:,1], c=Y, edgecolor='k')

plt.show()

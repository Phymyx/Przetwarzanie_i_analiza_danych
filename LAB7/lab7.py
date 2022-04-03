import numpy as np
import scipy.stats
import sklearn
from sklearn.cluster import AgglomerativeClustering
from sklearn import datasets
from sklearn.metrics import jaccard_score
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.cluster.hierarchy import dendrogram,linkage
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.metrics import pairwise_distances_argmin
from sklearn.datasets import load_sample_image
from sklearn.utils import shuffle
from time import time

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
#hull = scipy.spatial.ConvexHull(X_reduced)
plt.scatter(X_reduced[:,0], X_reduced[:,1])
#plt.plot(X_reduced[hull.vertices[0],0], X_reduced[hull.vertices[0],1], 'ro')

#plt.savefig('wykres1.svg')
plt.show()


#3D
fig = plt.figure(1, figsize=(8,6))
ax = Axes3D(fig, elev=-150, azim=110)
pca = PCA(n_components=3)
X_reduced = pca.fit_transform(X)
ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=Y)
ax.set_title("First three PCA directions")
ax.set_xlabel("1st eigenvector")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("2nd eigenvector")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("3rd eigenvector")
ax.w_zaxis.set_ticklabels([])

plt.show()

links = linkage(X_reduced, 'single')
plt.figure(figsize=(8,6))
dendrogram(links, orientation='top', distance_sort='descending', show_leaf_counts=True)
plt.show()

#k-means
#2D
x = pd.DataFrame(iris.data, columns=['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'])
y = pd.DataFrame(iris.target, columns=['Target'])
model = KMeans(n_clusters=3)
model.fit(x)
plt.figure(figsize=(8,6))
colors = np.array(['red', 'green', 'blue'])
predictedY = np.choose(model.labels_, [1, 0, 2]).astype(np.int64)
plt.scatter(x['Sepal Length'], x['Sepal Width'], c=colors[y['Target']], s=40)
plt.title("Model's classification")

plt.show()


estimators = [('k_means_iris_3', KMeans(n_clusters=3))]

#3D
#fignum = 1
#titles = ['3 clusters']
for name, est in estimators:
    fig = plt.figure(1, figsize=(8, 6))
    ax = Axes3D(fig, elev=-150, azim=110)
    est.fit(X)
    labels = est.labels_

    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels.astype(np.float), edgecolor='k')

    ax.set_title("First three KMeans directions")
    ax.set_xlabel("1st eigenvector")
    ax.w_xaxis.set_ticklabels([])
    ax.set_ylabel("2nd eigenvector")
    ax.w_yaxis.set_ticklabels([])
    ax.set_zlabel("3rd eigenvector")
    ax.w_zaxis.set_ticklabels([])

    plt.show()


n_colors = 64

flower = load_sample_image('flower.jpg')
flower = np.array(flower, dtype=np.float64) / 255
w, h, d = original_shape = tuple(flower.shape)
assert d == 3
image_array = np.reshape(flower, (w * h, d))

print("small sample")
t0 = time()
image_array_sample = shuffle(image_array, random_state=0)[:1000]
kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_sample)
print("done in %0.3fs." % (time() - t0))

print("k-means")
t0 = time()
labels = kmeans.predict(image_array)
print("done in %0.3fs." % (time() - t0))

codebook_random = shuffle(image_array, random_state=0)[:n_colors]
print("random")
t0 = time()
labels_random = pairwise_distances_argmin(codebook_random,
                                          image_array,
                                          axis=0)
print("done in %0.3fs." % (time() - t0))


def recreate_image(codebook, labels, w, h):
    d = codebook.shape[1]
    image = np.zeros((w, h, d))
    label_idx = 0
    for i in range(w):
        for j in range(h):
            image[i][j] = codebook[labels[label_idx]]
            label_idx += 1
    return image

# all results
plt.figure(1)
plt.clf()
plt.axis('off')
plt.title('Original image')
plt.imshow(flower)

plt.figure(2)
plt.clf()
plt.axis('off')
plt.title('Quantized image (K-Means)')
plt.imshow(recreate_image(kmeans.cluster_centers_, labels, w, h))

plt.figure(3)
plt.clf()
plt.axis('off')
plt.title('Quantized image (Random)')
plt.imshow(recreate_image(codebook_random, labels_random, w, h))
plt.show()
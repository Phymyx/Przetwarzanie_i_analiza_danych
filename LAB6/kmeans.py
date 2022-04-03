from sklearn import metrics
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
import math
import numpy as np


def distp(X,C):
    A=np.transpose(X-C)
    d=math.sqrt((X-C)*A)
    return d

def distm(X,C):
    A = np.transpose(X - C)
    V=np.cov(X)
    B=math.pow(V,(-1))
    d = math.sqrt((X - C) * B * A)
    return d

def odl(X, X1, Y, Y1):
    x = abs(X - X1)
    print(x)
    y = abs(Y - Y1)
    print(y)
    return x, y

def od(X, Y):
    c = math.sqrt(pow(X, 2) + pow(Y, 2))
    return c

df = pd.read_csv("autos.csv")
df=df.dropna(subset=['engine-size', 'horsepower'])
print(df)
print(df.shape)


plt.scatter(df['engine-size'],df['horsepower'])
plt.show()

km=KMeans(n_clusters=2)
print(km)


y_predicted = km.fit_predict(df[['engine-size','horsepower']])
print(y_predicted)


df['cluster'] = y_predicted

df1 = df[df.cluster==0]
df2 = df[df.cluster==1]
df3 = df[df.cluster==2]

plt.scatter(df1['engine-size'],df1['horsepower'],color='green')
plt.scatter(df2['engine-size'],df2['horsepower'],color='red')
plt.scatter(df3['engine-size'],df3['horsepower'],color='black')


plt.xlabel('engine-size')
plt.ylabel('horsepower')

print(km.cluster_centers_)
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='purple',marker='*',label='centroid')

plt.show()

k_rng = range(1,10)
sse = []
for k in k_rng:
    km=KMeans(n_clusters=k)
    km.fit(df[['engine-size','horsepower']])
    sse.append(km.inertia_)

print(sse)

plt.xlabel('K')
plt.ylabel('Sum of squared error')
plt.plot(k_rng,sse)
plt.show()

#distp(109.67096774, 86.29032258)
odl(182.35416667, 109.67096774, 162.27083333, 86.29032258)  #boki trójkąta prostokątnego
print(od(72.68319893000002, 75.98051075))       #odległość (przekątna) pomiędzy środkami klastrów



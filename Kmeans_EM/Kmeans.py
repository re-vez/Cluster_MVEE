import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def trunc(values, decs=0):
    return np.trunc(values*10**decs)/(10**decs)

np.random.seed(13)

mu1 = np.array([1, 2])
sigma1 = np.array([[13, -12], [-12, 13]])/12.5

mu2 = np.array([1, 2])
sigma2 = np.array([[13, 12], [12, 13]])/12.5


X1 = np.random.multivariate_normal(mu1, sigma1, 15)
X2 = np.random.multivariate_normal(mu2, sigma2, 15)

X = np.vstack((X1, X2))

print(X)

clustering = KMeans(n_clusters = 2)
clustering.fit(X)
print(clustering.labels_)

fig = plt.figure(figsize = (9, 9))
ax = fig.add_subplot(111)
ax.scatter(X1[:,0], X1[:,1], color = "red")
ax.scatter(X2[:,0], X2[:,1], color = "blue")
plt.title("Clusters")


fig = plt.figure(figsize = (9, 9))
ax = fig.add_subplot(111)
for i in range(len(X)):
    if clustering.labels_[i] == 0:
        ax.scatter(X[i,0],X[i,1], color = "red")
    else:
        ax.scatter(X[i,0],X[i,1], color = "blue")
plt.title("Predicted clusters")
plt.show()

X1 = trunc(X1,3)
X2 = trunc(X2,3)

with open("Example.txt","w") as f:
    for i in range(15):
        f.write("filldraw [color1] ("+str(X1[i,0])+","+str(X1[i,1])+") circle (1.5pt);\n")
    for i in range(15):
        f.write("filldraw [color3] ("+str(X2[i,0])+","+str(X2[i,1])+") circle (1.5pt);\n")

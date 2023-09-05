import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

np.random.seed(13)

mu1 = np.array([1, 3])
sigma1 = np.array([[13, -12], [-12, 13]])

mu2 = np.array([1, 3])
sigma2 = np.array([[13, 12], [12, 13]])

X1 = np.random.multivariate_normal(mu1, sigma1, 15)
X2 = np.random.multivariate_normal(mu2, sigma2, 15)

X = np.vstack((X1, X2))

EM = GaussianMixture(n_components = 2)
EM.fit(X)
cluster = EM.predict(X)

print(cluster)

fig = plt.figure(figsize = (9, 9))
ax = fig.add_subplot(111)
ax.scatter(X1[:,0], X1[:,1], color = "red")
ax.scatter(X2[:,0], X2[:,1], color = "blue")
plt.title("Clusters")

fig = plt.figure(figsize = (9, 9))
ax = fig.add_subplot(111)
for i in range(len(X)):
    if cluster[i] == 0:
        ax.scatter(X[i,0],X[i,1], color = "red")
    else:
        ax.scatter(X[i,0],X[i,1], color = "blue")
plt.title("Predicted clusters")
plt.show()

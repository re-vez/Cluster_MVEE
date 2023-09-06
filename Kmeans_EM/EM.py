import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

np.random.seed(13)



X1 = np.array([[1,1],[2,1],[2,2]])
X2 = np.array([[-1.,-1],[-2,-1],[-2,-2]])

X = np.vstack((X1, X2))

EM = GaussianMixture(n_components = 2)
EM.fit(X)
cluster = EM.predict(X)


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

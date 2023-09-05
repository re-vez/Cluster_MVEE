import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp

new_X1 = np.array([
    [2, -8, 3, -2, 8.2, 4.6,-3],
    [-2, 6, 4, 2, -2.4, -7.2,-4],
    [-2,-2,-2,-2,-2,-2,-2]
])+3

X1 = np.array([
    [2, -8, 3, -2, 8.2, 4.6,-3],
    [-2, 6, 4, 2, -2.4, -7.2,-4]
])+3

print(X1)

d1, n1 = X1.shape

# Solver use
H = cp.Variable((d1+1,d1+1), symmetric = True)

constraints = [
    new_X1[:,i].T @ H @ new_X1[:,i] <= d1+1 for i in range(n1)
]
constraints += [H >> 0]

prob = cp.Problem(cp.Minimize(-cp.log_det(H)), constraints)
prob.solve(solver=cp.MOSEK)
sol_H = H.value
print(sol_H)
center = np.linalg.solve(-sol_H[0:d1,0:d1], sol_H[0:d1,d1])

# Plots

increment = 0.025
xrange = np.arange(-10.0, 10.0, increment)
yrange = np.arange(-10.0, 10.0, increment)
X, Y = np.meshgrid(xrange,yrange)

F = sol_H[0,0]*(X - center[0])**2 + sol_H[1,1]*(Y - center[1])**2
F += 2*sol_H[1,0]*(X - center[0])*(Y - center[1]) - d1
print(sol_H[0:2,0:2])
print(center)
fig = plt.figure(figsize = (10, 7))
ax = fig.add_subplot(111)
ax.scatter(X1[0], X1[1], color = "red")
ax.contour(X, Y, F, [0])

plt.title("MVEE")

plt.show()

from copy import deepcopy as dpcp
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from random import sample
import time

def frac_norm(number: float):
    """
    This function imputs a number between 0 and 1 and outputs
    how far the input is from 0.5.
    """
    return abs(number-0.5)
    
def upper_bound_heuristic(data, runs, M,
                          data_size = None,
                          dim = None):
    """
    A simple heuristic for the minimization problem.
    """
    if data_size is None or dim is None:
        dim, data_size = data.shape
    upper_bound = np.inf
    iterations = 0
    while iterations < runs:
        iterations += 1
        num = np.random.randint(dim+1,data_size - dim)
        indexes = sample(range(data_size), num)
        y = [1 if i in indexes else 0 for
             i in range(data_size)]
        H1 = cp.Variable((dim,dim), symmetric = True)
        H2 = cp.Variable((dim,dim), symmetric = True)
        cons = [H1 >> 0, H2 >> 0]
        cons += [
            (data[:,i].T @ H1 @ data[:,i]) <= dim + M*y[i]
            for i in range(data_size)
            ]
        
        cons += [
            (data[:,i].T @ H2 @ data[:,i]) <= dim + M*(1-y[i])
            for i in range(data_size)
            ]
        prob1 = cp.Problem(cp.Minimize(-cp.log_det(H1+H2)),
                           cons)
        prob1.solve(solver=cp.MOSEK, warm_start=True)
        upper_cand = prob1.value
        if upper_cand < upper_bound:
            upper_bound = upper_cand
    return upper_bound

def integrality(array, tol, size):
    """
    This function imputs an array, a tolerance (0<float<1) 
    and the array's size. It outputs the index for 
    the BnB algorithm
    """
    condition = True
    ind = 0
    branching = -1
    while ind < size and condition:
        frac = array[ind]
        if tol < frac < 1-tol:
            condition = False
            branching = ind
            fr_norm = frac_norm(frac)
        ind += 1
    while ind < size:
        frac_comp_n = frac_norm(array[ind])
        if frac_comp_n < fr_norm:
            branching = ind
            fr_norm = frac_comp_n
        ind += 1
    return branching


def branch_and_bound_domed(data, runs, M, tol):
    """
    e
    """
    dim, data_size = data.shape
    id_count = 1
    upper_bound = np.inf #upper_bound_heuristic(data,
                 #runs, M, data_size = None, dim = None)
    H1 = cp.Variable((dim,dim), symmetric = True)
    H2 = cp.Variable((dim,dim), symmetric = True)
    y = cp.Variable(data_size, nonneg = True)
    cons = [H1 >> 0, H2 >> 0]
    cons += [y[i] <= 1 for i in range(data_size)]
    cons += [
            (data[:,i].T @ H1 @ data[:,i]) <= dim
            + M*y[i] for i in range(data_size)
            ]
    cons += [
            (data[:,i].T @ H2 @ data[:,i]) <= dim
            + M*(1-y[i]) for i in range(data_size)
            ]
    prob = cp.Problem(
        cp.Minimize(-cp.log_det(H1+H2)),
                           cons)
    prob.solve(solver=cp.MOSEK)
    int_index = integrality(y.value, tol, data_size)
    sol_val = prob.value
    Tree = {id_count:[sol_val, dpcp(H1.value),
            dpcp(H2.value), dpcp(y.value), 0, 0,
            int_index, cons.copy()]}
    Scout = np.array([[sol_val],[id_count]])
    while np.shape(Scout) != (2,0):
        current_index = Scout[1,0]
        if Tree[current_index][4] > data_size - dim \
            or Tree[current_index][5] > data_size - dim:
            Tree.pop(current_index)
            Scout = np.delete(Scout,0,1)
        elif Tree[current_index][0] <= upper_bound\
            and Tree[current_index][6] == -1:
            upper_bound = Tree[current_index][0]
            sol_H1 = Tree[current_index][1]
            sol_H2 = Tree[current_index][2]
            sol_y = Tree[current_index][3]
            Tree.pop(current_index)
            Scout = np.delete(Scout,0,1)
        elif Tree[current_index][0] > upper_bound:
            Tree.pop(current_index)
            Scout = np.delete(Scout,0,1)
        else:
        #####################################################
        # y = 0
            cons = Tree[current_index][7].copy()
            cons += [y[Tree[current_index][6]] == 0]
            prob = cp.Problem(
            cp.Minimize(-cp.log_det(H1+H2)),
                           cons)
            H1.value = Tree[current_index][1]
            H2.value = Tree[current_index][2]
            prob.solve(solver=cp.MOSEK, warm_start=True)
            int_index = integrality(y.value, tol, data_size)
            sol_val_0 = prob.value
            Tree.update({(id_count+1):[sol_val_0,
            dpcp(H1.value),dpcp(H2.value), dpcp(y.value),
            Tree[current_index][4]+1, Tree[current_index][5],
            int_index, cons.copy()]})
        #####################################################
        # y = 1
            cons = Tree[current_index][7].copy()
            cons += [y[Tree[current_index][6]] == 1]
            prob = cp.Problem(
            cp.Minimize(-cp.log_det(H1+H2)),
                           cons)
            H1.value = Tree[current_index][1]
            H2.value = Tree[current_index][2]
            prob.solve(solver=cp.MOSEK, warm_start=True)
            int_index = integrality(y.value, tol, data_size)
            sol_val_1 = prob.value
            Tree.update({(id_count+2):[sol_val_1, 
            dpcp(H1.value),dpcp(H2.value), dpcp(y.value),
            Tree[current_index][4], Tree[current_index][5]+1,
            int_index, cons.copy()]})
            #################################################
            Scout = np.delete(Scout,0,1)
            New_Nodes = np.array([[sol_val_0,sol_val_1],
                        [id_count+1,id_count+2]])
            Scout = np.hstack((Scout,New_Nodes.copy()))
            Scout = Scout[:, Scout[0].argsort()]
            Tree.pop(current_index)
            id_count += 2
    return sol_H1, sol_H2, sol_y, upper_bound




if __name__ == '__main__':
    np.random.seed(13)
    points = range(10,13)
    X1 = np.array([[x,
                    y,
                    1] for x in points for y in points])
    X2 = np.array([[-x,
                    -y,
                    1] for x in points for y in points])
    n1, d1 = X1.shape
    n2, d1 = X2.shape
    Obs = sum([n1,n2])
    new_X = np.hstack((X1.T, X2.T))
    start = time.time()
    H_1, H_2, solut_y, up = branch_and_bound_domed(new_X,
                                      10000, 2000, 1e-6)
    print(time.time()-start)
    print(H_1)
    print(H_2)
    print(solut_y)
    center1 = np.linalg.solve(-H_1[0:2,0:2], H_1[0:2,2])
    center2 = np.linalg.solve(-H_2[0:2,0:2], H_2[0:2,2])

    increment = 0.025
    xrange = np.arange(-20.0, 20.0, increment)
    yrange = np.arange(-20.0, 20.0, increment)
    X_1, Y_1 = np.meshgrid(xrange,yrange)
    X_2, Y_2 = np.meshgrid(xrange,yrange)

    F1 = H_1[0,0]*(X_1 - center1[0])**2
    F1 += H_1[1,1]*(Y_1 - center1[1])**2
    F1 += 2*H_1[1,0]*(X_1 - center1[0])*(Y_1 - center1[1]) - 2

    F2 = H_2[0,0]*(X_2 - center2[0])**2
    F2 += H_2[1,1]*(Y_2 - center2[1])**2
    F2 += 2*H_2[1,0]*(X_2 - center2[0])*(Y_2 - center2[1]) - 2
    print(up)
    fig = plt.figure(figsize = (10, 10))
    ax = fig.add_subplot(111)
    ax.scatter(X1[:,0], X1[:,1], color = "red")
    ax.scatter(X2[:,0], X2[:,1], color = "blue")
    ax.contour(X_1, Y_1, F1, [0])
    ax.contour(X_2, Y_2, F2, [0])
    plt.title("Clusters")
    plt.show()
    
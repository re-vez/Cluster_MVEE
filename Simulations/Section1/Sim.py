import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from Branch_and_bound_DOMED import branch_and_bound
import time
import ndtest

def MLE_mu(y,matrix_X):
    mu = sum([matrix_X[0:2,i]*y[i] for i in range(len(y))])/sum(y)
    return mu

def MLE_sigma(y,matrix_X,mu):
    mu.reshape((2, 1))
    sigma = sum([y[i]*(matrix_X[0:2,i].reshape((2, 1))-mu)@(matrix_X[0:2,i].reshape((2, 1))-mu).T for i in range(len(y))])
    sigma = sigma/(sum(y)-1)
    return sigma

def Simulation_Mixture(sample_size, mu1,mu2, sigma1,sigma2, pi):
    var_1 = np.random.binomial(size=1, n=sample_size, p=pi[0])
    X1 = np.random.multivariate_normal(mu1, sigma1, var_1).T
    X2 = np.random.multivariate_normal(mu2, sigma2, sample_size-var_1).T
    return(X1,X2)

np.random.seed(13)
mu1 = np.array([1,3])
sigma1 = np.array([[101, -99], [-99, 101]])/2

mu2 = np.array([1,3])
sigma2 = np.array([[101, 99], [99, 101]])/2


X1,X2 = Simulation_Mixture(36, mu1, mu2, sigma1, sigma2, [0.5,0.5])
#20 worked
#22 worked
#24 worked
#26 worked
y1,y2 = np.ones(len(X1[0])), np.zeros(len(X2[0]))
new_X1 = np.vstack((X1, np.ones((1,len(X1[0])))))
new_X2 = np.vstack((X2, np.ones((1,len(X2[0])))))
new_X = np.hstack((new_X1,new_X2)).T
X = np.hstack((X1,X2)).T
y = np.hstack((y1,y2)).T

x_train,x_test,y_train,y_test=train_test_split(new_X,y,test_size=0.3)
x_train,x_test,y_train,y_test = x_train.T,x_test.T,y_train.T,y_test.T

new_X = new_X.T
X = X.T
y = y.T

start_time = time.time()
H_1, H_2, solut_y, up = branch_and_bound(x_train, 100, 2000, 1e-6, 8)
total_time = time.time() - start_time

mle_mu1 = MLE_mu(solut_y,x_train)
mle_mu2 = MLE_mu(1-solut_y,x_train)

mle_sigma1 = MLE_sigma(solut_y,x_train,mle_mu1)
mle_sigma2 = MLE_sigma(1-solut_y,x_train,mle_mu2)

pi_1 = sum(solut_y)/len(solut_y)
MLEX1,MLEX2 = Simulation_Mixture(36, mle_mu1, mle_mu2, mle_sigma1, mle_sigma2, [pi_1])
MLEX = np.hstack((MLEX1,MLEX2))

p_val = 0
for i in range(1000):
    MLEX1,MLEX2 = Simulation_Mixture(36, mle_mu1, mle_mu2, mle_sigma1, mle_sigma2, [pi_1])
    MLEX = np.hstack((MLEX1,MLEX2))
    P, D = ndtest.ks2d2s(x_test[0], x_test[1], MLEX[0], MLEX[1], extra=True)
    p_val += P

print(p_val/1000)

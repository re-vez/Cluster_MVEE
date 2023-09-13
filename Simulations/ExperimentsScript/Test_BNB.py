import numpy as np
from Branch_and_bound_MVEE import branch_and_bound
import time


def trunc(values, decs=0):
    return np.trunc(values*10**decs)/(10**decs)

tag = [0,1,10,11,12,13,14,15]
size = range(5,25)

for i in tag:
    for N in size:
        the_seed = (i+1)**2 + 13*N
        np.random.seed(the_seed)
        print(i)
        mu1 = 10*np.random.random(2)
        sigma1 = np.random.rand(2,2)
        sigma1 = sigma1@np.array([[1,0],[0,10]])@sigma1.T

        mu2 = 10*np.random.random(2)
        sigma2 = sigma1@np.array([[1,0],[0,10]])@sigma1.T

        X1 = np.random.multivariate_normal(mu1, sigma1, N).T
        X2 = np.random.multivariate_normal(mu2, sigma2, N).T
    
        m1, n1 = X1.shape
        m2, n2 = X2.shape

        new_X1 = np.vstack((X1, np.ones((1,n1))))
        new_X2 = np.vstack((X2, np.ones((1,n2))))
        new_X = np.hstack((new_X1,new_X2))

        start_time = time.time()
        H_1, H_2, solut_y, up = branch_and_bound(new_X, 100, 2000, 1e-6)
        total_time = (time.time() - start_time)/60
        final_H1 = H_1[0:2,0:2]
        final_H2 = H_2[0:2,0:2]
        centre_1 = np.linalg.solve(-H_1[0:2,0:2], H_1[0:2,2])
        centre_2 = np.linalg.solve(-H_2[0:2,0:2], H_2[0:2,2])
        tail_H1 = H_1[2,2]
        tail_H2 = H_2[2,2]

    
        print(total_time)
        string = 'experiment_size'+str(N)+'_tag'+str(i)
        with open(string+'.txt', 'w') as f:
            f.write('Criteria: -lndet(H1)-lndet(H2)\n')
            f.write('seed = '+str(the_seed)+'\n')
            f.write('Time[min] = '+str(total_time)+'\n')
            f.write('mu1 = \n'+str(mu1)+'\n')
            f.write('mu2 = \n'+str(mu2)+'\n')
            f.write('sigma1 = \n'+str(sigma1)+'\n')
            f.write('sigma2 = \n'+str(sigma2)+'\n')
            f.write('X1 = \n'+str(X1)+'\n')
            f.write('X2 = \n'+str(X2)+'\n')
            f.write('H1 = \n'+str(final_H1)+'\n')
            f.write('H2 = \n'+str(final_H2)+'\n')
            f.write('tail_H1 = '+str(tail_H1)+'\n')
            f.write('tail_H2 = '+str(tail_H2)+'\n')
            f.write('centre_1 = \n'+str(centre_1)+'\n')
            f.write('centre_2 = \n'+str(centre_2)+'\n')
            f.write('y = \n'+str(solut_y))

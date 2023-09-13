import numpy as np
import time
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans


def trunc(values, decs=0):
    return np.trunc(values*10**decs)/(10**decs)

tag = [0,1,11,12,13,14,15]
size = range(5,24)

for i in tag:
    for N in size:
        the_seed = (i+1)**2 + 13*N
        np.random.seed(the_seed)
        print(i,' ',N)
        mu1 = 10*np.random.random(2)
        sigma1 = np.random.rand(2,2)
        sigma1 = sigma1@np.array([[1,0],[0,10]])@sigma1.T

        mu2 = 10*np.random.random(2)
        sigma2 = np.random.rand(2,2)
        sigma2 = sigma2@np.array([[1,0],[0,10]])@sigma2.T

        X1 = np.random.multivariate_normal(mu1, sigma1, N)
        X2 = np.random.multivariate_normal(mu2, sigma2, N)

        X = np.vstack((X1, X2))


        start_time = time.time()
        EM = GaussianMixture(n_components = 2)
        EM.fit(X)
        solut_y_EM = EM.predict(X)
        total_time = (time.time() - start_time)/60

    
        print(total_time)
        string = 'EM-experiment_size'+str(N)+'_tag'+str(i)
        with open(string+'.txt', 'w') as f:
            f.write('Criteria: Estimation-Maximisation\n')
            f.write('seed = '+str(the_seed)+'\n')
            f.write('Time[min] = '+str(total_time)+'\n')
            f.write('mu1 = \n'+str(mu1)+'\n')
            f.write('mu2 = \n'+str(mu2)+'\n')
            f.write('sigma1 = \n'+str(sigma1)+'\n')
            f.write('sigma2 = \n'+str(sigma2)+'\n')
            f.write('X1 = \n'+str(X1)+'\n')
            f.write('X2 = \n'+str(X2)+'\n')
            f.write('y = \n'+str(solut_y_EM))

for i in tag:
    for N in size:
        the_seed = (i+1)**2 + 13*N
        np.random.seed(the_seed)
        print(i,' ',N)
        mu1 = 10*np.random.random(2)
        sigma1 = np.random.rand(2,2)
        sigma1 = sigma1@np.array([[1,0],[0,10]])@sigma1.T

        mu2 = 10*np.random.random(2)
        sigma2 = np.random.rand(2,2)
        sigma2 = sigma1@np.array([[1,0],[0,10]])@sigma1.T

        X1 = np.random.multivariate_normal(mu1, sigma1, N)
        X2 = np.random.multivariate_normal(mu2, sigma2, N)

        m1, n1 = X1.shape
        m2, n2 = X2.shape

        X = np.vstack((X1, X2))

        start_time = time.time()
        clustering = KMeans(n_clusters = 2)
        clustering.fit(X)
        solut_y_Km = clustering.labels_
        total_time = (time.time() - start_time)/60


        print(total_time)
        string = 'Km-experiment_size'+str(N)+'_tag'+str(i)
        with open(string+'.txt', 'w') as f:
            f.write('Criteria: K-means\n')
            f.write('seed = '+str(the_seed)+'\n')
            f.write('Time[min] = '+str(total_time)+'\n')
            f.write('mu1 = \n'+str(mu1)+'\n')
            f.write('mu2 = \n'+str(mu2)+'\n')
            f.write('sigma1 = \n'+str(sigma1)+'\n')
            f.write('sigma2 = \n'+str(sigma2)+'\n')
            f.write('X1 = \n'+str(X1)+'\n')
            f.write('X2 = \n'+str(X2)+'\n')
            f.write('y = \n'+str(solut_y_Km))

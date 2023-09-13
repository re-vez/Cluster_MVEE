import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

df = pd.read_csv('Cluster_MVEE/Simulations/TimePlots/timeBnBMVEE.csv')

df['2^n'] = 2**df['n']
df['n2^n'] = 2**df['n']*df['2^n']
df['loglog'] = np.log(np.log(df['n']))*df['2^n']

X = df[['2^n', 'n2^n']]#,'loglog']]
y = df['t']

X = sm.add_constant(X)
est = sm.OLS(y, X).fit()
report = str(est.summary())
print(report)

n = np.linspace(10, 50, 50)
N = np.column_stack((n**0,2**n,n*2**n))
t_n = np.dot(N, est.params[0:3])

fig = plt.figure(figsize = (10, 10))
ax = fig.add_subplot(111)
ax.plot(n, t_n, "b-", label="estimation")
ax.scatter(df['n'], df['t'], color = "red")
plt.title("n vs t(n)")
plt.show()

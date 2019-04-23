import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D #testes para ver se tá gerando no cubo
import generateDataSet_4 as gn
import pandas as pd 
import numpy as np 

dataframe = gn.generate_array(1000, 0.120)

x = dataframe[:,:3]

weights = pd.read_csv('weights.csv', header=None, index_col=None).values

def func(x):
    return np.dot(x, weights[0,:3]) + weights[0,3]

x_axis = np.linspace(-0.5,1.5, 100)
xx, yy = np.meshgrid(x_axis, x_axis)

zz1 = (-weights[0,0]*xx - weights[0,1]*yy - weights[0,3]) / weights[0,2] 
zz2 = (-weights[1,0]*xx - weights[1,1]*yy - weights[1,3]) / weights[1,2] 
zz3 = (-weights[2,0]*xx - weights[2,1]*yy - weights[2,3]) / weights[2,2] 

alp = .5
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#ax.scatter(dataframe[:,0], dataframe[:,1], dataframe[:,2], c=dataframe[:,3], marker='o', cmap='Dark2')
ax.plot_surface(xx, yy, zz1, color='blue', zorder=-1 , alpha=alp)
ax.plot_surface(xx, yy, zz2, color='yellow', zorder=0, alpha=alp)
ax.plot_surface(xx, yy, zz3, color='cyan', zorder=1, alpha=alp)
plt.grid()
#plt.xlim([-1.5, 1.5])
#plt.ylim([-1.5, 1.5])

plt.show()


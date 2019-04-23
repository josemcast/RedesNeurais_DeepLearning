import numpy as np 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


dt_train = np.loadtxt('result_7b.csv', delimiter=',')

dimen = int(np.sqrt(len(dt_train)))
fig = plt.figure()

ax = fig.gca(projection='3d')
ax.plot_surface(dt_train[:,0].reshape(dimen, -1), dt_train[:,1].reshape(dimen, -1), dt_train[:,2].reshape(dimen, -1), alpha=0.5, label='Original')
#ax.plot_surface(dt_train[:,0].reshape(dimen, -1), dt_train[:,1].reshape(dimen, -1), dt_train[:,3].reshape(dimen, -1), alpha=0.5, label='Aproximada', color='r')

fig2 = plt.figure()
ax2 = fig2.gca(projection='3d')
ax2.plot_surface(dt_train[:,0].reshape(dimen, -1), dt_train[:,1].reshape(dimen, -1), dt_train[:,3].reshape(dimen, -1), alpha=0.5, label='Aproximada', color='r')

plt.show()


import numpy as np 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


result = np.loadtxt('pred_x_train.txt', delimiter=',')
size= int(np.sqrt(len(result[:,0])))
print(size)
xx, yy = result[:,0].reshape(size,-1), result[:,1].reshape(size,-1) 
zz = result[:,2].reshape(size,-1)

f_x = ((np.cos(2*np.pi*xx)/(1-(4*xx)**2))*(np.sin(np.pi*xx)/(np.pi*xx)))*((np.cos(2*np.pi*yy)/(1-(4*yy)**2))*(np.sin(np.pi*yy)/(np.pi*yy)))

fig = plt.figure()
ax = fig.gca(projection='3d')
#ax.plot_surface(xx, yy, f_x, alpha=0.5)
ax.plot_surface(xx, yy, zz, color='r', alpha=1)

plt.show()
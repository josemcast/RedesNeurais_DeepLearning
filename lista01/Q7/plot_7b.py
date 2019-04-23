
import numpy as np 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

x = np.linspace(-4*np.pi, 4*np.pi, 50)
xx, yy = np.meshgrid(x, x)

f_x = ((np.cos(2*np.pi*xx)/(1-(4*xx)**2))*(np.sin(np.pi*xx)/(np.pi*xx)))*((np.cos(2*np.pi*yy)/(1-(4*yy)**2))*(np.sin(np.pi*yy)/(np.pi*yy)))

print(len(f_x.shape))
fig = plt.figure()
ax = fig.gca(projection='3d')

ax.plot_surface(xx, yy, f_x)

plt.show()
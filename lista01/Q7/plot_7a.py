
import numpy as np 
import generateDataSet_7a as gn 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

dt = gn.generate_array(1000)


fig = plt.figure()
ax = fig.gca(projection='3d')

ax.scatter(dt[:, 0], dt[:, 1], dt[:, 2], c=dt[:,3], cmap='bwr')
plt.show()
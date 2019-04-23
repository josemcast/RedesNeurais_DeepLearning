import numpy as np 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

np.random.seed(42)


def f_x(xx, yy):
    return ((np.cos(2*np.pi*xx)/(1-(4*xx)**2))*(np.sin(np.pi*xx)/(np.pi*xx)))*((np.cos(2*np.pi*yy)/(1-(4*yy)**2))*(np.sin(np.pi*yy)/(np.pi*yy)))

def generate_array(size):
    
    n = np.linspace(-np.pi, np.pi, size)
    x, y = np.meshgrid(n, n)
    
    z = f_x(x, y)

    X = np.ravel(x)
    Y = np.ravel(y)
    Z = np.ravel(z)
    
    dt = np.c_[X, Y]
    dt = np.c_[dt, Z]
    #np.random.shuffle(dt)
    return dt

# size=100
# dataframe = generate_array(size)
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# print(len(dataframe))

# ax.plot_surface(dataframe[:,0].reshape(size,-1), dataframe[:,1].reshape(size,-1), dataframe[:,2].reshape(size,-1))

# plt.show()

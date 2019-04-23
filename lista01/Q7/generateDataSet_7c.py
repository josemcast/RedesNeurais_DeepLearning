import numpy as np 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

np.random.seed(73)

def f_x(xx1,xx2):

    # Fazendo por muliplicacao matricial (mais demorado)

    # m = np.array([[0, 0.5, -0.5],[0, 0.5, -0.5]])

    # a = np.exp(-0.5*np.matmul(xx-m[:,0],xx-m[:,0]))
    # b = np.exp(-0.5*np.matmul(xx-m[:,1],xx-m[:,1]))
    # c = np.exp(-0.5*np.matmul(xx-m[:,2],xx-m[:,2]))

    # return ((1/(2*np.pi))*np.diagonal(np.exp(-0.5*np.dot(xx-m[:,0], (xx-m[:,0]).T ))+
    #                                   np.exp(-0.5*np.dot(xx-m[:,1], (xx-m[:,1]).T ))+
    #                                   np.exp(-0.5*np.dot(xx-m[:,2], (xx-m[:,2]).T ))))


    m = np.array([0, 0.5, -0,5])

    return (1/(2*np.pi))*(np.exp(-0.5*((xx1 - m[0])**2 + (xx2 - m[0])**2)) + 
                          np.exp(-0.5*((xx1 - m[1])**2 + (xx2 - m[1])**2)) + 
                          np.exp(-0.5*((xx1 - m[2])**2 + (xx2 - m[2])**2)))


            
def generate_array(size=100):

    n = np.linspace(-10,10, size)
    x1, x2 = np.meshgrid(n,n)

    X1 = np.ravel(x1)
    X2 = np.ravel(x2)
    Z = f_x(X1,X2)

    dt = np.c_[X1,X2]
    dt = np.c_[dt, Z]
    np.random.shuffle(dt)
    return dt


# size=100
# dataframe = generate_array(size)

# fig = plt.figure()
# ax = fig.gca(projection='3d')
# print(len(dataframe))

# ax.contourf(dataframe[:,0].reshape(size,-1), dataframe[:,1].reshape(size,-1), dataframe[:,2].reshape(size,-1))
# surf = ax.plot_trisurf(dataframe[:,0], dataframe[:,1], dataframe[:,2],cmap=cm.coolwarm)
# ax.set_xlim3d(-1,1)
# ax.set_xlim3d(-1,1)
# surf = ax.plot_surface(dataframe[:,0].reshape(size,-1), dataframe[:,1].reshape(size,-1), dataframe[:,2].reshape(size,-1),
# cmap=cm.coolwarm)
# fig.colorbar(surf, shrink=0.5, aspect=5)

# ax.scatter(dataframe[:,0].reshape(size,-1), dataframe[:,1].reshape(size,-1), dataframe[:,2].reshape(size,-1),)

# plt.show()
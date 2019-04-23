import numpy as np 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

np.random.seed(42)

def x_eval(x):
    return 1 + np.cos(x + np.cos(x)**2)

def generate_array(size=1000, steps=3, win=0):
    #one to three-steps prediction
    n = np.linspace(0, 50, size)    
    x_n = x_eval(n)

    for i in range(win):
        x_n = np.c_[x_n, x_eval(n-(i+1))]

    for i in range(steps):
        x_n = np.c_[x_n, x_eval(n+i+1)]

    return (x_n, n)

#validate
# size=1000
# dt, n = generate_array(size=size, win=0)

# print(dt[:25,:])

# plt.plot(n, dt[:,0])
# plt.plot(n, dt[:,1], alpha=0.5, color='r')
# plt.show()




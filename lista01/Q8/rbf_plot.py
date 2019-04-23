import numpy as np 
import matplotlib.pyplot as plt 

x = np.linspace(-4,4,1000)

xx, yy = np.meshgrid(x, x)

zz = xx + 1j*yy

centers = [1+1j*1, 0 + 1j*0, -1 - 1j*1]

f = lambda x, y, center: np.exp(- np.abs(zz - center) / 0.5)

#plt.contourf(xx, yy, f(xx, yy))
fig, ax = plt.subplots()

for center in centers:
    ax.contour(xx, yy, f(xx, yy, center))

plt.show()



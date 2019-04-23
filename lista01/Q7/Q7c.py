import generateDataSet_7c as gn 
import numpy as np 
from keras import models, layers
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

dt_train = gn.generate_array(100)
dt_val = gn.generate_array(50)

#in_array, out_array = dt[:,:2], dt[:,2]

#X_train, Y_train = in_array[:7500], out_array[:7500]
X_train, Y_train = dt_train[:,:2], dt_train[:,2]
#X_val, Y_val = in_array[7500:], out_array[7500:]
X_val, Y_val = dt_val[:,:2], dt_val[:,2]

net = models.Sequential()
net.add(layers.Dense(8, activation='tanh', input_shape=(2,)))
net.add(layers.Dense(8, activation='tanh'))
net.add(layers.Dense(4, activation='tanh'))
net.add(layers.Dense(1))

net.compile(optimizer='rmsprop', loss='mse', metrics=['mse'])
print("Fitting")
history = net.fit(X_train, Y_train, epochs=7, batch_size=10, validation_data=(X_val, Y_val), verbose=1)
print("Finished")

loss = history.history['loss']
val_loss = history.history["val_loss"]

acc = history.history['mean_squared_error']
val_acc = history.history['val_mean_squared_error']

epochs = range(1, len(loss)+1)

#fig, (ax1, ax2) = plt.subplots(2,1)
plt.figure()
# ax1.plot(epochs, loss, 'b-', label='Training loss')
# ax1.plot(epochs, val_loss, 'bo', label='Validation loss')
# ax1.set_xlabel('Epochs')
# ax1.set_ylabel('Loss')
# ax1.legend()

plt.plot(epochs, acc, 'r-', label='MSE treinamento')
plt.plot(epochs, val_acc, 'ro', label='MSE validação')
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.legend()

plt.savefig('/home/dan/redes_neurais/fig_7b_epochs.png')
plt.show()

pred = net.predict(X_train)
np.savetxt("result_7c.csv", np.c_[dt_train, pred], delimiter=',')

# dimen = int(np.sqrt(len(X_train)))
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_trisurf(dt_train[:,0], dt_train[:,1], Y_train,cmap=cm.coolwarm)
plt.savefig('/home/dan/redes_neurais/7c_original.png')

# plt.show()

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_trisurf(dt_train[:,0], dt_train[:,1], pred.reshape(10000,),cmap=cm.coolwarm)
plt.savefig('/home/dan/redes_neurais/7c_pred.png')

# plt.show()

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_trisurf(dt_train[:,0], dt_train[:,1], Y_train - pred.reshape(10000,),cmap=cm.coolwarm)
plt.savefig('/home/dan/redes_neurais/7c_comp.png')

# plt.show()


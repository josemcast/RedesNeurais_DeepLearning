import generateDataSet_7b as gn 
import numpy as np 
from keras import models, layers
import matplotlib.pyplot as plt 

dt = gn.generate_array(100)
in_array, out_array = dt[:,:2], dt[:,2]

X_train, Y_train = in_array[:7500], out_array[:7500]
X_val, Y_val = in_array[7500:], out_array[7500:]

net = models.Sequential()
net.add(layers.Dense(2, activation='tanh', input_shape=(2,)))
net.add(layers.Dense(2, activation='tanh'))
net.add(layers.Dense(1))

net.compile(optimizer='rmsprop', loss='mse', metrics=['mse'])
history = net.fit(X_train, Y_train, epochs=15, batch_size=10, validation_data=(X_val, Y_val))

#print loss per epoch
loss = history.history['loss']
val_loss = history.history["val_loss"]

acc = history.history['mean_squared_error']
val_acc = history.history['val_mean_squared_error']

epochs = range(1, len(loss)+1)

fig, (ax1, ax2) = plt.subplots(2,1)

ax1.plot(epochs, loss, 'b-', label='Training loss')
ax1.plot(epochs, val_loss, 'bo', label='Validation loss')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')
ax1.legend()

ax2.plot(epochs, acc, 'r-', label='Training MSE')
ax2.plot(epochs, val_acc, 'ro', label='Validation MSE')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('MSE')
ax2.legend()

plt.show()
import generateDataSet_7a as gn 
import numpy as np 
from keras.utils import to_categorical
from keras import models, layers
from keras import optimizers
import matplotlib.pyplot as plt 

dt = gn.generate_array(1000)

x_train, y_train = dt[:,:3], dt[:,3]
#y_train = to_categorical(y_train)

x_val = x_train[:6000]
partial_x_train = x_train[6000:]
y_val = y_train[:6000]
partial_y_train = y_train[6000:]

print(x_val)
print(y_val.shape)

net = models.Sequential()
net.add(layers.Dense(2, activation='relu', input_shape=(3,)))
net.add(layers.Dense(2, activation='relu'))
net.add(layers.Dense(1))

#rms = optimizers.RMSprop(lr=0.003)
rms = optimizers.SGD(lr=0.005)
net.compile(optimizer=rms, loss='mse', metrics=['mae'])
history = net.fit(partial_x_train, partial_y_train, epochs=15, batch_size=1, validation_data=(x_val, y_val))

#print loss per epoch
loss = history.history['loss']
val_loss = history.history["val_loss"]

mae = history.history['mean_absolute_error']
val_mae = history.history['val_mean_absolute_error']

epochs = range(1, len(loss)+1)

fig, (ax1, ax2) = plt.subplots(2,1)

ax1.plot(epochs, loss, 'b-', label='MSE treinamento')
ax1.plot(epochs, val_loss, 'bo', label='MSE validação')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')
ax1.legend()

ax2.plot(epochs, mae, 'r-', label='MAE treinamento')
ax2.plot(epochs, val_mae, 'ro', label='MAE validação')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('MAE')
ax2.legend()

plt.show()

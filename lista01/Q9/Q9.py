import generateDataset_9 as gn 
import numpy as np 
from keras.utils import to_categorical
from keras import models, layers
import matplotlib.pyplot as plt 


size=1000
n_epochs=20
win_size = 5  # past memory
step=3
dt, n = gn.generate_array(size=size, steps = step, win = win_size)

#one-step
dt_one_x, dt_one_y = dt[:round(0.80*size), :win_size+1], dt[:round(0.80*size), win_size+1]
dt_one_x_val, dt_one_y_val = dt[round(0.80*size):, :win_size+1], dt[round(0.80*size):, win_size+1]

def model_build():
    model_one = models.Sequential()
    model_one.add(layers.Dense(4, activation='relu', input_shape=(dt_one_x.shape[1],)))
    #model_one.add(layers.Dense(2, activation='relu'))
    model_one.add(layers.Dense(1))

    model_one.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])

    return model_one 

print("Fitting")
fig, ax = plt.subplots(3,1)

for i in range(step):

    model = model_build()
    dt_one_x, dt_one_y = dt[:round(0.80*size), :win_size+1], dt[:round(0.80*size), win_size+1+i]
    dt_one_x_val, dt_one_y_val = dt[round(0.80*size):, :win_size+1], dt[round(0.80*size):, win_size+1+i]
    
    history = model.fit(dt_one_x, dt_one_y, epochs=n_epochs, batch_size=1, validation_data=(dt_one_x_val, dt_one_y_val))
    print("Finished")

    loss = history.history['loss']
    val_loss = history.history["val_loss"]

    epochs = range(1, len(loss)+1)
    ax[i].plot(epochs, loss, 'r-', label=f'MSE treinamento {i+1} passo(s)')
    ax[i].plot(epochs, val_loss, 'ro', label=f'MSE validação {i+1} passo(s)')
    ax[i].set_xlabel('Epochs')
    ax[i].set_ylabel('MSE')
    ax[i].legend()

plt.show()



# #print loss per epoch
# loss = history.history['loss']
# val_loss = history.history["val_loss"]
# print(history.history.keys())
# mae = history.history['mean_absolute_error']
# val_mae = history.history['val_mean_absolute_error']

# epochs = range(1, len(loss)+1)

# #fig, (ax1, ax2) = plt.subplots(2,1)
# #fig, ax1 = plt.figure()

# plt.plot(epochs, loss, 'b-', label='MSE treinamento ')
# plt.plot(epochs, val_loss, 'bo', label='MSE validação')
# plt.xlabel('Epochs')
# plt.ylabel('MSE')
# plt.legend()

# ax2.plot(epochs, mae, 'r-', label='Training mae')
# ax2.plot(epochs, val_mae, 'ro', label='Validation mae')
# ax2.set_xlabel('Epochs')
# ax2.set_ylabel('Accuracy')
# ax2.legend()

# plt.show()
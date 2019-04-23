import generateDataSet_7c as gn 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import RBFNet as rbf

n_epochs=5
size=100
dt = gn.generate_array(size)
size_val = 50
dt_val = gn.generate_array(size_val)
x_train, y_train = dt[:,:2], dt[:,2]
x_train_mod = np.c_[x_train, np.ones(len(x_train))]
x_val, y_val = dt_val[:,:2], dt_val[:,2]
x_val_mod = np.c_[x_val, np.ones(len(x_val))]

#x_train_mod = x_train.copy()
#X_train, Y_train = x_train_mod[:8000], y_train[:8000]
#val_x_train, val_y_train = x_train_mod[8000:], y_train[8000:]

number_of_neurons = round(0.025*len(x_train_mod))

print(f"Number of Neurons: {number_of_neurons}")
print("Fitting...")
rbfnet = rbf.rbf_net(number_of_neurons, epochs=n_epochs)
rbfnet.fit(x_train_mod, y_train, val_data=(x_val_mod, y_val))

print("Predicting...")
pred_x = rbfnet.predict(x_train_mod)
print("Finished...")

np.savetxt('pred_x_train.txt', np.c_[x_train_mod[:,:2], pred_x], delimiter=',')

plt.plot(range(n_epochs), rbfnet.loss_epoch, 'bo-', label="MSE por época treinamento")
plt.plot(range(n_epochs), rbfnet.loss_epoch_val, 'ro-', label="MSE por época validação")
plt.legend()
plt.show()
plt.plot()
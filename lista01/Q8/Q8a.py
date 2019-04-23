import generateDataSet_7a as gn 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.metrics import mean_squared_error
import RBFNet as rbf

dt = gn.generate_array(1000)
x_train, y_train = dt[:,:3], dt[:,3]
x_train_mod = np.c_[x_train, np.ones(len(x_train))]
#x_train_mod = x_train.copy()
X_train, Y_train = x_train_mod[:6000], y_train[:6000]
val_x_train, val_y_train = x_train_mod[6000:], y_train[6000:]

number_of_neurons = 8
n_epochs = 5        

rbfnet = rbf.rbf_net(number_of_neurons, epochs=n_epochs)
rbfnet.fit(X_train, Y_train, val_data=(val_x_train, val_y_train))


#pred = rbfnet.predict(val_x_train)


# print(pred.shape)
# print(val_y_train.shape)
# print(rbfnet.w)

#mse = mean_squared_error(val_y_train, pred)

#print(f"MSE: {mse:.2f}")

#np.set_printoptions(2)
#print(pred[:10].T)
#print(val_y_train[:10])

#print(np.array(rbfnet.loss_epoch_val).shape)
plt.plot(range(n_epochs), rbfnet.loss_epoch, 'bo-', label="MSE por época treinamento")
plt.plot(range(n_epochs), rbfnet.loss_epoch_val, 'ro-', label="MSE por época validação")
plt.legend()
plt.show()

    


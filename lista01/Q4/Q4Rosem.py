import matplotlib.pyplot as plt
import numpy as np
import generateDataSet_4 as gn 

class neuralNet():

    def __init__(self, n_neurons, n_dim, epochs=5):

        self.n_neurons = n_neurons
        self.w = np.zeros((self.n_neurons, n_dim))
        self.epochs = epochs
    
    @staticmethod
    def d_func (cls):
        #print(cls)
        if (cls==1):
            return 1
        else:
            return -1

    def signum(self, x, neuron):
        
        if (np.dot(x, self.w[neuron,:]) > 0):
            return 1
        else:
            return -1
    def signum_output(self, x, neuron):
        if (np.dot(x, self.w[neuron,:]) > 0):
            return 1
        else:
            return 0
    def activation_func(self, x, neuron):
        return self.signum(x, neuron)
    
    def fit (self, x, labels, learning_rate=0.08):
        
        for _ in range(self.epochs):
            for k, j in enumerate(x):
                for i in range(self.n_neurons):
                    y = self.activation_func(j, i)
                    self.w[i,:] += learning_rate*(neuralNet.d_func(labels[k][i]) - y)*j 
        
        return self.w

    def predict(self, x_pred):

        pred = []
        for j in x_pred:
            row = []
            for i in range(self.n_neurons):
                row.append(self.signum_output(j, i))
            pred.append(row)
        
        pred = np.array(pred).round()
        return pred


dt = gn.generate_array(1000, 0.09)

x_train, y_train = dt[:,:3], dt[:,4:]
x_train_mod = np.c_[x_train, np.ones(len(x_train))]
X_train, Y_train = x_train_mod[:6000], y_train[:6000]
val_x_train, val_y_train = x_train_mod[6000:], y_train[6000:]

number_of_neurons = 3
myNet = neuralNet(number_of_neurons, x_train_mod.shape[1], epochs=1)
w = myNet.fit(X_train, Y_train, learning_rate=0.004)
        
pred = myNet.predict(val_x_train)

print(pred)
print(w)


count = 0 
for i,j in zip(val_y_train, pred):
    
    if (np.array_equal(i,j)):
        count +=1

print("accuracy: {}".format(count/len(pred)))




import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import distance

class rbf_net():

    def __init__(self, n_neurons, epochs=5):

        self.n_neurons = n_neurons
        self.w = np.zeros((self.n_neurons, 1))
        self.epochs = epochs
        #self.lr = lr
        self.loss_epoch = []
        self.loss_epoch_val = []
        self.P = (10)*np.eye(len(self.w))
    def rbf_activation(self, x, c, s):
        #return np.exp(- 1 / (2 * s**2) * (x-c)**2)
        return np.exp(- 1 / (2 * s**2) * (distance.euclidean(x,c)**2))
 
    def generate_clusters(self, x):

        #idx = np.random.randint(len(x), size=self.n_neurons)
        return KMeans(n_clusters=self.n_neurons, random_state=42).fit(x).cluster_centers_
        #return x[idx,:]
    
    def fit (self, X, y, val_data=None):

        self.centers = self.generate_clusters(X)
        dMax = np.max([np.abs(c1 - c2) for c1 in self.centers for c2 in self.centers])
        self.stds = dMax / np.sqrt(2*self.n_neurons)
        for ep in range(self.epochs):
            print(f"epoch: {ep}")
            loss = 0
            for k, j in enumerate(X):
                a = []
                for i in range(self.n_neurons):
                    a.append (self.rbf_activation(j, self.centers[i], self.stds))
                
                a = np.array(a).reshape(self.n_neurons, 1)

                self.P -= ((self.P@a)@(a.T))@self.P / (1 + (a.T@self.P)@a)
                
                g = self.P@a
                #print(g)
                alpha = y[k] - a.T.dot(self.w)
                loss += alpha
                # online update
                self.w = self.w + g*alpha
            if val_data == None:
                self.loss_epoch.append(float(loss)/len(X))
                pass
            else:
                pred = self.predict(val_data[0])
                pred.shape = (len(pred), )
                self.loss_epoch_val.append(sum( (pred - val_data[1])**2)/ len(pred))
                self.loss_epoch.append(float(loss)/len(X))

            

    def predict(self, X):
        y_pred = []
        
        for j in X:
            out = 0
            a = []
            for i in range(self.n_neurons):
                a.append (self.rbf_activation(j, self.centers[i], self.stds))
            
            a = np.array(a)
            out = a.T.dot(self.w)
            y_pred.append(out)
        
        return np.array(y_pred)

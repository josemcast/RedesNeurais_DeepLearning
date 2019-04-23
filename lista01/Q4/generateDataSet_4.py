import numpy as np 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D #testes para ver se tá gerando no cubo

np.random.seed(42)

digits = [[0,0,0],
        [0,0,1],
        [0,1,0],
        [0,1,1],
        [1,0,0],
        [1,0,1],
        [1,1,0],
        [1,1,1]]

digits = iter(digits)
def generate_array(n, sigma):
    dataset = []
    dataset_original = []
    labels = []    
    for i in [0,1,2,3,4,5,6,7]:
        actual_digits = np.array(next(digits))
        for _ in range(n):
            dataset.append((actual_digits + np.random.randn(1,3)*sigma)[0,:])
            dataset_original.append(actual_digits)
            labels.append(i)
    
    dt = np.append(np.array(dataset), np.array(labels).reshape(8*n,1), axis=1)
    dt = np.append(dt, np.array(dataset_original), axis=1)
    np.random.shuffle(dt)
    return dt

# dataframe = generate_array(1000, 0.120)

# print(dataframe)

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(dataframe[:,0], dataframe[:,1], dataframe[:,2], c='b', marker='o')

# plt.show()

#fazer shuffle antes de usar para o resto
#fazer um print bonito para deixar legal de ver
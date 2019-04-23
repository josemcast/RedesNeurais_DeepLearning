import numpy as np 
import matplotlib.pyplot as plt 


np.random.seed(42)

digits = [[0,0,0],
        [0,0,1],
        [0,1,0],
        [0,1,1],
        [1,0,0],
        [1,0,1],
        [1,1,0],
        [1,1,1]]

def f_x(array):
    return array[0]^array[1]^array[2]

digits = iter(digits)
def generate_array(n):
    dataset = []
    labels = []    
    for i in [0,1,2,3,4,5,6,7]:
        actual_digits = np.array(next(digits))
        for _ in range(n):
            dataset.append(actual_digits)
            labels.append(np.apply_along_axis(f_x, 0, actual_digits))
    
    dt = np.append(np.array(dataset), np.array(labels).reshape(8*n,1), axis=1)
    np.random.shuffle(dt)
    return dt

#dataframe = generate_array(1000)

#print(dataframe.shape)


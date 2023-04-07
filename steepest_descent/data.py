import numpy as np

# tutorial data
D = np.array([[1, 3], [3, 6], [6, 5], [8, 7]])
# practice data
X = np.array([ 0.  ,  0.16,  0.22,  0.34,  0.44,  0.5 ,  0.67,  0.73,  0.9 ,  1.  ])
Y = np.array([-0.06,  0.94,  0.97,  0.85,  0.25,  0.09, -0.9 , -0.93, -0.53,  0.08])

def get_X(x = D[:, 0], dim = 1):
    n = len(x)
    X = np.ones((n, dim+1))
    
    for i in range(1, dim+1):
        for j in range(n):
            X[j, i] = x[j]**i
    
    return X

def get_Y():
    return D[:, 1]
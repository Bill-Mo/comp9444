import numpy as np

w = np.array([[0, 0, -1, 0, 0],
              [0, 0, 0, 0, 1], 
              [-1, 0, 0, 1, 0],
              [0, 0, 1, 0, -1],
              [0, 1, 0, -1, 0]])
a1 = [-1, 1, 1, 1, 1]
a2 = [1, 1, -1, -1, -1]
a3 = [1, 1, -1, 1, 1]

print(np.matmul(w, np.transpose(a1)))
print(np.matmul(w, np.transpose(a2)))
print(np.matmul(w, np.transpose(a3)))
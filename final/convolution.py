j = 65
k = 77
l = 3
m = 9
n = 9
s = 2

num_weights = 1 + m * n * l
width = 1 + (j - m) / s
height = 1 + (k - n) / s
num_neurons = width * height * 18
num_connection = num_neurons * num_weights
param = 18 * num_weights
print(num_weights, num_neurons, num_connection, param)
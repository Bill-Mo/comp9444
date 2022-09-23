from math import log2

def entropy(p):
	sum = 0
	for i in range(len(p)):
		sum += p[i]*(-log2(p[i]))
	return sum

def KL(p, q): 
	sum = 0
	for i in range(len(p)):
		sum += p[i] * (log2(p[i]) - log2(q[i]))
	return sum

p = [1/16, 1/16, 1/8, 1/4, 1/2]
q = [1/2, 1/8, 1/16, 1/4, 1/16]
print('entropy: {}'.format(entropy(p)))
print('KL: {}'.format(KL(p, q)))
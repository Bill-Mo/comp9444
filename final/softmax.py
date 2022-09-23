from math import e


z1 = 1.3
z2 = 2.1
z3 = 3.1

def prob(z):
    p = e**z / (e**z1 + e**z2 + e**z3)
    return p

def log_prob(z):
    return -prob(z)

print([log_prob(z1), 1+log_prob(z2), log_prob(z3)])
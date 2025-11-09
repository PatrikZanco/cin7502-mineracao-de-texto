import numpy as np

x= np.random.rand(5) 
w = np.random.rand(5)

b = 0.25
u = 0

x = np.linspace(-10, 10, 100)
z = 1/(1 + np.exp(-x))

print(z)
####


d1= np.random.rand(12) 

def merge():
    for i in range(len(d1)):
        z=d1[i]*w[i]
        u+=z

def main():
    u = merge()
    y = 1



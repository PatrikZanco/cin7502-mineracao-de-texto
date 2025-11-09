import numpy as np

# criando valores randomicos
x= np.random.rand(5) 
w = np.random.rand(5)


b = 1 # byas
u = 0

for i in range(len(x)):
    v = x[i]*w[i]
    u +=v

u = u - b

if u >= 0:
    y = 1
    print('neuronio ativado', u)

elif u < 0:
    print('desativado', u)
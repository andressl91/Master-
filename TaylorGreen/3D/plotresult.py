import matplotlib.pyplot as plt
import numpy as np

E_k = np.loadtxt('E_k.txt')
t_star = np.loadtxt('t_star.txt')

E_k = np.asarray(E_k)
t_star = np.asarray(t_star)

E_t = (E_k[1:] - E_k[:-1])/0.01

plt.figure(1)
plt.plot(t_star[:-1], E_t)
plt.show()

import matplotlib.pyplot as plt
import numpy as np

E_k = np.loadtxt('E_k.txt')
t_star = np.loadtxt('t_star.txt')

E_k = np.asarray(-E_k)
t_star = np.asarray(t_star)

E_t = (E_k[1:] - E_k[:-1])/0.01


plt.figure(2)
plt.title("Plot of Kinetic energy in the domain")
plt.xlabel('Time t* (t/L)')
plt.ylabel(' dE_k/dt')
plt.plot(t_star[:-1], E_t)

plt.figure(1)
plt.title("Plot of Dissipation in the domain")
plt.xlabel('Time t* (t/L)')
plt.ylabel(' dE_k/dt')
plt.plot(t_star[:-1], E_t)

#32, Re = 1000, dt = 0.001 REFERENCE
#20 30, Oasis, egne lsere, Choring, IPCS

#Spectral DNS, MIKAEM github
#kUN FOR VISSE CASER! Bruk til aa
#sammenligne resultater

#PYTHON SETUP PY.INSTALL
#FIL KAN KJORE TG!


#plt.figure(2)
#plt.plot((t_star, E_k))

#TIL NESTE GANG
#32**3 Re = 1000
#REferansedata

plt.show()

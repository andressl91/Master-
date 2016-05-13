import numpy as np
import matplotlib.pyplot as plt

#Chorin
chorin_Ek = np.loadtxt('./chorindata/N32_Re1000_dt1E-3/E_k.txt', delimiter=',')
chorin_dkdt = np.loadtxt('./chorindata/N32_Re1000_dt1E-3/dkdt.txt', delimiter=',')
t_star = np.loadtxt('./chorindata/N32_Re1000_dt1E-3/t_star.txt', delimiter=',')

#IPCS
#~/Desktop/Master-/TaylorGreen/3D/ipcsdata/N_32_Re_1000/E_k.txt
ipcs_Ek = np.loadtxt("./ipcsdata/N_32_Re_1000/E_k.txt", delimiter=',')
ipcs_dkdt =  np.loadtxt("./ipcsdata/N_32_Re_1000/dkdt.txt", delimiter=',')

#OASIS P1 vel
oasis_Ek = np.loadtxt("./oasisdata/N32_Re1000_VelP1/Ek.txt")
oasis_dkdt = np.loadtxt("./oasisdata/N32_Re1000_VelP1/dkdt.txt")

#OASIS P2 vel
oasis_Ek2 = np.loadtxt("./oasisdata/N32_Re1000_VelP2/Ek.txt")
oasis_dkdt2 = np.loadtxt("./oasisdata/N32_Re1000_VelP2/dkdt.txt")

plt.figure(3)
plt.plot(t_star[:100], chorin_Ek[:100], label='Chorin')
plt.plot(t_star[:100], ipcs_Ek[:100], label='IPCS')
plt.plot(t_star[:100], oasis_Ek[:100], label='Oasis IPCS_ABCN P1')
plt.plot(t_star[:100], oasis_Ek2[:100], label='Oasis IPCS_ABCN P2')
plt.xlabel("Time t_star = t/L")
plt.ylabel("Kinetic Energy E_k")
plt.legend(loc=3)
plt.savefig("plots/Compare_kin.png")

plt.figure(4)
#plt.title("Abs.Value Dissipation Compare \n N=32, Re = 1000 \n")
plt.plot(t_star[:100], -chorin_dkdt[:100], label='Chorin')
plt.plot(t_star[:100], -ipcs_dkdt[:100], label='IPCS')
plt.plot(t_star[:100], -oasis_dkdt[:100], label='Oasis IPCS_ABCN P1')
plt.plot(t_star[:100], -oasis_dkdt2[:100], label='Oasis IPCS_ABCN P2')
plt.xlabel("Time t_star = t/L")
plt.ylabel("Dissipation Energy E_k")
plt.axis([0, 10, 0, 0.017])
plt.legend(loc=2)
plt.savefig("plots/Compare_diss.png")

plt.show()

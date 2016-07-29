import numpy as np
import matplotlib.pyplot as plt

def Re_1000():
    #t_star
    t_star = np.loadtxt('./ipcsdata/N_32_Re_1000/t_star.txt', delimiter=',')
    print len(t_star)
    #Chorin
    chorin_Ek = np.loadtxt('./chorindata/N_32_Re_1000.0_14:40:41/E_k.txt', delimiter=',')
    chorin_dkdt = np.loadtxt('./chorindata/N_32_Re_1000.0_14:40:41/dkdt.txt', delimiter=',')


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

    print len(chorin_Ek), len(t_star)

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

def Re_1600():
    #t_star
    t_star = np.loadtxt('./ipcsdata/N=32_Re=1600.0_T=10.0_dt=0.0001/t_star.txt', delimiter=',')
    print len(t_star)
    #Chorin
    chorin_Ek = np.loadtxt('./chorindata/N=32_Re=1600.0_T=10.0_dt=0.0001/E_k.txt', delimiter=',')

    chorin_dkdt = np.loadtxt('./chorindata/N=32_Re=1600.0_T=10.0_dt=0.0001/dkdt.txt', delimiter=',')


    #IPCS
    #~/Desktop/Master-/TaylorGreen/3D/ipcsdata/N_32_Re_1000/E_k.txt
    ipcs_Ek = np.loadtxt("./ipcsdata/N=32_Re=1600.0_T=10.0_dt=0.0001/E_k.txt", delimiter=',')
    ipcs_dkdt =  np.loadtxt("./ipcsdata/N=32_Re=1600.0_T=10.0_dt=0.0001/dkdt.txt", delimiter=',')

    print len(ipcs_dkdt)
    """
    plt.figure(1)
    plt.plot(t_star, chorin_Ek[:-1], label='Chorin')
    plt.plot(t_star, ipcs_Ek, label='IPCS')
    plt.xlabel("Time t_star = t/L")
    plt.ylabel("Kinetic Energy E_k")
    plt.legend(loc=3)
    plt.axis([0, 10, 0, 0.14])
    plt.savefig("plots/Compare_kin_re1600.png")
    """

    plt.figure(4)
    #plt.title("Abs.Value Dissipation Compare \n N=32, Re = 1000 \n")
    plt.plot(t_star, -chorin_dkdt, label='Chorin')
    plt.plot(t_star, -ipcs_dkdt, label='IPCS')
    #plt.plot(t_star[:100], -oasis_dkdt[:100], label='Oasis IPCS_ABCN P1')
    #plt.plot(t_star[:100], -oasis_dkdt2[:100], label='Oasis IPCS_ABCN P2')
    plt.xlabel("Time t_star = t/L")
    plt.ylabel("Dissipation Energy E_k")
    #plt.axis([0, 10, 0, 0.017])
    plt.legend(loc=2)
    plt.savefig("plots/Compare_diss_re1000.png")

    plt.show()
    """
    #IPCS
    #~/Desktop/Master-/TaylorGreen/3D/ipcsdata/N_32_Re_1000/E_k.txt
    #ipcs_Ek = np.loadtxt("./ipcsdata/N_32_Re_1000/E_k.txt", delimiter=',')
    #ipcs_dkdt =  np.loadtxt("./ipcsdata/N_32_Re_1000/dkdt.txt", delimiter=',')
    """
Re_1600()

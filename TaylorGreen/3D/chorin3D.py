from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
import time
# Load mesh fdefault
def chorin(N, dt, T, L, nu, save_step):
    tic()
    mesh = BoxMesh(Point(-pi*L, -pi*L, -pi*L), Point(pi*L, pi*L, pi*L), N, N, N)

    def near(x, y, tol=1e-12):
        return bool(abs(x-y) < tol)

    class PeriodicDomain(SubDomain):
        def inside(self, x, on_boundary):
            return bool((near(x[0], -pi*L) or near(x[1], -pi*L) or near(x[2], -pi*L)) and
                    (not (near(x[0], pi*L) or near(x[1], pi*L) or near(x[2], pi*L))) and on_boundary)

        def map(self, x, y):
            if near(x[0], pi*L) and near(x[1], pi*L) and near(x[2], pi*L):
                y[0] = x[0] - 2.0*pi*L
                y[1] = x[1] - 2.0*pi*L
                y[2] = x[2] - 2.0*pi*L
            elif near(x[0], pi*L) and near(x[1], pi*L):
                y[0] = x[0] - 2.0*pi*L
                y[1] = x[1] - 2.0*pi*L
                y[2] = x[2]
            elif near(x[1], pi*L) and near(x[2], pi*L):
                y[0] = x[0]
                y[1] = x[1] - 2.0*pi*L
                y[2] = x[2] - 2.0*pi*L
            elif near(x[1], pi*L):
                y[0] = x[0]
                y[1] = x[1] - 2.0*pi*L
                y[2] = x[2]
            elif near(x[0], pi*L) and near(x[2], pi*L):
                y[0] = x[0] - 2.0*pi*L
                y[1] = x[1]
                y[2] = x[2] - 2.0*pi*L
            elif near(x[0], pi*L):
                y[0] = x[0] - 2.0*pi*L
                y[1] = x[1]
                y[2] = x[2]
            else: # near(x[2], pi):
                y[0] = x[0]
                y[1] = x[1]
                y[2] = x[2] - 2.0*pi*L

    constrained_domain = PeriodicDomain()
    test = PeriodicDomain()


    # Define function spaces (P2-P1)
    V = VectorFunctionSpace(mesh, "CG", 2, constrained_domain = test)
    Q = FunctionSpace(mesh, "CG", 1, constrained_domain = test)

    # Define trial and test functions
    u = TrialFunction(V)
    p = TrialFunction(Q)
    v = TestFunction(V)
    q = TestFunction(Q)

    # Set parameter values
    dt = dt
    T = T


    # Define time-dependent pressure boundary condition
    p_e = Expression('1./16.*(cos(2*x[0])+cos(2*x[1]))*(cos(2*x[2])+2)', degree=3)
    u_e = Expression(('sin(x[0]/L)*cos(x[1]/L)*cos(x[2]/L)', \
            '-cos(x[0]/L)*sin(x[1]/L)*cos(x[2]/L)', \
            '0'), L = L, degree=3)

    #p_e = interpolate(p_e, Q)


    # Define boundary conditions
    bcu = []
    bcp = []

    # Create functions
    u0 = interpolate(u_e, V)
    p0 = interpolate(p_e, Q)
    u1 = Function(V)
    p1 = Function(Q)

    # Define coefficients
    k = Constant(dt)
    f = Constant((0, 0, 0))

    # Tentative velocity step
    F1 = (1/k)*inner(u - u0, v)*dx + inner(grad(u0)*u0, v)*dx + \
         nu*inner(grad(u), grad(v))*dx - inner(f, v)*dx
    a1 = lhs(F1)
    L1 = rhs(F1)

    # Pressure update
    a2 = inner(grad(p), grad(q))*dx
    L2 = -(1/k)*div(u1)*q*dx

    # Velocity update
    a3 = inner(u, v)*dx
    L3 = inner(u1, v)*dx - k*inner(grad(p1), v)*dx

    # Assemble matrices
    A1 = assemble(a1); A2 = assemble(a2); A3 = assemble(a3)
    b1 = None; b2 = None; b3 = None

    # Create files for storing solution
    #ufile = File("velocity/velocity.pvd")
    #pfile = File("pressure/pressure.pvd")

    # Time-stepping
    #t = dt
    t = 0
    t_star.append(0)
    E_k.append(assemble(0.5*dot(u0, u0)*dx) / (2*pi)**3)
    count = 0; kin = np.zeros(1)
    while t < T:
        if MPI.rank(mpi_comm_world()) == 0:
            print "Iterating for time %.4g" % t
        time.append(t)
        # Compute tentative velocity step
        begin("Computing tentative velocity")
        b1 = assemble(L1, tensor=b1)
        pc = PETScPreconditioner("jacobi")
        sol = PETScKrylovSolver("bicgstab", pc)
        sol.solve(A1, u1.vector(), b1)
        #b1 = assemble(L1)
        #solve(A1, u1.vector(), b1, "bicgstab", "hypre_euclid")
        end()

        # Pressure correction
        begin("Computing pressure correction")
        b2 = assemble(L2, tensor=b2)
        solve(A2, p1.vector(), b2, "gmres", "hypre_amg") #cg
        end()

        # Velocity correction
        begin("Computing velocity correction")
        b3 = assemble(L3, tensor=b3)
        pc2 = PETScPreconditioner("jacobi")
        sol2 = PETScKrylovSolver("bicgstab", pc2)
        sol2.solve(A3, u1.vector(), b3)
        #b3 = assemble(L3)
        #solve(A3, u1.vector(), b3, "cg", "hypre_euclid")
        end()

        if (count % save_step == 0 or count % save_step == 1):
            kinetic = assemble(0.5*dot(u1, u1)*dx) / (2*pi)**3
            print "Loop kick at %.5f" % (count*dt)
            if (count % save_step == 0):
                kin[0] = kinetic
            else :#(count % save == 1):
                print "Total kinetic energy %.4f " % kinetic
                t_star.append(t/L)
                E_k.append(kinetic)
                diss = (kinetic-kin[0])/dt
                dkdt.append(diss)

        count += 1
        u0.assign(u1)
        p0.assign(p1)
        t += dt


    degree = V.dim() #DOF Degrees of freedom
    time_calc.append(toc())
    #plot(u0, interactive=True)

set_log_active(False)
error = []; dof = []; K = []; time_calc = []
E_k = []; time = [0]; t_star = []
N = [32]; dkdt = [];
L = 1.; nu = 0.002; dt=0.0001; T = 20.
Re = L*1./nu
print "Reynolds number %.1f" % Re
#Watch nu
for i in N:
    chorin(i, dt=dt, T = T, L = L, nu = nu, save_step = 1000)
if MPI.rank(mpi_comm_world()) == 0:
    import time, os
    clock = time.strftime("%H:%M:%S")
    s = "N_" + str(N[0]) + "_Re_"+str(Re) + "_"
    os.system("mkdir chorindata/"+s+clock)
    np.savetxt('chorindata/' +s+ clock + '/dkdt.txt', dkdt, delimiter=',')
    np.savetxt('chorindata/'+s + clock + '/E_k.txt', E_k, delimiter=',')
    np.savetxt('chorindata/'+s+ clock + '/t_star.txt', t_star, delimiter=',')

    plt.figure(1)
    plt.title("Kinetic Energy, Time %.1f, Re = %.1f" % (T, Re))
    plt.xlabel('Time t* (t/L),  dt = %.4f' % dt)
    plt.ylabel('E_k')
    plt.plot(t_star, E_k)
    plt.savefig('plots/Chorin_Ek.png')

    E_k = np.asarray(E_k); t_star = np.asarray(t_star)
    E_t = (E_k[1:] - E_k[:-1])/dt
    plt.figure(2)
    plt.title("Dissipation Time %.1f, Re = %.1f" % (T, Re))
    plt.xlabel('Time t* (t/L),  dt = %.4f' % dt)
    plt.ylabel('dE_k/dt')
    plt.plot(t_star[:-1], E_t)
    plt.savefig('plots/Chorin_dissipation.png')
    #plt.show()


#T = np.loadtxt('test.txt')
#print T

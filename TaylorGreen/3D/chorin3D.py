from dolfin import *
import numpy as np
import matplotlib.pyplot as plt

set_log_active(False)

# Load mesh fdefault
def NS(N, dt, T, L, nu):
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
    t = dt
    t_star.append(0)
    E_k.append(assemble(0.5*dot(u0, u0)*dx) / (2*pi)**3)

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

        # Plot solution
        #plot(p1, title="Pressure", rescale=True)
        #plot(u1, title="Velocity", rescale=True)

        # Save to file
        #ufile << u1
        #pfile << p1

        # Move to next time step
        t_star.append(t/L)
        E_k.append(assemble(0.5*dot(u1, u1)*dx) / (2.*pi*L)**3)
        u0.assign(u1)
        t += dt


    degree = V.dim() #DOF Degrees of freedom
    time_calc.append(toc())
    #plot(u0, interactive=True)

error = []; dof = []; K = []; time_calc = []
E_k = []; time = [0]; t_star = []
N = [10];
L = 1.; nu = 0.01; dt=0.01
Re = L*1./nu
print "Reynolds number %.1f" % Re
#Watch nu
for i in N:
    NS(i, dt=dt, T = 10, L = L, nu = nu)
if MPI.rank(mpi_comm_world()) == 0:

    print time_calc
    plt.figure(1)
    plt.title("Plot of Kinetic Energy in the domain Re = %.1f" % Re)
    plt.xlabel('Time t* (t/L),  dt = %.2f' % dt)
    plt.ylabel('E_k')
    plt.plot(t_star, E_k)
    plt.show()
    np.savetxt('E_k.txt', E_k, delimiter=',')
    np.savetxt('t_star.txt', t_star, delimiter=',')

#T = np.loadtxt('test.txt')
#print T

from dolfin import *
import numpy as np
import matplotlib.pyplot as plt

set_log_active(False)

# Load mesh fdefault
def ipcs(N, dt, T, L, rho, mu):

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


    nu = Constant(mu/rho)
    if MPI.rank(mpi_comm_world()) == 0:
        print "Reynolds number %.2f" % (L/nu)

    # Define function spaces (P2-P1)
    V = VectorFunctionSpace(mesh, "CG", 2, constrained_domain = test)
    Q = FunctionSpace(mesh, "CG", 1, constrained_domain = test)

    u = TrialFunction(V)
    p = TrialFunction(Q)
    v = TestFunction(V)
    q = TestFunction(Q)

    # Define time-dependent pressure boundary condition

    # Set parameter values
    dt = dt
    T = T


    # Define time-dependent pressure boundary condition
    p_e = Expression('1./16.*(cos(2*x[0]/L)+cos(2*x[1]/L))*(cos(2*x[2]/L)+2)', L = L, degree=3)
    u_e = Expression(('sin(x[0]/L)*cos(x[1]/L)*cos(x[2]/L)', '-cos(x[0]/L)*sin(x[1]/L)*cos(x[2]/L)', '0'),\
	L = L, degree=3)

    #Create Functions
    #p_e = interpolate(p_e, Q)
    u0 = interpolate(u_e, V)
    p0 = interpolate(p_e, Q)

    u1 = Function(V)
    p1 = Function(Q)

    # Define boundary conditions
    bcu = []
    bcp = []

    f =Constant((0,0,0)) #BODYFORCE

    #plot(boundaries, interactive=True)

    def sigma (u_o, p_o):
        return 2.0*mu*sym(grad(u_o)) - p_o*Identity(len(u_o))

    def eps(u):
        return sym(grad(u))

    n = FacetNormal(mesh)

    #STEP 1: TENTATIVE VELOCITY
    F = inner(rho*1./dt*(u - u0), v)*dx + dot(rho*dot(u0, nabla_grad(u0)), v)*dx \
    + inner( sigma(0.5*(u + u0), p0), eps(v) )*dx \
    - mu*dot(dot(grad(0.5*(u + u0)), n), v)*ds(3) + dot(p0*n ,v)*ds(3)- inner(f,v)*dx

    a1 = lhs(F)
    L1 = rhs(F)

    #STEP 2:PRESSURE CORRECTION
    a2 = dot(dt*grad(p), grad(q))*dx
    L2 = dot(dt*grad(p0), grad(q))*dx - rho*div(u1)*q*dx

    #STEP 3: VELOCITY CORRECTION
    a3 = dot(rho*u, v)*dx
    L3 = dot(rho*u1, v)*dx + dot(dt*grad(p0-p1), v)*dx

    #ASSEMBLE MATRIX
    A1 = assemble(a1); A2 = assemble(a2); A3 = assemble(a3)
    b1 = None; b2 = None; b3 = None

    t = dt
    progress = Progress("Time-Stepping")
    #plot(u0, interactive=True)
    while(t < T):
        if MPI.rank(mpi_comm_world()) == 0:
            print "Iterating for time %.4g" % t

        begin("Solving Velocity star")
        b1 = assemble(L1, tensor=b1)
        pc = PETScPreconditioner("jacobi")
        sol = PETScKrylovSolver("bicgstab", pc)
        sol.solve(A1, u1.vector(), b1)
        end()

        begin("Pressure CORRECTION")
        b2 = assemble(L2, tensor=b2)
        [bc.apply(A2, b2) for bc in bcp]
        solve(A2, p1.vector(), b2, "gmres", "amg")
        #solve(A2, p1.vector(), b2, "minres", "amg")
        end()

        begin("Velocity CORRECTION")
        b3 = assemble(L3, tensor=b3)
        pc2 = PETScPreconditioner("jacobi")
        sol2 = PETScKrylovSolver("bicgstab", pc2)
        sol2.solve(A1, u1.vector(), b3)
        end()

        t_star.append(t/L)
        E_k.append(assemble(0.5*dot(u1, u1)*dx) / (2.*pi*L)**3)

        u0.assign(u1)
        p0.assign(p1)

        t += dt

    #plot(u0, interactive=True)
    #p_e.t = t; u_e.t = t
    #V1 = VectorFunctionSpace(mesh, "CG", 3, constrained_domain = test)
    #u_e = interpolate(u_e, V); p_e = interpolate(p_e, Q)
    #plot(p0, interactive = True); plot(p_e, interactive = True)

    time_calc.append(toc())


set_log_active(False)
N = [10]
rho = 1000.; mu = 1.; T= 10.; dt = 0.01; L = 1.
h = []; E = []; E_k = []; t_star = []; time_calc = []
for n in N:
    ipcs(N = n, dt = dt, T = T, L = L,rho = rho, mu = mu)


if MPI.rank(mpi_comm_world()) == 0:
	print N
	print E
	print time_calc
	Re = (L*rho/mu)
	plt.figure(1)
	plt.title("Plot of Kinetic Energy in the domain Re = %.1f" % Re)
	plt.xlabel('Time t* (t/L),  dt = %.2f' % dt)
	plt.ylabel('E_k')
	plt.plot(t_star, E_k)
	plt.show()
	np.savetxt('E_k.txt', E_k, delimiter=',')
	np.savetxt('t_star.txt', t_star, delimiter=',')


	"""
	for i in range(len(E)-1):
		#print E[i],E[i+1]
		#print h[i],h[i+1]
		r = np.log(E[i]-E[i+1])/np.log(h[i]-h[i+1] )
		print r
	"""

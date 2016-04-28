from dolfin import *
import numpy as np

def ipcs(N, dt, T, rho, mu):
    mesh = RectangleMesh(Point(-1, -1), Point(1, 1), N, N)

    class PeriodicDomain(SubDomain):

        def inside(self, x, on_boundary):
            # return True if on left or bottom boundary AND NOT on one of the two corners (0, 2) and (2, 0)
            return bool((near(x[0], -1) or near(x[1], -1)) and
                  (not ((near(x[0], -1) and near(x[1], 1)) or
                        (near(x[0], 1) and near(x[1], -1)))) and on_boundary)

        def map(self, x, y):
            if near(x[0], 1) and near(x[1], 1):
                y[0] = x[0] - 2.0
                y[1] = x[1] - 2.0
            elif near(x[0], 1):
                y[0] = x[0] - 2.0
                y[1] = x[1]
            else:
                y[0] = x[0]
                y[1] = x[1] - 2.0

    constrained_domain = PeriodicDomain()
    test = PeriodicDomain()

    nu = mu/rho
    if MPI.rank(mpi_comm_world()) == 0:
        print "Reynolds number %.2f" % (2./nu)

    # Define function spaces (P2-P1)
    V = VectorFunctionSpace(mesh, "CG", 2, constrained_domain = test)
    Q = FunctionSpace(mesh, "CG", 1, constrained_domain = test)

    u = TrialFunction(V)
    p = TrialFunction(Q)
    v = TestFunction(V)
    q = TestFunction(Q)

    # Define time-dependent pressure boundary condition
    p_e = Expression("-0.25*(cos(2*pi*x[0]) + cos(2*pi*x[1]))*exp(-4*t*nu*pi*pi )", nu=nu, t=0.0)
    u_e = Expression(("-cos(pi*x[0])*sin(pi*x[1])*exp(-2*t*nu*pi*pi)",\
                        "cos(pi*x[1])*sin(pi*x[0])*exp(-2*t*nu*pi*pi)"), nu=nu, t=0)
    #u_0 = Function(V)
    #p_0 = Function(Q)
    u0 = interpolate(u_e, V)
    p0 = interpolate(p_e, Q)

    u1 = Function(V)
    p1 = Function(Q)

    # Define boundary conditions
    bcu = []
    bcp = []

    # Set parameter values
    dt = dt
    T = T

    f =Constant((0,0)) #BODYFORCE

    #plot(boundaries, interactive=True)

    def sigma (u_o, p_o):
        return 2.0*mu*sym(grad(u_o)) - p_o*Identity(len(u_o))

    def eps(u):
        return sym(grad(u))

    n = FacetNormal(mesh)

    #STEP 1: TENTATIVE VELOCITY
    F = inner(rho*1./dt*(u - u0), v)*dx + dot(rho*dot(u0, nabla_grad(u0)), v)*dx \
    + inner( sigma(0.5*(u + u0), p0), eps(v) )*dx \
    - mu*dot(dot(grad(0.5*(u + u0)), n), v)*ds(3) + dot(p0*n ,v)*ds(3) - inner(f,v)*dx

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
        [bc.apply(A1, b1) for bc in bcu]
        #solve(A1, u1.vector(), b1, "cg", "hypre_amg")
        solve(A1, u1.vector(), b1, "gmres", "hypre_amg")
        end()

        begin("Pressure CORRECTION")
        b2 = assemble(L2, tensor=b2)
        [bc.apply(A2, b2) for bc in bcp]
        solve(A2, p1.vector(), b2, "gmres", "amg")
        #solve(A2, p1.vector(), b2, "cg", "hypre_amg")
        end()

        begin("Velocity CORRECTION")
        b3 = assemble(L3, tensor=b3)
        [bc.apply(A3, b3) for bc in bcu]
        #solve(A3, u1.vector(), b3, "cg", "hypre_amg")
        solve(A3, u1.vector(), b3, "gmres", "hypre_amg")
        end()


        u0.assign(u1)
        p0.assign(p1)

        t += dt

    #plot(u0, interactive=True)
    p_e.t = t
    u_e.t = t
    V1 = VectorFunctionSpace(mesh, "CG", 3, constrained_domain = test)
    u_e = interpolate(u_e, V1)
    p_e = interpolate(p_e, Q)
    #plot(p0, interactive = True)
    #plot(p_e, interactive = True)
    L2_u = errornorm(u_e, u0, norm_type='l2')
    L2_p = errornorm(p_e, p1, norm_type='l2', degree_rise=3)
    #print "L2 norm velocity %.3f " % L2_u
    #print "L2 norm pressure %.3f " % L2_p
    h.append(mesh.hmin())
    E.append(L2_u)


set_log_active(False)
N = [2**i for i in range(3,7)]
N = [8 ,16, 32, 64, 74, 84]
#N = [25, 30, 35, 40, 45, 50, 55]
h = []; E = []
for n in N:
    ipcs(N = n, dt = 0.005, T = 1, rho = 500., mu = 1.)


if MPI.rank(mpi_comm_world()) == 0:
    print N
    print E
    for i in range(len(E)-1):
        #print E[i],E[i+1]
        #print h[i],h[i+1]
        r = np.log(E[i]-E[i+1])/np.log(h[i]-h[i+1] )
        print r

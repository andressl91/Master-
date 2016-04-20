from dolfin import *
import matplotlib.pyplot as plt

set_log_active(False)

# Load mesh from file
def NS(N):
    #N = int(N)
    mesh = RectangleMesh(Point(-1, -1), Point(1, 1), N, N, "right")
    #plot(mesh, interactive=True)

    class PeriodicDomain(SubDomain):

        def inside(self, x, on_boundary):
            # return True if on left or bottom boundary AND NOT on one of the two corners (0, 2) and (2, 0)
            return bool((near(x[0], -1) or near(x[1], -1)) and
                  (not ((near(x[0], -1) and near(x[1], 1)) or
                        (near(x[0], 1) and near(x[1], -1)))) and on_boundary)

        def map(self, x, y):
            if near(x[0], -1) and near(x[1], 1):
                y[0] = x[0] - 2.0
                y[1] = x[1] - 2.0
            elif near(x[0], -1):
                y[0] = x[0] - 2.0
                y[1] = x[1]
            else:
                y[0] = x[0]
                y[1] = x[1] - 2.0

    constrained_domain = PeriodicDomain()
    test = PeriodicDomain()



    # Define function spaces (P2-P1)
    V = VectorFunctionSpace(mesh, "CG", 2, constrained_domain = test)
    Q = FunctionSpace(mesh, "CG", 1)

    # Define trial and test functions
    u = TrialFunction(V)
    p = TrialFunction(Q)
    v = TestFunction(V)
    q = TestFunction(Q)

    # Set parameter values
    dt = 0.001
    T = 0.5
    nu = 0.01

    # Define time-dependent pressure boundary condition
    p_e = Expression("-0.25*(cos(2*pi*x[0]) + cos(2*pi*x[1]))*exp(-4*t*nu*pi*pi )", nu=nu, t=0.0)
    u_e = Expression(("-cos(pi*x[0])*sin(pi*x[1])*exp(-2*t*nu*pi*pi)",\
                    "cos(pi*x[1])*sin(pi*x[0])*exp(-2*t*nu*pi*pi)"), nu=nu, t=0)
    #u_e = interpolate(u_e, V)
    #p_e = interpolate(p_e, Q)
    #plot(u_e, interactive=True)

    # Define boundary conditions
    sides = DomainBoundary()
    boundaries = FacetFunction("size_t", mesh)
    boundaries.set_all(0)
    sides.mark(boundaries, 1)
    #plot(boundaries, interactive=True)

    bc_v = DirichletBC(V, u_e, boundaries, 1)
    bc_p = DirichletBC(Q, p_e, boundaries, 1)
    bcu = [bc_v]
    bcp = [bc_p]

    # Create functions
    u0 = Function(V)
    u1 = Function(V)
    p1 = Function(Q)

    # Define coefficients
    k = Constant(dt)
    f = Constant((0, 0))

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
    A1 = assemble(a1)
    A2 = assemble(a2)
    A3 = assemble(a3)

    # Use amg preconditioner if available
    prec = "amg" if has_krylov_solver_preconditioner("amg") else "default"

    # Create files for storing solution
    ufile = File("velocity/velocity.pvd")
    pfile = File("pressure/pressure.pvd")

    # Time-stepping
    t = dt
    while t < T + DOLFIN_EPS:
        # Update pressure boundary condition
        p_e.t = t
        u_e.t = t

        # Compute tentative velocity step
        begin("Computing tentative velocity")
        b1 = assemble(L1)
        [bc.apply(A1, b1) for bc in bcu]
        solve(A1, u1.vector(), b1, "gmres", "default")
        end()

        # Pressure correction
        begin("Computing pressure correction")
        b2 = assemble(L2)
        [bc.apply(A2, b2) for bc in bcp]
        solve(A2, p1.vector(), b2, "gmres", prec)
        end()

        # Velocity correction
        begin("Computing velocity correction")
        b3 = assemble(L3)
        [bc.apply(A3, b3) for bc in bcu]
        solve(A3, u1.vector(), b3, "gmres", "default")
        end()

        # Plot solution
        #plot(p1, title="Pressure", rescale=True)
        #plot(u1, title="Velocity", rescale=True)

        # Save to file
        #ufile << u1
        #pfile << p1

        # Move to next time step
        u0.assign(u1)
        t += dt
        print "t =", t



    degree = V.dim() #DOF Degrees of freedom
    L2_u= errornorm(u_e, u0, norm_type='l2', degree_rise=3)
    #L2_p = errornorm(p_e, p1, norm_type='l2', degree_rise=3)

    Kinetic = 0.5*L2_u*L2_u
    K.append(Kinetic)

    print 0.5*norm(u0, 'l2')**2
    u_e.t = t-dt
    V1 = VectorFunctionSpace(mesh, 'Lagrange', 3)
    print norm(interpolate(u_e, V1), 'l2')**2
    error.append(L2_u); dof.append(degree)

error = []; dof = []; K = []
N = [2**i for i in range(4,7)]

for i in N:
    NS(i)


plt.figure(1)
plt.loglog(dof, error, '*')
plt.show()

print "K VALUES", K
print "ERROR AND DOF", error, dof
print "N VALUES", N

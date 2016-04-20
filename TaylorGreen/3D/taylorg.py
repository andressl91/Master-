from dolfin import *

import matplotlib.pyplot as plt

set_log_active(False)

# Load mesh from file

def NS(N, dt, T):
    tic()

    mesh = BoxMesh(Point(-pi, -pi, -pi), Point(pi, pi, pi), N, N, N)

    def near(x, y, tol=1e-12):
        return bool(abs(x-y) < tol)

    class PeriodicDomain(SubDomain):

        def inside(self, x, on_boundary):
            return bool((near(x[0], -pi) or near(x[1], -pi) or near(x[2], -pi)) and
                    (not (near(x[0], pi) or near(x[1], pi) or near(x[2], pi))) and on_boundary)

        def map(self, x, y):
            if near(x[0], pi) and near(x[1], pi) and near(x[2], pi):
                y[0] = x[0] - 2.0*pi
                y[1] = x[1] - 2.0*pi
                y[2] = x[2] - 2.0*pi
            elif near(x[0], pi) and near(x[1], pi):
                y[0] = x[0] - 2.0*pi
                y[1] = x[1] - 2.0*pi
                y[2] = x[2]
            elif near(x[1], pi) and near(x[2], pi):
                y[0] = x[0]
                y[1] = x[1] - 2.0*pi
                y[2] = x[2] - 2.0*pi
            elif near(x[1], pi):
                y[0] = x[0]
                y[1] = x[1] - 2.0*pi
                y[2] = x[2]
            elif near(x[0], pi) and near(x[2], pi):
                y[0] = x[0] - 2.0*pi
                y[1] = x[1]
                y[2] = x[2] - 2.0*pi
            elif near(x[0], pi):
                y[0] = x[0] - 2.0*pi
                y[1] = x[1]
                y[2] = x[2]
            else: # near(x[2], pi):
                y[0] = x[0]
                y[1] = x[1]
                y[2] = x[2] - 2.0*pi

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
    nu = 0.005

    # Define time-dependent pressure boundary condition
    p_e = Expression('1./16.*(cos(2*x[0])+cos(2*x[1]))*(cos(2*x[2])+2)')
    u_e = Expression(('sin(x[0])*cos(x[1])*cos(x[2])', \
            '-cos(x[0])*sin(x[1])*cos(x[2])', \
            '0'))

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
    A1 = assemble(a1)
    A2 = assemble(a2)
    A3 = assemble(a3)

    # Create files for storing solution
    ufile = File("velocity/velocity.pvd")
    pfile = File("pressure/pressure.pvd")

    # Time-stepping
    t = dt
    print "STARTING TIME LOOP"
    while t < T:

        # Compute tentative velocity step
        begin("Computing tentative velocity")
        b1 = assemble(L1)
        #[bc.apply(A1, b1) for bc in bcu]
        solve(A1, u1.vector(), b1, "gmres", "default")
        end()

        # Pressure correction
        begin("Computing pressure correction")
        b2 = assemble(L2)
        #[bc.apply(A2, b2) for bc in bcp]
        solve(A2, p1.vector(), b2, "gmres", "amg")
        end()

        # Velocity correction
        begin("Computing velocity correction")
        b3 = assemble(L3)
        #[bc.apply(A3, b3) for bc in bcu]
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
    print "END TIMELOOP"

    degree = V.dim() #DOF Degrees of freedom
    time.append(toc())
    plot(u0, interactive=True)

error = []; dof = []; K = []; time = []
N = [10]

for i in N:
    NS(i, dt=0.01, T=0.2)

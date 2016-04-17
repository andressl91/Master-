from dolfin import *
import numpy as np

set_log_active(False)
E =[]; h = []
for n in [2**i for i in range(2,5)]:
    mesh = UnitSquareMesh(n ,n)

    #mesh = UnitCubeMesh(50,50,50)

    V = FunctionSpace(mesh, 'Lagrange', 2)

    u = TrialFunction(V)
    v = TestFunction(V)


    alpha = 3; beta = 1.2
    T = 1.8; dt = 0.1; t = dt
    f = Expression("beta - 2 - 2*alpha", beta=beta, alpha=alpha)

    g = Expression("1 + x[0]*x[0] + alpha*x[1]*x[1] + beta*t", alpha=alpha, \
                                                beta=beta, t=0, degree=2)
    u_0 = interpolate(g, V)

    bc = DirichletBC(V, g, "on_boundary")

    a = u*v*dx + dt*inner(grad(u), grad(v) ) *dx
    L = u_0*v*dx + dt*f*v*dx

    A = assemble(a)
    u_1 = Function(V)


    while t <= T:
            b = assemble(L)
            g.t = t
            bc.apply(A, b)

            solve(A, u_1.vector(), b)
            u_0.assign(u_1)
            t += dt

    #plot(u_0, interactive=False)
    u_e = project(g, V)

    L2 = errornorm(u_e, u_0, norm_type='l2', degree_rise=2)
    E.append(L2); h.append(mesh.hmin() )
    s = str(L2)
    print "Error is %s for %d points" % (s, n)

print E, h
for i in range(len(E)-1):
    r = np.log(E[i+1]-E[i]) / np.log(h[i+1] - h[i])
    print r

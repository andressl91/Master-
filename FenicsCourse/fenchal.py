from dolfin import *
import numpy as np
set_log_active(False)
for n in [2**i for i in range(2, 4)]:
    E = []; h = []
    for i in [1]:
        mesh = UnitSquareMesh(n, n)
        V = FunctionSpace(mesh, 'Lagrange', i)
        u = TrialFunction(V)
        v = TestFunction(V)

        f = Expression("2*pi*pi*sin( pi*x[0] ) * sin( pi*x[1] )", degree=4)

        u_ex = Expression("sin(pi*x[0]) * sin(pi*x[1])")

        a = inner(grad(u), grad(v))*dx
        L = f*v*dx(degree=4)

        bc0 = DirichletBC(V, 0, "on_boundary")

        u_h = Function(V)

        solve(a == L, u_h, bc0)

        #plot(u_h, interactive=True)

        u_e = interpolate(u_ex, V)
        L2 = errornorm(u_h, u_e, norm_type='l2', degree_rise=2)

        s = str(L2)
        E.append(L2); h.append(mesh.hmin() )

        print "For %d points, elements of degree %d the L2 norm is %s." % (n, i, s)

for i in range(len(E)-1):
    r = np.log(E[i+1]-E[i]) / np.log(h[i+1] - h[i])
    print r

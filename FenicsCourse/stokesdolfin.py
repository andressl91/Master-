from dolfin import *

mesh = Mesh("dolfin_fine.xml")

V = VectorFunctionSpace(mesh, 'Lagrange', 2)
P = FunctionSpace(mesh, 'Lagrange', 1)

W = V*P


w = Function(W)
u, p = TrialFunctions(W)
v, q = TestFunctions(W)

n = FacetNormal(mesh)
nu = 0.01
eps = grad(u) + grad(u).T
#def nu(u):
#    return 0.5*pow(grad(u)**2, 1.0/(2*(4-1)))

p0 = Expression("1 - x[0]", degree = 1)

F = (2*nu*inner(eps,grad(v)) - p*div(v) - div(u)*q ) * dx + \
    (p0*dot(v,n) ) * ds

#Find boundaries

inflow = AutoSubDomain(lambda x: near(x[0], 0))
outflow = AutoSubDomain(lambda x: near(x[0], 1))
nos = DomainBoundary()


boundaries = FacetFunction("size_t", mesh)
boundaries.set_all(0)
nos.mark(boundaries, 3)
inflow.mark(boundaries,1)
outflow.mark(boundaries, 2)
#plot(boundaries, interactive=True)

noslip = DirichletBC(W.sub(0), (0, 0), boundaries, 3)

up = Function(W)

solve(lhs(F) == rhs(F), up, noslip)

u, p = split(up)
plot(u, interactive=True)
plot(p, interactive=True)

from dolfin import *
import numpy as np

mesh = Mesh("lego_beam.xml")
V = VectorFunctionSpace(mesh, 'Lagrange', 1)
#u = TrialFunction(V) #SOLVING NEWTON REQUIRE Function
u = Function(V)
v = TestFunction(V)

################################################
#   CONSTANTS

lamb = Constant(0.0105*1E9)
mu = Constant(0.0023*1E9)
rho = Constant(1.45*1E3)
g = Constant(-9.81)

B = Constant((0,0,g*rho))
T = Constant((0, 0, -5000))

F = Identity(3) + grad(u)
C = F.T*F

E = 0.5 * (C - Identity(3))
E = variable(E)
W = lamb/2.*(tr(E))*(tr(E)) + mu * (tr(E*E))
S = diff(W, E)
P = F*S

#plot(mesh, interactive=True)

################################################3
#    DEFINE BOUNDARIES
wall = AutoSubDomain(lambda x: "on_boundary" and near(x[0], 0.0001) )
end = AutoSubDomain(lambda x: "on_boundary" and near(x[0], 0.0799) )
allwalls = DomainBoundary()

boundaries = FacetFunction("size_t", mesh)
boundaries.set_all(0)
allwalls.mark(boundaries, 0)
wall.mark(boundaries, 1)
end.mark(boundaries, 2)
#plot(boundaries, interactive=True)


#dsN = Measure('ds', subdomain_id=2, subdomain_data=boundaries)
ds = Measure("ds", subdomain_data=boundaries)

bc_wall = DirichletBC(V, ((0,0,0)), boundaries, 1)
F = inner( P, grad(v) )*dx - dot( B, v)*dx - dot(T,v)*ds(2)


#solve(lhs(F) == rhs(F), u, bc_wall)
solve(F == 0, u, bc_wall)

plot(u, interactive=True, mode='displacement')
#displace = np.max(u[2,:])#- np.min(u[2])**2

#FRA FORELESER
#assemble integrates function over given domain, restricted by dx/ds
displacement = assemble(u[2]*dx) / assemble(1.0*dx(mesh))
s = str(displacement)
#                                   Integrate brick to get volume
print "The average displacement %.16g" % displacement

###

average = np.sqrt( np.average(u.vector().array())**2 )
print average

print np.max( np.sqrt(u.vector().array()**2 ))

from dolfin import *
import numpy as np

mesh = Mesh("lego_beam.xml")
#DEINE MIXED SPACE
V = VectorFunctionSpace(mesh, 'Lagrange', 1)

VV = V*V
#u = TrialFunction(V) #SOLVING NEWTON REQUIRE Function
up = Function(VV)
u, p = split(up)
vq = TestFunction(VV)
v, q = split(vq)

up0 = Function(VV)
u0, p0 = split(up0)

up1 = Function(VV)
u1, p1 = split(up1)

################################################
#   CONSTANTS

lamb = Constant(0.0105*1E9)
mu = Constant(0.0023*1E9)
rho = Constant(1.45*1E3)
g = Constant(-9.81)

B = Constant((0,0,g*rho))
T = Constant((0, 0, -5000))

F = Identity(3) + grad(0.5*(u1 - u0))
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

bc_wall = DirichletBC(VV.sub(0), ((0,0,0)), boundaries, 1)

#dsN = Measure('ds', subdomain_id=2, subdomain_data=boundaries)
ds = Measure("ds", subdomain_data=boundaries)

Time = 0.5; dt = 0.1; t = dt
                                                        #inner ?
lhs = rho*dot((p1 - p0),v)*dx + dt*inner(P, grad(v))*dx + dot((u1 - u0),q)*dx \
    - dt*dot((p1 + p0)/2., q)*dx
rhs = dt*dot(B,v)*dx + dt*dot(T,v)*ds(2)
Non = lhs -rhs

while t < Time:
    F = Identity(3) + grad((u1 + u0)/2.)
    solve(Non == 0, up1, bc_wall)
    up0.assign(up1)
    t += dt
    displacement = assemble(u[2]*dx) / assemble(1.0*dx(mesh))
    print "The average displacement %.16g at time %.3f" % (displacement, t)

u, p = split(up1)
plot(u, interactive=True, mode='displacement' )

#plot(u, interactive=True, mode='displacement')
#displace = np.max(u[2,:])#- np.min(u[2])**2

#FRA FORELESER
#assemble integrates function over given domain, restricted by dx/ds
displacement = assemble(u[2]*dx) / assemble(1.0*dx(mesh))
s = str(displacement)
#                                   Integrate brick to get volume
print "The average displacement %s" % displacement

###

average = np.sqrt( np.average(u.vector().array())**2 )
print average

print np.max( np.sqrt(u.vector().array()**2 ))

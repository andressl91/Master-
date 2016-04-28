from dolfin import *

mesh = Mesh("dolfin_fine.xml")
V = VectorFunctionSpace(mesh, 'Lagrange', 2)
Q = FunctionSpace(mesh, 'Lagrange', 1)


u = TrialFunction(V)
p = TrialFunction(Q)

u_0 = Function(V)
p_0 = Function(Q)


u_1 = Function(V)
p_1 = Function(Q)

v = TestFunction(V)
q = TestFunction(Q)

rho = 1000; mu = 1
dt = 0.0005;

f =Constant((0,0)) #BODYFORCE

#FIX TO EXLUDE THE CORNERS
left = AutoSubDomain(lambda x: "on_boundary" and near(x[0],0) )
right = AutoSubDomain(lambda x: "on_boundary" and near(x[0],1) )
both = AutoSubDomain(lambda x: "on_boundary" and near(x[0],0) or near(x[0],1) )
nos = DomainBoundary()

boundaries = FacetFunction("size_t", mesh)
boundaries.set_all(0)
nos.mark(boundaries, 1)
#left.mark(boundaries,1)
#right.mark(boundaries, 2)
both.mark(boundaries, 3)

#dsN = Measure('ds', subdomain_id=2, subdomain_data=boundaries)
ds = Measure("ds", subdomain_data=boundaries)

bc = DirichletBC(V, (0,0), boundaries, 1 )
bc1 = DirichletBC(Q, 1, left)
bc2 = DirichletBC(Q, 0, right)

bc_velocity = [bc]
bc_pressure = [bc1, bc2]

#plot(boundaries, interactive=True)

def sigma (u_o, p_o):
    return 2.0*mu*sym(grad(u_o)) - p_o*Identity(len(u_o))

def eps(u):
    return sym(grad(u))

n = FacetNormal(mesh)

#STEP 1: TENTATIVE VELOCITY
F = inner(rho*1./dt*(u - u_0), v)*dx + dot(rho*dot(u_0, nabla_grad(u_0)), v)*dx \
+ inner( sigma(0.5*(u + u_0), p_0), eps(v) )*dx \
- mu*dot(dot(grad(0.5*(u + u_0)), n), v)*ds(3) + dot(p_0*n ,v)*ds(3) - inner(f,v)*dx

a1 = lhs(F)
L1 = rhs(F)

#STEP 2:PRESSURE CORRECTION
a2 = dot(dt*grad(p), grad(q))*dx
L2 = dot(dt*grad(p_0), grad(q))*dx - rho*div(u_1)*q*dx

#STEP 3: VELOCITY CORRECTION
a3 = dot(rho*u, v)*dx
L3 = dot(rho*u_1, v)*dx + dot(dt*grad(p_0-p_1), v)*dx

#ASSEMBLE MATRIX
A1 = assemble(a1); A2 = assemble(a2); A3 = assemble(a3)
b1 = None; b2 = None; b3 = None

t = dt; T = 0.1
progress = Progress("Time-Stepping")
while(t < T):
    print "Iterating for time %.4g" % t

    begin("Solving Velocity star")
    b1 = assemble(L1, tensor=b1)
    [bc.apply(A1, b1) for bc in bc_velocity]
    solve(A1, u_1.vector(), b1)#, "gmres", "ilu")
    end()

    begin("Pressure CORRECTION")
    b2 = assemble(L2, tensor=b2)
    [bc.apply(A2, b2) for bc in bc_pressure]
    solve(A2, p_1.vector(), b2)
    end()

    begin("Velocity CORRECTION")
    b3 = assemble(L3, tensor=b3)
    [bc.apply(A3, b3) for bc in bc_velocity]
    solve(A3, u_0.vector(), b3)
    end()

    u_0.assign(u_1)
    p_0.assign(p_1)

    t += dt

plot(u_0, interactive=True)

#Calculat the scaled ODE function of Navier-Stokes
#for the von Karman swirl flow
from dolfin import *

import numpy as np

def Ffunc(x, x_0, a):
    return (1. - x/x_0)**2 * (a*x + ( 2.*a/x_0 - 0.5) * x**2)

def Gfunc(x, x_0, a):
    return (1 - x/x_0)**2 * (1 + x/(2*x_0))

#x = np.linspace(0, 20, 101)

#a = 0.51023
#x_0 = 2./(3.*0.61592)

from sympy import *
x, a, x_0, b = symbols('x, a, x_0, b')
#b = -0.616
G_0 = -3./(2*x_0)
F_0 = a

F = (1 - x/x_0)**2 * (a*x + (2*a/x_0 - 0.5)*x**2)
G = (1 - x/x_0)**2 * (1 + x/(2*x_0))

# solve(G_0 + 4*integrate(F*G, (x, 0, x_0)), x)
sol =  solve([F_0 + integrate((3*F**2 - G**2), (x, 0, x_0)) ,
                G_0 + 4*integrate(F*G, (x, 0, x_0))], [x_0, a])

#print G_0.subs(x_0, sol[2][0])
#print F_0.subs(a, sol[2][1])

y = np.linspace(0, 5, 51)
f = np.vectorize(Ffunc)
g = np.vectorize(Gfunc)
#x_0 = sol[2][0]; a = sol[2][1]

x_0 = -3./(2*-0.616)
a = 0.510
for i in range(len(y)):
    print "y = %.1f F = %.4f" % (y[i] ,f(y[i], x_0, a) )

#import matplotlib.pyplot as plt
#plt.figure(1)
#plt.axis([0, 5, 0, 0.4])
#plt.plot(y, f(y, x_0, a))
#plt.show()

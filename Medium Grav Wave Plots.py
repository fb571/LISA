import matplotlib.pyplot as plt
import numpy as np
from scipy.special import jv  # Bessel function of the first kind

def A(a,n,e):
    ne = n*e
    return (a**2/n)*(jv(n-2,ne) - jv(n+2,ne) -2*e*(jv(n-1,ne)-jv(n+1,ne)))

def B(a,n,e):
    ne = n*e
    return (a**2/n)*(1-e**2)*(jv(n+2,ne)-jv(n-2,ne))

def C(a,n,e):
    ne = n*e
    return (a**2/n)*((1-e**2)**(1/2))*(jv(n+2,ne)+jv(n-2,ne)-e*(jv(n+1,ne)+jv(n-1,ne)))

#"Independent" variables (at least independent to begin with)
a0 = 0.5 #Initial Semimajor Axis
e0 = 0.2 #Initial Eccentricity
G = 1.24e-6 #Gravitational Constant
m1 = 1 #Mass of Primary
m2 = 1 #Mass of Secondary
c = 63072 #Speed of Light
r = 1 #Distance of Binary from Observer
I = 0 #Angle of Inclination
φ = 0 #Angle of Pericentre

#Number of timesteps and the time interval value
N = 780
h = 10

y0 = np.array([a0,e0])

def f(y):
    a = y[0]
    e = y[1]
    aprime = ((-64/5)*((G**3)*(m1**2)*(m2**2))*(m1+m2)/((a**5)*(1-e**2)**(7/2)))*(1+(73/24)*e**2+(37/96)*e**4)
    eprime = -(e*(304/15)*((G**3)*(m1**2)*(m2**2))*(m1+m2)/((a**4)*(1-e**2)**(5/2)))*(1+(121/304)*e**2)
    derivative = np.array([aprime,eprime])
    return derivative

def rk4(y0,func,N,h):
    M = np.zeros([N+1,2])
    M[0,:] = y0
    for i in range(1,N+1):
        y = M[i-1,:]
        k1 = h*func(y)
        k2 = h*func(y+0.5*k1)
        k3 = h*func(y+0.5*k2)
        k4 = h*func(y+k3)
        M[i,:] = y + (k1 + 2*k2 + 2*k3 + k4)/6
    return M

Y = rk4(y0,f,N,0.0001)
    
mu = (m1*m2)/(m1+m2)    

def hplus(a,e,t):
    HPLUS = np.linspace(0,0,10)
    ω = (G*(m1+m2)/a**3)**(1/2)

    for i in range(0,10):
        HPLUS[i] = ((A(a,i+1,e))*((np.cos(φ))**2-((np.sin(φ))**2)*(np.cos(I))**2)+(B(a,i+1,e))*((np.sin(φ))**2-((np.cos(φ))**2)*(np.cos(I))**2)-(C(a,i+1,e))*(np.sin(2*φ))*(1+(np.cos(I))**2))*np.cos((i+1)*ω*t)

    return -(mu*(ω**2)*G/(r*c**4))*np.sum(HPLUS)

def hcross(a,e,t):
    HCROSS = np.linspace(0,0,10)
    ω = (G*(m1+m2)/a**3)**(1/2)

    for i in range(0,10):
        HCROSS[i] = ((np.cos(2*φ))*(np.cos(I))*C(a,i+1,e)+(A(a,i+1,e)-B(a,i+1,e))*(np.sin(2*φ))*(np.sin(I)))*np.sin((i+1)*ω*t)

    return -(mu*(ω**2)*G/(r*c**4))*np.sum(HCROSS)

a = np.zeros(N+1)
e = np.zeros(N+1)
t = np.zeros(N+1)

for i in range(0,N+1):
    a[i] = Y[i,0]
    e[i] = Y[i,1]
    t[i] = i*h

hcr = np.zeros(N+1)
hpl = np.zeros(N+1)
for i in range(0,N+1):
    hcr[i] = hcross(a[i],e[i],t[i])
    hpl[i] = hplus(a[i],e[i],t[i])

plt.plot(t,hcr, label='H+ over time', color='blue')
plt.show()

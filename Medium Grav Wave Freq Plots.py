import matplotlib.pyplot as plt
import numpy as np
from scipy.special import jv  # Bessel function of the first kind
from scipy.signal import get_window

def A(a, nmax, e):
    n = np.arange(1, nmax+1)
    ne = n*e
    return (a**2/n)*(jv(n-2, ne) - jv(n+2, ne) - 2*e*(jv(n-1, ne) - jv(n+1, ne)))

def B(a, nmax, e):
    n = np.arange(1, nmax+1)
    ne = n*e
    return (a**2/n)*(1-e**2)*(jv(n+2, ne)-jv(n-2, ne))

def C(a, nmax, e):
    n = np.arange(1, nmax+1)
    ne = n*e
    return (a**2/n)*np.sqrt(1-e**2)*(jv(n+2, ne)+jv(n-2, ne)-e*(jv(n+1, ne)+jv(n-1, ne)))

#"Independent" variables (at least independent to begin with)
# Units of 0.001AU = 0.001year = 1solar mass = 1
a0 = 10 #Initial Semimajor Axis
e0 = 0.6 #Initial Eccentricity
G = 0.03948 #Gravitational Constant
m1 = 1 #Mass of Primary
m2 = 1 #Mass of Secondary
c = 63072 #Speed of Light
r = 1 #Distance of Binary from Observer
I = 0 #Angle of Inclination
φ = 0 #Angle of Pericentre

#Number of timesteps and the time interval value
N = 10000
h = 0.1

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

Y = rk4(y0,f,N,h)
    
mu = (m1*m2)/(m1+m2)    

def hplus(a, e, t, nmax=100):
    n = np.arange(1, nmax+1)
    ω = np.sqrt(G*(m1+m2)/a**3)

    A_n = A(a, nmax, e)
    B_n = B(a, nmax, e)
    C_n = C(a, nmax, e)

    cos_term = np.cos(n * ω * t)
    HPLUS = (
        A_n*((np.cos(φ))**2 - (np.sin(φ))**2*(np.cos(I))**2)
        + B_n*((np.sin(φ))**2 - (np.cos(φ))**2*(np.cos(I))**2)
        - C_n*(np.sin(2*φ))*(1+(np.cos(I))**2)
    ) * cos_term

    return -(mu*(ω**2)*G/(r*c**4)) * np.sum(HPLUS)

def hcross(a, e, t, nmax=100):
    n = np.arange(1, nmax+1)
    ω = np.sqrt(G*(m1+m2)/a**3)

    A_n = A(a, nmax, e)
    B_n = B(a, nmax, e)
    C_n = C(a, nmax, e)

    sin_term = np.sin(n * ω * t)
    HCROSS = (
        np.cos(2*φ)*np.cos(I)*C_n
        + (A_n - B_n)*np.sin(2*φ)*np.sin(I)
    ) * sin_term

    return -(mu*(ω**2)*G/(r*c**4)) * np.sum(HCROSS)

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

w = get_window('hann', N+1)

# Compute FFTs (normalized)
X = np.fft.rfft(hcr * w) / np.sum(w)
Y = np.fft.rfft(hpl * w) / np.sum(w)

# Frequency axis (properly scaled)
λ = np.fft.rfftfreq(N+1, d=h)

plt.figure(figsize=(8,5))
plt.plot(t, hcr), label='Hx (cross)', color='blue')
plt.plot(t, hpl), label='Hp (plus)', color='red')
plt.legend()
plt.xlabel("Time (0.001 years")
plt.ylabel("Strain amplitude")
plt.title("Gravitational Wave Polarizations Over Frequency")
plt.grid(True)
plt.tight_layout()
plt.show()
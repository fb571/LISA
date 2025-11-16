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
# Units of 0.01AU = 100s = 1 solar mass = 1

ω0 = 0.03795*(2*np.pi) #Initial Frequency
e0 = 0.3 #Initial Eccentricity
m1 = 0.32 #Mass of Primary
m2 = 0.37 #Mass of Secondary
r = 2784574900 #Distance of Binary from Observer
I = np.pi #Angle of Inclination
φ = 0 #Angle of Pericentre

G = 0.0003965 #Gravitational Constant
c = 20.04 #Speed of Light
a0 = (G*(m1+m2)/ω0**2)**(1/3) #Initial Semimajor Axis

#Number of timesteps and the time interval value
N = 1000
h = 1

y0 = np.array([a0,e0])

def f(y):
    a = y[0]
    e = y[1]
    coeff = (G**3) * m1 * m2 * (m1 + m2) / (c**5)
    aprime = (-64/5 * coeff *(1 + (73/24)*e**2 + (37/96)*e**4)/(a**3 * (1 - e**2)**(7/2)))
    eprime = (-304/15 * coeff * e *(1 + (121/304)*e**2) /(a**4 * (1 - e**2)**(5/2)))
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

def hplus(a, e, t, m1, m2, r, I, φ, nmax=100):
    mu = (m1*m2)/(m1+m2)   
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

def hcross(a, e, t, m1, m2, r, I, φ, nmax=100):
    mu = (m1*m2)/(m1+m2)   
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
    hcr[i] = hcross(a[i], e[i], t[i], m1, m2, r, I, φ)
    hpl[i] = hplus(a[i], e[i], t[i], m1, m2, r, I, φ)

noise = np.random.normal(0, 1e-22, N+1)

w = get_window('hann', N+1)

# Compute FFTs (normalized)
X = np.fft.rfft(hcr*w+noise) / np.sum(w)
Y = np.fft.rfft(hpl*w+noise) / np.sum(w)

# Frequency axis (properly scaled)
λ = np.fft.rfftfreq(N+1, d=h)

plt.figure(figsize=(8,5))
plt.plot(λ, np.abs(X), label='Hx (cross) Fourier transform strain amplitude', color='blue')
plt.plot(λ, np.abs(Y), label='Hp (plus) Fourier transform strain amplitude', color='red')
plt.legend()
plt.xlabel("Frequency (cHz)")
plt.ylabel("Strain amplitude (|FFT|)")
plt.title("Gravitational Wave Polarizations Over Frequency")
plt.grid(True)
plt.tight_layout()
plt.show()

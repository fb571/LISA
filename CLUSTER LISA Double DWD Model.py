import os
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false"
os.environ["OMP_NUM_THREADS"] = "1"
import matplotlib
matplotlib.use('Agg')
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

import bilby    
import numpy as np
import matplotlib.pyplot as plt
from scipy import special
from scipy.signal import get_window
import jax
import jax.numpy as jnp
from functools import partial
import bilby.core.sampler.base_sampler as bs

csi = 3e8
L = 2.5e9

@jax.jit
def Fplus(theta,phi,psi):
    return jnp.sqrt(3)/2 * (0.5*(1+jnp.cos(theta)**2)*jnp.cos(2*phi)*jnp.cos(2*psi) - jnp.cos(theta)*jnp.sin(2*phi)*jnp.sin(2*psi))

@jax.jit
def Fcross(theta,phi,psi):
    return jnp.sqrt(3)/2 * (0.5*(1+jnp.cos(theta)**2)*jnp.cos(2*phi)*jnp.sin(2*psi) + jnp.cos(theta)*jnp.sin(2*phi)*jnp.cos(2*psi))

jax.config.update("jax_enable_x64", True)

# ============================================================
# 1. Time grid
# ============================================================

np.random.seed(123)
  
label = "freq_peak_with_noise"
outdir = "LISADoubleModel1"
bilby.utils.check_directory_exists_and_if_not_mkdir(outdir)

N = 10000
dt = 10
t = jnp.arange(N) * dt

# ============================================================
# 2. True signal parameters
# ============================================================

theta = np.pi/3
phi = np.pi/4
psi = 0

true_f0 = 0.00311
ω = 2 * true_f0 * np.pi
true_e = 0
m1 = 0.5 * 2e30
m2 = 0.5 * 2e30
r = 1.5e19
true_CI = 1

G = 6.67e-11
c = 3e8

a0 = (G * (m1 + m2) / ω**2) ** (1/3)
mu = (m1 * m2) / (m1 + m2)
prefactor = ω**2 * a0**2 * (mu * G) / (r * c**4)

true_A = prefactor
true_peri = 0

# ============================================================
# 3. Allocation-free time-domain signal model
# ============================================================

@jax.jit
def gw_circular_time_domain_jax(t, A, f0, CI, peri):
    omega = 2.0 * jnp.pi * f0
    pref = -2 * A
    
    phase = 2 * omega * t - 2 * peri

    hp = (1 + CI**2) * jnp.cos(phase)

    return pref * hp

h_true = gw_circular_time_domain_jax(
    t,
    true_A,
    true_f0,
    true_CI,
    true_peri,
) + gw_circular_time_domain_jax(
    t,
    0.3*true_A,
    true_f0+2.5e-5,
    true_CI,
    true_peri,
)

response = h_true
window = get_window("hann", N)
window_jax = jnp.asarray(window)
W = jnp.sum(window**2) / N
norm = jnp.sqrt(W)

h_fft_true = dt * jnp.fft.rfft(response * window_jax) / norm
freqs = jnp.fft.rfftfreq(N, d=dt)

@jax.jit
def omegaL(f):
    return 2*jnp.pi*f*L/csi

@jax.jit
def Sacc(f):
    return (3e-15/(2*jnp.pi*f*csi))**2 * (1+(0.4e-3/f)**2)*(1+(f/(8e-3))**4)

@jax.jit
def Soms(f):
    return (15e-12)**2 * (2*jnp.pi*f/csi)**2 * (1+(2e-3/f)**4)

@jax.jit
def SnX20(f):
    omegal = omegaL(f)
    return 64*(jnp.sin(omegal))**2 * (jnp.sin(2*omegal))**2 * (Soms(f)+(3+jnp.cos(2*omegal))*Sacc(f))

@jax.jit
def ShX20(f):
    omegal = omegaL(f)
    return (20/3) * (1+0.6*(omegal)**2) * (SnX20(f))/(((4*omegal)**2)*((jnp.sin(omegal))**2)*(2*jnp.sin(2*omegal))**2)

@jax.jit
def Fplus(theta,phi,psi):
    return jnp.sqrt(3)/2 * (0.5*(1+jnp.cos(theta)**2)*jnp.cos(2*phi)*jnp.cos(2*psi) - jnp.cos(theta)*jnp.sin(2*phi)*jnp.sin(2*psi))

@jax.jit
def Fcross(theta,phi,psi):
    return jnp.sqrt(3)/2 * (0.5*(1+jnp.cos(theta)**2)*jnp.cos(2*phi)*jnp.sin(2*psi) + jnp.cos(theta)*jnp.sin(2*phi)*jnp.cos(2*psi))

b = jnp.zeros(len(freqs))
b = b.at[1:].set(jnp.sqrt(ShX20(freqs[1:]) * N * dt / 4))

noise_fft = jnp.asarray(np.random.normal(0, b) + 1j * np.random.normal(0, b))
data_fft = h_fft_true + noise_fft

true_R = 2*true_A * (1+true_CI**2)

# ============================================================
# 4. Plotting
# ============================================================

plt.figure(figsize=(10, 6))
plt.plot(t[:1000], np.array(response)[:1000], label='GW Signal', color='blue')
plt.xlabel('Time (s)')
plt.ylabel('Strain')
plt.title('Time-Domain GW Signal')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("LISADoubleModel1/time_domain_signal.png")

plt.figure(figsize=(10, 6))
plt.plot(freqs[200:], jnp.abs(data_fft[200:]), label='Data FFT', color='green')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.title('FFT of GW Signal with Noise')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("LISADoubleModel1/frequency_domain_signal.png")

plt.figure(figsize=(10, 6))
plt.plot(freqs[1:], jnp.abs(h_fft_true[1:]/(jnp.sqrt(2)*b[1:])), label='SNR', color='green')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.title('SNR of GW Signal with Noise')
plt.legend()
plt.minorticks_on()
plt.grid(True, which='major', linestyle='-', linewidth=0.8, alpha=0.7)
plt.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.4)
plt.tight_layout()
plt.savefig("LISADoubleModel1/snr_plot.png")

# ============================================================
# 5. Likelihood (allocation-free)
# ============================================================

noise_magnitude = b

freq_mask = (freqs > 0.005) & (freqs < 0.02)
freq_indices = np.where(freq_mask)[0]  # static Python array
freq_indices_jax = jnp.asarray(freq_indices)

noise_slice = jnp.take(noise_magnitude, freq_indices_jax)
noise2 = noise_slice * noise_slice
log_norm = jnp.sum(jnp.log(2 * jnp.pi * noise2))

@jax.jit
def log_likelihood_fn(R1, R2, f01, f02, phi1, phi2, t, data_fft, dt,
                      window_jax, norm, noise2, freq_indices):
    h_td1 = R1*jnp.sin(4*jnp.pi*f01*t - phi1)
    h_td2 = R2*jnp.sin(4*jnp.pi*f02*t - phi2)
    h_td = h_td1 + h_td2
    h_fft = dt * jnp.fft.rfft(h_td * window_jax) / norm
    diff = jnp.take(data_fft, freq_indices) - jnp.take(h_fft, freq_indices)
    return -0.5 * jnp.sum((diff.real**2 + diff.imag**2) / noise2) \
           -0.5 * log_norm

log_likelihood_jit = jax.jit(
    partial(
        log_likelihood_fn,
        t=t,
        data_fft=data_fft,
        dt=dt,
        window_jax=window_jax,
        norm=norm,
        noise2=noise2,
        freq_indices=freq_indices_jax
    )
)

class FFTGWLikelihood(bilby.Likelihood):
    def __init__(self, t, data_fft, dt):
        super().__init__(parameters=dict(R1=None, R2=None, f01=None, f02=None, phi1=None, phi2=None))
        self.t = t
        self.data_fft = data_fft
        self.dt = dt

    def log_likelihood(self):
        R1   = jnp.asarray(self.parameters["R1"])
        R2   = jnp.asarray(self.parameters["R2"])
        f01  = jnp.asarray(self.parameters["f01"])
        f02 = jnp.asarray(self.parameters["f02"])
        phi1 = jnp.asarray(self.parameters["phi1"])
        phi2 = jnp.asarray(self.parameters["phi2"])
        return log_likelihood_jit(R1, R2, f01, f02, phi1, phi2)

# ============================================================
# 6. Priors
# ============================================================

def convert_f01_f02_to_df0(parameters):
    """
    Function to convert between sampled parameters and constraint parameter.

    Parameters
    ----------
    parameters: dict
        Dictionary containing sampled parameter values, 'f01', 'f02'.

    Returns
    -------
    dict: Dictionary with constraint parameter 'df0' added.
    """
    converted_parameters = parameters.copy()
    converted_parameters['df0'] = parameters['f02'] - parameters['f01']
    return converted_parameters

from bilby.core.prior import PriorDict, Uniform, Constraint
priors = PriorDict(conversion_function=convert_f01_f02_to_df0)
priors["R1"] = Uniform(0, 3e-21, name="R1", latex_label=r"$R1$")
priors["R2"] = Uniform(0, 3e-21, name="R2", latex_label=r"$R2$")
priors["f01"] = Uniform(0.003, 0.005, name="f_01", latex_label=r"$f_01\ \mathrm{[Hz]}$")
priors["f02"] = Uniform(0.003, 0.005, name="f_02", latex_label=r"$f_02\ \mathrm{[Hz]}$")
priors["df0"] = Constraint(minimum=3/(N*dt), maximum=0.002, name="df0", latex_label=r"\Delta f_0\ \mathrm{[Hz]}")
priors["phi1"] = Uniform(0, 2*np.pi, name="phi1", latex_label=r"\phi_1")
priors["phi2"] = Uniform(0, 2*np.pi, name="phi2", latex_label=r"\phi_2")

# ============================================================
# 9. Run bilby
# ============================================================

likelihood = FFTGWLikelihood(t, data_fft, dt)

if __name__ == "__main__":
    result = bilby.run_sampler(
    likelihood=likelihood,
    priors=priors,
    sampler="dynesty",
    outdir="LISADoubleModel1",
    label="fft_gw_freq_noise",
    nlive=200,
    npool=16,
    sample='slice',
    )

    param_order = ["R1", "R2", "f01", "f02", "phi1", "phi2"]

    result.plot_corner(
        parameters=param_order,
        bins=40,
        smooth=1.0,
        quantiles=[0.16, 0.84],
    )

    result.save_to_file()

    posterior = result.posterior
    idx_mle = posterior["log_likelihood"].idxmax()
    mle_parameters = posterior.loc[idx_mle]

    R1_mle = mle_parameters["R1"]
    R2_mle = mle_parameters["R2"]
    f01_mle = mle_parameters["f01"]
    f02_mle = mle_parameters["f02"]
    phi1_mle = mle_parameters["phi1"]
    phi2_mle = mle_parameters["phi2"]

    h_td1_mle = R1_mle*jnp.sin(4*jnp.pi*f01_mle*t - phi1_mle)
    h_td2_mle = R2_mle*jnp.sin(4*jnp.pi*f02_mle*t - phi2_mle)
    h_td_mle = h_td1_mle + h_td2_mle
    h_fft_mle = dt * jnp.fft.rfft(h_td_mle * window_jax) / norm

    plt.figure(figsize=(10, 6))
    plt.plot(freqs[200:], jnp.abs(h_fft_mle[200:]), label='Data FFT', color='green')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.title('FFT of GW Signal of MLE Parameters')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("LISADoubleModel1/frequency_domain_signal_mle.png")
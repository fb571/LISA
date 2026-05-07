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
outdir = "LISATripleFit3"
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
true_e = 0.25
m1 = 0.5 * 2e30
m2 = 0.5 * 2e30
r = 1.5e19
true_CI = np.cos(np.pi/4)

G = 6.67e-11
c = 3e8

a0 = (G * (m1 + m2) / ω**2) ** (1/3)
mu = (m1 * m2) / (m1 + m2)
prefactor = ω**2 * a0**2 * (mu * G) / (r * c**4)

true_A = 0.2*prefactor
true_peri = np.pi

nmax = 10
n_vals = jnp.arange(1, nmax + 1)

e_min, e_max, de = 0, 1, 0.001
e_grid = np.arange(e_min, e_max + de/2, de)
Ne = len(e_grid)
e_grid_jax = jnp.asarray(e_grid)

# --- build using SciPy (once) ---
J_table_3d = np.empty((Ne, nmax, 4), dtype=np.float64)

for ie, e in enumerate(e_grid):
    for j, nj in enumerate(n_vals):
        x = nj * e
        J_table_3d[ie, j, 0] = special.jv(nj - 2, x)
        J_table_3d[ie, j, 1] = special.jv(nj - 1, x)
        J_table_3d[ie, j, 2] = special.jv(nj + 1, x)
        J_table_3d[ie, j, 3] = special.jv(nj + 2, x)

# --- convert ONCE to JAX ---
J_table_3d_jax = jnp.asarray(J_table_3d)

@jax.jit
def get_J_slice_interp_jax(e):
    x = (e - e_min) / de
    i0 = jnp.floor(x).astype(jnp.int32)

    i0 = jnp.clip(i0, 0, J_table_3d_jax.shape[0] - 2)
    w = x - i0

    return (1.0 - w) * J_table_3d_jax[i0] + w * J_table_3d_jax[i0 + 1]

# ============================================================
# 3. Allocation-free time-domain signal model
# ============================================================

@jax.jit
def gw_time_domain_jax(t, A, f0, e, J_slice, CI, peri, n_vals):
    omega = 2.0 * jnp.pi * f0
    pref = -A

    Cperi = jnp.cos(peri)
    Speri = jnp.sin(peri)
    S2peri = jnp.sin(2.0*peri)

    Jm2 = J_slice[:, 0]
    Jm1 = J_slice[:, 1]
    Jp1 = J_slice[:, 2]
    Jp2 = J_slice[:, 3]
    
    nAn = n_vals * (Jm2 - Jp2 - 2.0 * e * (Jm1 - Jp1))
    nBn = n_vals * (1.0 - e * e) * (Jp2 - Jm2)
    nCn = n_vals * jnp.sqrt(1.0 - e * e) * (Jp2 + Jm2 - e * (Jp1 + Jm1))

    phase = omega * t[:, None] * n_vals[None, :]

    A_term = (Cperi**2 - Speri**2 * CI**2)
    B_term = (Speri**2 - Cperi**2 * CI**2)
    C_term = S2peri * (1.0 + CI**2)

    hp = (
        (nAn[None, :] * A_term + nBn[None, :] * B_term) * jnp.cos(phase)
        - nCn[None, :] * C_term * jnp.sin(phase)
    )

    return pref * jnp.sum(hp, axis=1)

J_slice = get_J_slice_interp_jax(true_e)

h_true = gw_time_domain_jax(
    t,
    true_A,
    true_f0,
    true_e,
    J_slice,
    true_CI,
    true_peri,
    n_vals
)

response_true = h_true
window = get_window("hann", N)
window_jax = jnp.asarray(window)
W = jnp.sum(window_jax**2) / N
norm = jnp.sqrt(W)

h_fft_true = dt * jnp.fft.rfft(response_true * window_jax) / norm
freqs = jnp.fft.rfftfreq(N, d=dt)

csi = 3e8
L = 2.5e9

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

# ============================================================
# 4. Plotting
# ============================================================

plt.figure(figsize=(10, 6))
plt.plot(t, np.array(response_true), label='GW Signal', color='blue')
plt.xlabel('Time (s)')
plt.ylabel('Strain')
plt.title('Time-Domain GW Signal')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("LISATripleFit3/time_domain_signal.png")

plt.figure(figsize=(10, 6))
plt.plot(freqs[200:], jnp.abs(data_fft[200:]), label='Data FFT', color='green')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.title('FFT of GW Signal with Noise')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("LISATripleFit3/frequency_domain_signal.png")

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
plt.savefig("LISATripleFit3/snr_plot.png")

# ============================================================
# 5. Likelihood (allocation-free)
# ============================================================

noise_magnitude = b

freq_mask = (freqs > 0.005) & (freqs < 0.02)
freq_indices = jnp.where(freq_mask)[0]

noise_slice = jnp.take(noise_magnitude, freq_indices)
noise2 = noise_slice * noise_slice
log_norm = jnp.sum(jnp.log(2 * jnp.pi * noise2))

@jax.jit
def log_likelihood_fn(R1, R2, R3, f01, f02, f03, phi1, phi2, phi3, t, data_fft, dt,
                      window_jax, norm, noise2, freq_indices):
    h_td1 = -2*R1*jnp.cos(4*jnp.pi*f01*t + 2*phi1)
    h_td2 = -2*R2*jnp.sin(4*jnp.pi*f02*t + 2*phi2)
    h_td3 = -2*R3*jnp.sin(4*jnp.pi*f03*t + 2*phi3)
    h_fft = dt * jnp.fft.rfft((h_td1+h_td2+h_td3) * window_jax) / norm
    diff = jnp.take(data_fft - h_fft, freq_indices)
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
        freq_indices=freq_indices
    )
)

class FFTGWLikelihood(bilby.Likelihood):
    def __init__(self, t, data_fft, dt):
        super().__init__(parameters=dict(R1=None, R2=None, R3=None, f01=None, f02=None, f03=None, phi1=None, phi2=None, phi3=None))
        self.t = t
        self.data_fft = data_fft
        self.dt = dt

    def log_likelihood(self):
        return log_likelihood_jit(
        self.parameters["R1"],
        self.parameters["R2"],
        self.parameters["R3"],
        self.parameters["f01"],
        self.parameters["f02"],
        self.parameters["f03"],
        self.parameters["phi1"],
        self.parameters["phi2"],
        self.parameters["phi3"]
    )

# ============================================================
# 6. Priors
# ============================================================

def convert_to_df0(parameters):
    """
    Function to convert between sampled parameters and constraint parameter.

    Parameters
    ----------
    parameters: dict
        Dictionary containing sampled parameter values, 'f01', 'f02', 'f03'.

    Returns
    -------
    dict: Dictionary with constraint parameter 'df012', 'df023' added.
    """
    converted_parameters = parameters.copy()
    converted_parameters['df012'] = parameters['f02'] - parameters['f01']
    converted_parameters['df023'] = parameters['f03'] - parameters['f02']
    return converted_parameters

from bilby.core.prior import PriorDict, Uniform, Constraint
priors = PriorDict(conversion_function=convert_to_df0)
priors["R1"] = Uniform(0, 2e-21, name="R1", latex_label=r"$R1$")
priors["R2"] = Uniform(0, 2e-21, name="R2", latex_label=r"$R2$")
priors["R3"] = Uniform(0, 2e-21, name="R3", latex_label=r"$R3$")
priors["f01"] = Uniform(0.003, 0.01, name="f_01", latex_label=r"$f_01\ \mathrm{[Hz]}$")
priors["f02"] = Uniform(0.003, 0.01, name="f_02", latex_label=r"$f_02\ \mathrm{[Hz]}$")
priors["f03"] = Uniform(0.003, 0.01, name="f_03", latex_label=r"$f_03\ \mathrm{[Hz]}$")
priors["df012"] = Constraint(minimum=3/(N*dt), maximum=0.002, name="df012", latex_label=r"\Delta f_0\ \mathrm{[Hz]}")
priors["df023"] = Constraint(minimum=3/(N*dt), maximum=0.002, name="df023", latex_label=r"\Delta f_0\ \mathrm{[Hz]}")
priors["phi1"] = Uniform(0, 2*np.pi, name="phi1", latex_label=r"\phi_1")
priors["phi2"] = Uniform(0, 2*np.pi, name="phi2", latex_label=r"\phi_2")
priors["phi3"] = Uniform(0, 2*np.pi, name="phi3", latex_label=r"\phi_3")

# ============================================================
# 9. Run bilby
# ============================================================

likelihood = FFTGWLikelihood(t, data_fft, dt)

if __name__ == "__main__":
    result = bilby.run_sampler(
    likelihood=likelihood,
    priors=priors,
    sampler="dynesty",
    outdir="LISATripleFit3",
    label="fft_gw_freq_noise",
    nlive=1800,
    npool=24,
    sample='slice',
    )


    param_order = ["R1", "R2", "R3", "f01", "f02", "f03", "phi1", "phi2", "phi3"]

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
    R3_mle = mle_parameters["R3"]
    f01_mle = mle_parameters["f01"]
    f02_mle = mle_parameters["f02"]
    f03_mle = mle_parameters["f03"]
    phi1_mle = mle_parameters["phi1"]
    phi2_mle = mle_parameters["phi2"]
    phi3_mle = mle_parameters["phi3"]

    h_td1_mle = -2*R1_mle*jnp.sin(4*jnp.pi*f01_mle*t + 2*phi1_mle)
    h_td2_mle = -2*R2_mle*jnp.sin(4*jnp.pi*f02_mle*t + 2*phi2_mle)
    h_td3_mle = -2*R3_mle*jnp.sin(4*jnp.pi*f03_mle*t + 2*phi3_mle)
    h_td_mle = h_td1_mle + h_td2_mle + h_td3_mle
    h_fft_mle = dt * jnp.fft.rfft(h_td_mle * window_jax) / norm

    plt.figure(figsize=(10, 6))
    plt.plot(freqs[200:], jnp.abs(h_fft_mle[200:]), label='Data FFT', color='green')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.title('FFT of GW Signal of MLE Parameters')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("LISATripleFit3/frequency_domain_mle_signal.png")



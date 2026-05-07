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

def omegaL(f):
    return 2*np.pi*f*L/csi

def sacc(f): #given by equation 13
    return (3e-15/(2*np.pi*f*csi))**2 * (1+(0.4e-3/f)**2)*(1+(f/(8e-3))**4)

def soms(f): #given by equation 10
    return (15e-12)**2 * (2*np.pi*f/csi)**2 * (1+(2e-3/f)**4)

def SnX20(f): #given by equation 20
    omegal = omegaL(f)
    return 64*(np.sin(omegal))**2 * (np.sin(2*omegal))**2 * (soms(f)+(3+np.cos(2*omegal))*sacc(f))

def ShX20(f): #given by equation 57
    omegal = omegaL(f)
    return (20/3) * (1+0.6*(omegal)**2) * (SnX20(f))/(((4*omegal)**2)*((np.sin(omegal))**2)*(2*np.sin(2*omegal))**2)

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

np.random.seed(789)
  
label = "freq_peak_with_noise"
outdir = "LISADoubleCircFitEccentricModel,nlive=2000,SNRecc=16"
bilby.utils.check_directory_exists_and_if_not_mkdir(outdir)

N = 10000
dt = 10
t = jnp.arange(N) * dt

# ============================================================
# 2. True signal parameters
# ============================================================

true_f01 = 0.004
ω1 = 2 * true_f01 * np.pi
true_e1 = 0.4
m11 = 0.5 * 2e30
m21 = 0.5 * 2e30
r1 = 1.5e19
true_CI1 = np.cos(np.pi/4)

true_f02 = 0.723089457293 * true_f01
ω2 = 2 * true_f02 * np.pi
true_e2 = 0
m12 = 0.5 * 2e30
m22 = 0.5 * 2e30
r2 = 1.5e19
true_CI2 = np.cos(np.pi/3)

G = 6.67e-11
c = 3e8

a01 = (G * (m11 + m21) / ω1**2) ** (1/3)
mu1 = (m11 * m21) / (m11 + m21)

prefactor1 = ω1**2 * a01**2 * (mu1 * G) / (r1 * c**4)

true_Amp1 = (16/34.51281763790965)*prefactor1
true_peri1 = np.pi

a02 = (G * (m12 + m22) / ω2**2) ** (1/3)
mu2 = (m12 * m22) / (m12 + m22)
prefactor2 = (2 * np.pi * true_f02)**2 * a02**2 * (mu2 * G) / (r2 * c**4)

true_Amp2 = (0/23.188323113413922)*prefactor2
true_peri2 = np.pi/2

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
    C2peri = jnp.cos(2.0*peri)
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

J_slice_true1 = get_J_slice_interp_jax(true_e1)
J_slice_true2 = get_J_slice_interp_jax(true_e2)

h_true = gw_time_domain_jax(
    t,
    true_Amp1,
    true_f01,
    true_e1,
    J_slice_true1,
    true_CI1,
    true_peri1,
    n_vals
) + gw_time_domain_jax(
    t,
    true_Amp2,
    true_f02,
    true_e2,
    J_slice_true2,
    true_CI2,
    true_peri2,
    n_vals
)

response_true = h_true
window = get_window("hann", N)
window_jax = jnp.asarray(window)
W = np.sum(window**2) / N
norm = np.sqrt(W)

h_fft_true = dt * jnp.fft.rfft(response_true * window_jax) / norm
freqs = np.fft.rfftfreq(N, d=dt)

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

@jax.jit
def compute_snr(h_fft, ShX20, freqs):
    return jnp.sqrt(4*jnp.abs(h_fft[1:])**2 / ShX20(freqs[1:]))

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
plt.savefig("LISADoubleCircFitEccentricModel,nlive=2000,SNRecc=16/time_domain_signal.pdf")

plt.figure(figsize=(10, 6))
plt.plot(freqs[200:], jnp.abs(h_fft_true[200:]), label='Data FFT', color='green')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.title('FFT of GW Signal with Noise')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("LISADoubleCircFitEccentricModel,nlive=2000,SNRecc=16/pure_frequency_domain_signal.pdf")

plt.figure(figsize=(10, 6))
plt.plot(freqs[200:], jnp.abs(data_fft[200:]), label='Data FFT', color='green')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.title('FFT of GW Signal with Noise')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("LISADoubleCircFitEccentricModel,nlive=2000,SNRecc=16/frequency_domain_signal.pdf")

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
plt.savefig("LISADoubleCircFitEccentricModel,nlive=2000,SNRecc=16/snr_plot.pdf")

plt.figure(figsize=(10, 6))
plt.plot(freqs[1:], jnp.abs(data_fft[1:]/(jnp.sqrt(2)*b[1:])), label='SNR', color='green')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.title('SNR of GW Signal with Noise')
plt.legend()
plt.minorticks_on()
plt.grid(True, which='major', linestyle='-', linewidth=0.8, alpha=0.7)
plt.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.4)
plt.tight_layout()
plt.savefig("LISADoubleCircFitEccentricModel,nlive=2000,SNRecc=16/snr_plot_data.pdf")

# ============================================================
# 5. Likelihood (allocation-free)
# ============================================================

noise_magnitude = b

freq_mask = (freqs > 0.002) & (freqs < 0.02)
freq_indices = jnp.where(freq_mask)[0]

noise_slice = jnp.take(noise_magnitude, freq_indices)
noise2 = noise_slice * noise_slice
log_norm = jnp.sum(jnp.log(2 * jnp.pi * noise2))

@jax.jit
def log_likelihood_fn(R1, f01, phi01, R2, f02, phi02, t, data_fft, dt,
                      window_jax, norm, noise2, freq_indices, n_vals):

    htd = - 2*R1 * jnp.cos(4*jnp.pi*f01*t + 2*phi01) - 2*R2 * jnp.cos(4*jnp.pi*f02*t + 2*phi02)

    h_fft = dt * jnp.fft.rfft(htd * window_jax) / norm
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
        freq_indices=freq_indices,
        n_vals=n_vals
    )
)

class FFTGWLikelihood(bilby.Likelihood):
    def __init__(self, t, data_fft, dt):
        super().__init__(parameters=dict(R1=None, f01=None, phi01=None, R2=None, f02=None, phi02=None))
        self.t = t
        self.data_fft = data_fft
        self.dt = dt

    def log_likelihood(self):
        R1   = jnp.asarray(self.parameters["R1"])
        f01  = jnp.asarray(self.parameters["f01"])
        phi01 = jnp.asarray(self.parameters["phi01"])

        R2   = jnp.asarray(self.parameters["R2"])
        f02  = jnp.asarray(self.parameters["f02"])
        phi02 = jnp.asarray(self.parameters["phi02"])

        return log_likelihood_jit(R1, f01, phi01, R2, f02, phi02)

# ============================================================
# 6. Priors
# ============================================================

def convert_f01_f02_to_df012(parameters):
    """
    Function to convert between sampled parameters and constraint parameter.

    Parameters
    ----------
    parameters: dict
        Dictionary containing sampled parameter values, 'f01', 'f02'.

    Returns
    -------
    dict: Dictionary with constraint parameters 'df012' added.
    """
    converted_parameters = parameters.copy()
    converted_parameters['df012'] = parameters['f02'] - parameters['f01']
    return converted_parameters

from bilby.core.prior import PriorDict, Uniform, Constraint
priors = PriorDict(conversion_function=convert_f01_f02_to_df012)
priors["R1"] = Uniform(0, 2e-21, name="R1", latex_label=r"$R_1$")
priors["f01"] = Uniform(0.001, 0.01, name="f01", latex_label=r"$f_{0,1}\ \mathrm{[Hz]}$")
priors["phi01"] = Uniform(0, np.pi, name="phi01", latex_label=r"$\phi_{0,1}$")
priors["R2"] = Uniform(0, 2e-21, name="R2", latex_label=r"$R_2$")
priors["f02"] = Uniform(0.001, 0.01, name="f02", latex_label=r"$f_{0,2}\ \mathrm{[Hz]}$")
priors["phi02"] = Uniform(0, np.pi, name="phi02", latex_label=r"$\phi_{0,2}$")

priors["df012"] = Constraint(minimum=3/(N*dt), maximum=0.01, name="df012", latex_label=r"\Delta f_0\ \mathrm{[Hz]}")

# ============================================================
# 9. Run bilby
# ============================================================

likelihood = FFTGWLikelihood(t, data_fft, dt)

param_order = ["R1", "f01", "phi01", "R2", "f02", "phi02"]

if __name__ == "__main__":
    result = bilby.core.result.read_in_result(
    outdir="LISADoubleCircFitEccentricModel,nlive=2000,SNRecc=16",
    label="fft_gw_freq_noise"
    )

    print(result.log_evidence)
    print(result.log_evidence_err)

    param_order = ["R1", "f01", "phi01", "R2", "f02", "phi02"]

    result.plot_corner(
    parameters=param_order,
    bins=40,
    smooth=1.0,
    title_quantiles=[0.16, 0.5, 0.84],
    show_titles=True,
    title_kwargs={"fontsize": 12},
)

    result.save_to_file()

    posterior = result.posterior
    idx_mle = posterior["log_likelihood"].idxmax()
    mle_parameters = posterior.loc[idx_mle]

    R1_mle = mle_parameters["R1"]
    f01_mle = mle_parameters["f01"]
    phi01_mle = mle_parameters["phi01"]

    R2_mle = mle_parameters["R2"]
    f02_mle = mle_parameters["f02"]
    phi02_mle = mle_parameters["phi02"]

    @jax.jit
    def gw_time_domain_jax_2(t, R1, f01, phi01, R2, f02, phi02, n_vals):
        return - 2 * R1 * jnp.cos(4*jnp.pi*f01*t + 2*phi01) - 2 * R2 * jnp.cos(4*jnp.pi*f02*t + 2*phi02)

    h_td_mle = gw_time_domain_jax_2(t, R1_mle, f01_mle, phi01_mle, R2_mle, f02_mle, phi02_mle, n_vals)
    h_fft_mle = dt * jnp.fft.rfft(h_td_mle * window_jax) / norm

    plt.figure(figsize=(10, 6))
    plt.plot(freqs, jnp.abs(h_fft_mle), label='Data FFT', color='green')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.title('FFT of GW Signal of MLE Parameters')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("LISADoubleCircFitEccentricModel,nlive=2000,SNRecc=16/frequency_domain_signal_mle.pdf")

# ============================================================
# 10. Posterior draws with per-frequency SNR masking
# ============================================================

    N_draws = 100

    posterior_samples = result.posterior.sample(N_draws).reset_index(drop=True)

    @jax.jit
    def compute_hfft_jax(R1, f01, phi01, R2, f02, phi02):
        h_fft = dt * jnp.fft.rfft(gw_time_domain_jax_2(t, R1, f01, phi01, R2, f02, phi02, n_vals) * window_jax) / norm
        return h_fft

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, row in posterior_samples.iterrows():
        h_fft_i = compute_hfft_jax(
            jnp.float64(row["R1"]),
            jnp.float64(row["f01"]),
            jnp.float64(row["phi01"]),
            jnp.float64(row["R2"]),
            jnp.float64(row["f02"]),
            jnp.float64(row["phi02"]),
        )

        amps = np.array(jnp.abs(h_fft_i))
        f01_i = float(row["f01"])
        f02_i = float(row["f02"])

        idx_circ_1 = np.argmin(np.abs(freqs - 2*f01_i))
        circ_amp_1 = amps[idx_circ_1]

        idx_circ_2 = np.argmin(np.abs(freqs - 2*f02_i))
        circ_amp_2 = amps[idx_circ_2]

        ax.scatter([2*f01_i], [circ_amp_1], s=8, alpha=0.3, color='tomato')
        ax.scatter([2*f02_i], [circ_amp_2], s=8, alpha=0.3, color='steelblue')

    ax.plot(freqs[200:], np.array(b[200:]), color='black', lw=1.5, label='Noise $\\sigma(f)$')
    ax.plot(freqs[200:], np.array(jnp.abs(h_fft_true[200:])), color='green', lw=1.5, label='SNR of True Signal')

    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Amplitude')
    ax.set_title('Posterior draws at harmonic frequencies')
    ax.grid(True)
    plt.tight_layout()
    plt.savefig("LISADoubleCircFitEccentricModel,nlive=2000,SNRecc=16/posterior_draws_harmonics.pdf", dpi=150)
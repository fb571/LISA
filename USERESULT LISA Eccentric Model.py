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

np.random.seed(123)
  
label = "freq_peak_with_noise"
outdir = "LISACombinedPeaks"
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

true_f0 = 0.004
ω = 2 * true_f0 * np.pi
true_e = 0
m1 = 0.5 * 2e30
m2 = 0.5 * 2e30
r = 1.5e19
true_CI = np.cos(np.pi/4)

G = 6.67e-11
c = 3e8

a0 = (G * (m1 + m2) / ω**2) ** (1/3)
mu = (m1 * m2) / (m1 + m2)
prefactor = ω**2 * a0**2 * (mu * G) / (r * c**4)

true_Amp = (40/34.51281763790965) * prefactor
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

@jax.jit
def gw_time_domain_jax_individual(t, A, f0, e, J_slice, CI, peri, n_vals):
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

    return pref * hp

J_slice = get_J_slice_interp_jax(true_e)

h_true = gw_time_domain_jax(
    t,
    true_Amp,
    true_f0,
    true_e,
    J_slice,
    true_CI,
    true_peri,
    n_vals
)

h_true_individual = gw_time_domain_jax_individual(
    t,
    true_Amp,
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
def compute_snr(h_fft, freqs):
    return jnp.sqrt(4*jnp.sum(jnp.abs(h_fft[1:])**2 / ShX20(freqs[1:]))/(N*dt))

print("SNR:", compute_snr(h_fft_true, freqs))

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
plt.ylim(-2e-21, 2e-21)
plt.tight_layout()
plt.savefig("LISACombinedPeaks/time_domain_signal.png")

plt.figure(figsize=(10, 6))
plt.plot(t, np.array(h_true_individual[:, 0]), label='GW Signal', color='blue')
plt.xlabel('Time (s)')
plt.ylabel('Strain')
plt.title('Time-Domain GW Signal')
plt.legend()
plt.grid(True)
plt.ylim(-2e-21, 2e-21)
plt.tight_layout()
plt.savefig("LISACombinedPeaks/time_domain_signal_1.png")

plt.figure(figsize=(10, 6))
plt.plot(t, np.array(h_true_individual[:, 1]), label='GW Signal', color='blue')
plt.xlabel('Time (s)')
plt.ylabel('Strain')
plt.title('Time-Domain GW Signal')
plt.legend()
plt.grid(True)
plt.ylim(-2e-21, 2e-21)
plt.tight_layout()
plt.savefig("LISACombinedPeaks/time_domain_signal_2.png")

plt.figure(figsize=(10, 6))
plt.plot(t, np.array(h_true_individual[:, 2]), label='GW Signal', color='blue')
plt.xlabel('Time (s)')
plt.ylabel('Strain')
plt.title('Time-Domain GW Signal')
plt.legend()
plt.grid(True)
plt.ylim(-2e-21, 2e-21)
plt.tight_layout()
plt.savefig("LISACombinedPeaks/time_domain_signal_3.png")

plt.figure(figsize=(10, 6))
plt.plot(freqs[300:], jnp.abs(h_fft_true[300:]), label='Data FFT', color='green')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.title('FFT of GW Signal with Noise')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("LISACombinedPeaks/pure_frequency_domain_signal.png")

plt.figure(figsize=(10, 6))
plt.plot(freqs[300:], jnp.abs(data_fft[300:]), label='Data FFT', color='green')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.title('FFT of GW Signal with Noise')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("LISACombinedPeaks/frequency_domain_signal.png")

plt.figure(figsize=(10, 6))
plt.plot(freqs[300:], jnp.abs(data_fft[300:]/(jnp.sqrt(2)*b[300:])), label='SNR', color='green')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.title('SNR of GW Signal with Noise')
plt.legend()
plt.minorticks_on()
plt.grid(True, which='major', linestyle='-', linewidth=0.8, alpha=0.7)
plt.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.4)
plt.tight_layout()
plt.savefig("LISACombinedPeaks/snr_true_signal.png")

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
plt.savefig("LISACombinedPeaks/snr_signal_with_noise.png")

# ============================================================
# 5. Likelihood (allocation-free)
# ============================================================

noise_magnitude = b

freq_mask = (freqs > 0.003) & (freqs < 0.02)
freq_indices = jnp.where(freq_mask)[0]

noise_slice = jnp.take(noise_magnitude, freq_indices)
noise2 = noise_slice * noise_slice
log_norm = jnp.sum(jnp.log(2 * jnp.pi * noise2))

@jax.jit
def log_likelihood_fn(A, B, C, f0, e, phi0, t, data_fft, dt,
                      window_jax, norm, noise2, freq_indices, n_vals):
    
    omega = 2.0 * jnp.pi * f0
    J_slice = get_J_slice_interp_jax(e)
    phase = omega * t[:, None] * n_vals[None, :] - phi0

    Jm2 = J_slice[:, 0]
    Jm1 = J_slice[:, 1]
    Jp1 = J_slice[:, 2]
    Jp2 = J_slice[:, 3]
    
    nAn = n_vals * (Jm2 - Jp2 - 2.0 * e * (Jm1 - Jp1))
    nBn = n_vals * (1.0 - e * e) * (Jp2 - Jm2)
    nCn = n_vals * jnp.sqrt(1.0 - e * e) * (Jp2 + Jm2 - e * (Jp1 + Jm1))

    hp = (
        (nAn[None, :] * A + nBn[None, :] * B) * jnp.cos(phase)
        - nCn[None, :] * C * jnp.sin(phase)
    )

    htd = jnp.sum(hp, axis=1)

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
        super().__init__(parameters=dict(A=None, B=None, C=None, f0=None, e=None, phi0=None))
        self.t = t
        self.data_fft = data_fft
        self.dt = dt

    def log_likelihood(self):
        A   = jnp.asarray(self.parameters["A"])
        B   = jnp.asarray(self.parameters["B"])
        C   = jnp.asarray(self.parameters["C"])
        f0  = jnp.asarray(self.parameters["f0"])
        e = jnp.asarray(self.parameters["e"])
        phi0 = jnp.asarray(self.parameters["phi0"])
        return log_likelihood_jit(A, B, C, f0, e, phi0)

# ============================================================
# 6. Priors
# ============================================================

from bilby.core.prior import PriorDict, Uniform, Constraint
priors = PriorDict()
priors["A"] = Uniform(-(4e-21), 4e-21, name="A", latex_label=r"$A$")
priors["B"] = Uniform(-(4e-21), 4e-21, name="B", latex_label=r"$B$")
priors["C"] = Uniform(-(4e-21), 4e-21, name="C", latex_label=r"$C$")
priors["f0"] = Uniform(0.001, 0.01, name="f_0", latex_label=r"$f_0\ \mathrm{[Hz]}$")
priors["e"] = Uniform(0, 1, name="e", latex_label=r"e")
priors["phi0"] = Uniform(0, np.pi, name="phi0", latex_label=r"\phi0")

# ============================================================
# 9. Run bilby
# ============================================================

likelihood = FFTGWLikelihood(t, data_fft, dt)

param_order = ["A", "B", "C", "f0", "e", "phi0"]

truths_dict = dict(
    A=true_Amp*(jnp.cos(true_peri)**2 - jnp.sin(true_peri)**2 * true_CI**2),
    B=true_Amp*(jnp.sin(true_peri)**2 - jnp.cos(true_peri)**2 * true_CI**2),
    C=true_Amp*jnp.sin(2*true_peri)*(1 + true_CI**2),
    f0=true_f0,
    e=true_e,
    phi0=true_peri
)

truths_list = [truths_dict[p] for p in param_order]
print(truths_list)

if __name__ == "__main__":
    result = bilby.core.result.read_in_result(
    outdir="LISACombinedPeaks",
    label="fft_gw_freq_noise"
    )

    param_order = ["A", "B", "C", "f0", "e", "phi0"]

    print(result.log_evidence)
    print(result.log_evidence_err)

    truths_dict = dict(
        A=true_Amp*(jnp.cos(true_peri)**2 - jnp.sin(true_peri)**2 * true_CI**2),
        B=true_Amp*(jnp.sin(true_peri)**2 - jnp.cos(true_peri)**2 * true_CI**2),
        C=true_Amp*jnp.sin(2*true_peri)*(1 + true_CI**2),
        f0=true_f0,
        e=true_e,
        phi0=true_peri
    )

    truths_list = [truths_dict[p] for p in param_order]
    print(truths_list)

    ranges_dict = dict(
        A=(-1e-21, 1e-21),
        B=(-1e-21, 1e-21),
        C=(-1e-21, 1e-21),
        f0=(0.003, 0.0032),
        e=(0.38, 0.42),
        phi0=(0, np.pi)
    )

    ranges_list = [ranges_dict[p] for p in param_order]

    result.plot_corner(
        parameters=param_order,
        bins=40,
        smooth=1.0,
        quantiles=[0.16, 0.84],
        truths=truths_list,
        ranges=ranges_list
    )

    result.save_to_file()

    param_order = ["A", "B", "C", "f0", "e", "phi0"]

    posterior = result.posterior
    idx_mle = posterior["log_likelihood"].idxmax()
    mle_parameters = posterior.loc[idx_mle]

    A_mle = mle_parameters["A"]
    B_mle = mle_parameters["B"]
    C_mle = mle_parameters["C"]
    f0_mle = mle_parameters["f0"]
    e_mle = mle_parameters["e"]
    phi0_mle = mle_parameters["phi0"]
    J_slice_mle = get_J_slice_interp_jax(e_mle)

    @jax.jit
    def gw_time_domain_jax_2(t, A, B, C, f0, e, J_slice, phi0, n_vals):
        omega = 2.0 * jnp.pi * f0
        phase = omega * t[:, None] * n_vals[None, :] - phi0

        Jm2 = J_slice[:, 0]
        Jm1 = J_slice[:, 1]
        Jp1 = J_slice[:, 2]
        Jp2 = J_slice[:, 3]
        
        nAn = n_vals * (Jm2 - Jp2 - 2.0 * e * (Jm1 - Jp1))
        nBn = n_vals * (1.0 - e * e) * (Jp2 - Jm2)
        nCn = n_vals * jnp.sqrt(1.0 - e * e) * (Jp2 + Jm2 - e * (Jp1 + Jm1))

        hp = (
            (nAn[None, :] * (-A) + nBn[None, :] * (-B)) * jnp.cos(phase)
            - nCn[None, :] * (-C) * jnp.sin(phase)
        )

        return jnp.sum(hp, axis=1)

    h_td_mle = gw_time_domain_jax_2(
        t,
        A_mle,
        B_mle,
        C_mle,
        f0_mle,
        e_mle,
        J_slice_mle,
        phi0_mle,
        n_vals
    )

    h_fft_mle = dt * jnp.fft.rfft(h_td_mle * window_jax) / norm

    plt.figure(figsize=(10, 6))
    plt.plot(freqs[200:], jnp.abs(h_fft_mle[200:]), label='Data FFT', color='green')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.title('FFT of GW Signal of MLE Parameters')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("LISACombinedPeaks/fft_mle_signal.png")

# ============================================================
# 10. Posterior draws with per-frequency SNR masking
# ============================================================

    N_draws = 100

    posterior_samples = result.posterior.sample(N_draws).reset_index(drop=True)

    @jax.jit
    def compute_hfft_jax(A, B, C, f0, e, phi0):
        h_fft = dt * jnp.fft.rfft((gw_time_domain_jax_2(t, A, B, C, f0, e, get_J_slice_interp_jax(e), phi0, n_vals)) * window_jax) / norm
        return h_fft

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, row in posterior_samples.iterrows():
        h_fft_i = compute_hfft_jax(
            jnp.float64(row["A"]),
            jnp.float64(row["B"]),
            jnp.float64(row["C"]),
            jnp.float64(row["f0"]),
            jnp.float64(row["e"]),
            jnp.float64(row["phi0"]),
        )

        amps = np.array(jnp.abs(h_fft_i))
        f01 = float(row["f0"])

        harmonic_freqs = []
        harmonic_amps  = []
        for n in range(1, nmax + 1):
            f_harm = n * f01
            idx = np.argmin(np.abs(freqs - f_harm))
            harmonic_freqs.append(f_harm)
            harmonic_amps.append(amps[idx])

        ax.scatter(harmonic_freqs, harmonic_amps, s=8, alpha=0.3, color='steelblue')

    ax.plot(freqs[200:], np.array(b[200:]), color='black', lw=1.5, label='Noise $\\sigma(f)$')
    ax.plot(freqs[200:], np.array(jnp.abs(h_fft_true[200:])), color='green', lw=1.5, label='SNR of True Signal')

    for n in range(1, nmax + 1):
        ax.axvline(n * true_f0, color='steelblue', lw=0.5, linestyle='--', alpha=0.5)

    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Amplitude')
    ax.set_title('Posterior draws at harmonic frequencies')
    ax.legend(['Noise', 'Circular $f_{02}$', 'Eccentric harmonics'])
    ax.grid(True)
    plt.tight_layout()
    plt.savefig("LISACombinedPeaks/posterior_draws_harmonics.png", dpi=150)
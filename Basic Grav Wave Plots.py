import bilby
import matplotlib.pyplot as plt
import numpy as np
from bilby.core.utils import random
from scipy.special import jn  # Bessel function of the first kind

random.seed(321)

# Setup
label = "linear_regression"
outdir = "outdir"
bilby.utils.check_directory_exists_and_if_not_mkdir(outdir)

def g(n, e):
    ne = n * e
    
    # Define f(n, e)
    f = jn(n - 2, ne) - 2 * e * jn(n - 1, ne) + (2 / n) * jn(n, ne) + 2 * e * jn(n + 1, ne) - jn(n + 2, ne)
    
    # Define f''(n, e)
    f_dd = jn(n - 2, ne) - 2 * jn(n, ne) + jn(n + 2, ne)
    
    # Compute g(n, e)
    g = (n**4 / 32.0) * (f**2 + (1 - e**2) * f_dd**2 + (4 / (3 * n**2)) * jn(n, ne)**2)
    
    return g

G = 1
c = 1

def P(n, a, e, m1, m2):
    return (32/5)*(G**4/c**5)*(m1**2*m2**2/a**5)*(m1+m2)*g(n, e)

m1 = 1
m2 = 1

# Signal Model
def model(n, a, e):
    return P(n, a, e, m1, m2)

# Injection Parameters
injection_parameters = dict(a=0.6, e=0.5, sigma=0.1)

# Gaussian Noise used in this eample

# Fake Data Generation
number_of_harmonics = 10
n = np.arange(1, number_of_harmonics +1, 1)
N = len(n)
data = model(n, injection_parameters["a"], injection_parameters["e"]) + random.rng.normal(0, injection_parameters["sigma"], N)

# Plotting the data for the true parameters against the noisy data
fig, ax = plt.subplots()
ax.plot(n, data, "o", label="data")
ax.plot(n, model(n, injection_parameters["a"], injection_parameters["e"]), "o", label="signal")
ax.set_xlabel("n")
ax.set_ylabel("y")
ax.legend()
fig.savefig("{}/{}_data.png".format(outdir, label))

# Gaussian Likelihood
class GaussianLikelihoodWithSigma(bilby.Likelihood):
    def __init__(self, n, data, model):
        super().__init__(parameters=dict(a=None, e=None, sigma=None))
        self.n = n
        self.data = data
        self.model = model

    def log_likelihood(self):
        a = self.parameters["a"]
        e = self.parameters["e"]
        sigma = self.parameters["sigma"]
        y_model = self.model(self.n, a, e)
        N = len(self.data)
        return -0.5 * np.sum(((self.data - y_model) / sigma) ** 2 + np.log(2 * np.pi * sigma ** 2))

likelihood = GaussianLikelihoodWithSigma(n, data, model)

# Uniform Priors
priors = dict()
priors["a"] = bilby.core.prior.Uniform(0.5, 0.7, "a")
priors["e"] = bilby.core.prior.Uniform(0.45, 0.55, "e")
priors["sigma"] = bilby.core.prior.Uniform(0.09, 0.11, "sigma")

# Running Sampler
result = bilby.run_sampler(
    likelihood=likelihood,
    priors=priors,
    sampler="bilby_mcmc",
    nlive=250,
    injection_parameters=injection_parameters,
    outdir=outdir,
    label=label,
)

# Finally plot a corner plot: all outputs are stored in outdir
result.plot_corner()
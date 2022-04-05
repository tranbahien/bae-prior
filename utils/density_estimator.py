import pickle

from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.neighbors import KernelDensity


class DensityEstimator:
    
    def __init__(self, method="gmm", n_components=None, max_iter=2000):
        self.method = method
        self.n_components = n_components
        self.model = None
        self.max_iter = max_iter

        self._initialize_model()

    def _initialize_model(self):

        if self.method.lower() == "gmm":
            self.model = GaussianMixture(
                n_components=self.n_components, covariance_type="full",
                max_iter=self.max_iter, verbose=2, tol=1e-3)

        elif self.method.lower() == "gmm_dirichlet":
            self.model = BayesianGaussianMixture(
                n_components=self.n_components, covariance_type="full",
                weight_concentration_prior=1.0/self.n_components,
                max_iter=self.max_iter, verbose=2, tol=1e-3)

    def fit(self, x):
        self._initialize_model()
        self.model.fit(x)

    def sample(self, n_samples):
        return self.model.sample(n_samples)[0]

    def save_model(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    def load_model(self, path):
        with open(path, "rb") as f:
            self.model = pickle.load(f)

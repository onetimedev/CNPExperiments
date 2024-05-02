import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
tfb = tfp.bijectors
tfd = tfp.distributions
tfk = tfp.math.psd_kernels
class GP():


    def __init__(self, length_scale: float, amplitude: float):

        self._length_scale = length_scale
        self._amplitude = amplitude


    def build_conditional_dist(self):

        x_values = np.random.uniform(-1., 1., (64, 1))
        x_values = x_values.astype(np.float64)

        kernel = tfk.ExponentiatedQuadratic(self._amplitude, self._length_scale)
        gp = tfd.GaussianProcess(
                kernel=kernel,
                index_points=x_values,
                observation_noise_variance=0)

        gp_joint_model = tfd.JointDistributionNamed({
            'amplitude': tfd.LogNormal(loc=0., scale=np.float64(1.)),
            'length_scale': tfd.LogNormal(loc=0., scale=np.float64(1.)),
            'observation_noise_variance': tfd.LogNormal(loc=0., scale=np.float64(1.)),
            'observations': gp,
        })

        samples = gp_joint_model.sample()
        lp = gp_joint_model.log_prob(samples)

        return samples



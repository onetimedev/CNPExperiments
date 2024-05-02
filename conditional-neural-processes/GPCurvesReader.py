import time

import torch
import torch.nn as nn
import torch.optim as optim
import collections
import matplotlib.pyplot as plt
import datetime

CNPRegressionDescription = collections.namedtuple("CNPRegressionDescription", ("context_set", "target_set"))

class GPCurvesReader(object):
    """
    Generates functions using a Gaussian Process (GP)

    - Supports vector inputs (x) and vector outputs (y).
    - Kernel is mean-squared exponential, using the x-value l2 norm coordinate distance scaled by some factor chosen randomly in a range.
    - Outputs are independent Gaussian Processes.
    """

    def __init__(self, batch_size, max_num_context, x_size = 1, y_size = 1, l1_scale = 0.4, sigma_scale = 1.0, testing = False):


        """
        Creates a regression dataset of functions sampled from a GP

        :param batch_size: An Integer
        :param max_num_context: The maximum number of observations in the context
        :param x_size: Integer >= 1 for length of "x values" vector.
        :param y_size: Integer >= 1 for length of "y values" vector.
        :param l1_scale: Float; typical scale for kernel distance function.
        :param sigma_scale: Float; Typical scale for variance.
        :param testing: Boolean that indicates whether we are testing.
        """

        self._batch_size = batch_size
        self._max_num_context = max_num_context
        self._x_size = x_size
        self._y_size = y_size
        self._l1_scale = l1_scale
        self._sigma_scale = sigma_scale
        self._testing = testing

    def _gaussian_kernel(self, xdata, l1, sigma_f, sigma_noise=2e-2):

        """
        Applies the Gaussian Kernel to generate curve data.

        :param xdata: Tensor with shape [batch_size, num_total_points, x_size] with the values of the x-axis data.
        :param l1: Tensor with shape [batch_size, y_size, x_size], the scale parameter of the Gaussian kernel
        :param sigma_f: Float tensor with shape [batch_size, y_size]; the magnitude of the std.
        :param sigma_noise: Float, std of the noise that we add for stability.

        :return:
            The kernel, a float tensor with shape [batch size, y_size, num_total_points, num_total_points].
        """
        # Determines the total number of points in the input data
        num_total_points = xdata.shape[1]

        # Add a new dimension to xdata at index 1
        x1 = xdata.unsqueeze(dim = 1)

        # Add a new dimension to xdata at index 2
        x2 = xdata.unsqueeze(dim = 2)

        # Compute the difference between every paid of points in xdata.
        diff = x1 - x2

        # Compute the normalised difference
        norm = (diff[:, None, :, :, :] / l1[:, :, None, None, :]) ** 2

        # Sum the normalised differences along the last dimension
        norm = norm.sum(dim=-1)

        # Compute the gaussian kernel
        kernel = (sigma_f ** 2)[:, :, None, None] * torch.exp(-0.5 * norm)

        # Add some small noise to the diagonal of the matrix to ensure numerical stability during the cholesky decomposition.
        kernel += (sigma_noise ** 2) * torch.eye(num_total_points)

        return kernel

    def generate_curves(self):
        """
        Builds the operation delivering the data.

        Generated functions are 'float32' with x values between -2 and 2

        :return:
            A 'CNPRegressionDescription' namedtuple
        """
        #  Generates a random integer tensor, between low=3 (inclusive) and high=n+1 (exclusive), with shape = 1d tensor with 1 element
        num_context = torch.randint(low = 3, high=self._max_num_context + 1, size = (1,))
        torch.random.seed()
        if self._testing:
            num_target = 400
            num_total_points = num_target

            # Generates a tensor of evenly spaced values within the range -2 (inclusive), 2 (inclusive).
            # 'steps' specifies the number of equally spaced steps within the range
            x_values = torch.linspace(start=-2, end=2, steps=num_target)

            # x_values.unsqueeze(dim=0) adds a new dimension to the tensor at index 0 (the first dimension), transforming
            # the 1d tensor into a 2d tensor, the new dimension has size = 1.
            # The .repeat() call repeats the new tensor along the specified dimensions, here it repeats the tensor 'self._batch_size'
            # times along the first dimension. The .unsqueeze(-1) call adds a new dimension to the tensor at the last index, the new
            # dimension has size 1.
            x_values = (
                x_values.unsqueeze(dim = 0).repeat([self._batch_size, 1]).unsqueeze(-1)
            )

        else:
            # NOTE: During training the number of target points and their x-positions are selected at random
            num_target = torch.randint(low=3, high=self._max_num_context + 1, size=(1,))
            num_total_points = num_context + num_target
            # The following line generates a 3 dimensional tensor with shape (self._batch_size, num_total_points, self._x_size)
            # The arithmetic operation first scales the values to be within the range [0,4] and subtracting 2 shifts them
            # to be in the range [-2, 2]
            x_values = (torch.rand((self._batch_size, num_total_points, self._x_size)) * 4 -2)


            # Set kernel parameters
        # Creates a 3d tensor of '1s' and then scales it by the self._l1_scale.
        l1 = torch.ones(self._batch_size, self._y_size, self._x_size) * self._l1_scale
        sigma_f = torch.ones(self._batch_size, self._y_size) * self._sigma_scale

        # Pass the x_values through the Gaussian kernel
        # [batch_size, y_size, num_total_points, num_total_points]
        kernel = self._gaussian_kernel(x_values, l1, sigma_f)

        # Calculate Cholesky decomposition, using double precision for better numerical stability.
        # 'kernel' is a matrix and this call performs the cholesky decomposition on said matrix.
        # The .type() call converts the kernel to a tensor of type double, this is because a Cholesky
        # decomposition often requires high numerical stability and this provides that. The final .type()
        # call converts the result of the Cholesky decomposition back to type float.
        cholesky = torch.linalg.cholesky(kernel.type(torch.DoubleTensor)).type(torch.FloatTensor)

        # Sample a curve
        # This line performs a matrix multiplication operation between the cholesky decomposition
        # (lower triangular matrix of the kernel matrix and its conjugate transpose) and a 4 dimensional matrix
        # composed of randomly sampled values from a standard gaussian (\mu = 0, \sigma = 1).
        y_values = torch.matmul(cholesky, torch.randn(self._batch_size, self._y_size, num_total_points, 1))

        # Reshape the y_values tensor. The first call to .squeeze(dim = 3) removes the dimension at index 3,
        # which in this case is the fourth dimension of y_values. The resulting tensor is 3d, and the call to .permute()
        # reorders the dimensions of the tensor, in this case the first dimension (0) remains in its original order, the
        # second dimension becomes num_total_points, and the third dimension becomes self.y_size
        y_values = y_values.squeeze(dim = 3).permute([0, 2, 1])

        if self._testing:
            # Select the targets
            target_x = x_values
            target_y = y_values

            # Select the observations. .randperm() generates a 1d tensor of integers of size 'num_target', filled with random
            # permutations of integers from 0 - num_target-1. These serve as indices for the context set.
            idx = torch.randperm(num_target)

            context_x = x_values[:, idx[:num_context], :]
            context_y = y_values[:, idx[:num_context], :]
        else:
            # Select the targets which will consist of the context points as well as some new target points
            target_x = x_values[:, : num_target + num_context, :]
            target_y = y_values[:, : num_target + num_context, :]

            # Select the observations
            context_x = x_values[:, :num_context, :]
            context_y = y_values[:, :num_context, :]

        context_set = (context_x, context_y)
        target_set = (target_x, target_y)

        return CNPRegressionDescription(context_set=context_set, target_set=target_set)

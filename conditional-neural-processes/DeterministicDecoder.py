import torch
import torch.nn as nn

class DeterministicDecoder(nn.Module):

    def __init__(self, sizes):
        super(DeterministicDecoder, self).__init__()

        # Initialize an empty container used to store layers of the neural network.
        self.linears = nn.ModuleList()

        for i in range(len(sizes)-1):
            # The nn.Linear() call creates a linear transformation layer. Aka fully connected layer aka dense layer.
            # Mathematically given as output = input x weight^T + bias
            # where input is the input tensor to the layer, weight is the learned weight matrix of the layer,
            # and bias is a learned bias vector of the layer.
            self.linears.append(nn.Linear(sizes[i], sizes[i + 1]))

    def forward(self, representation, target_x):
        """
           Take representation of current training set, and a target input x, return the probability of x being positive.

           :param representation: batch_size x representation_size
           :param target_x: batch_size x set_size x d
           :return:
           """
        # Obtain dimensionality of target_x
        batch_size, set_size, d = target_x.shape

        # The call to .unsqueeze(1) adds a new dimension to the tensor at index 1.
        # This effectively increases the tensor's dimensionality by one. The call
        # to .repeat() indicates how many repeats along each dimension.
        representation = representation.unsqueeze(1).repeat([1, set_size, 1])

        input = torch.cat((representation, target_x), dim=-1)

        x = input.view(batch_size * set_size, -1)

        # Pass the input forward through each hidden linear layer.
        for i, linear in enumerate(self.linears[:-1]):
            x = torch.relu(linear(x))
        # Pass the result to the output layer.
        x = self.linears[-1](x)

        # reshapes the tensor
        out = x.view(batch_size, set_size, -1)

        # Split the tensor into two along the final dimension '-1'. The first now represents the mean, and the second
        # represents the log variance.
        mu, log_sigma = torch.split(out, 1, dim=-1)

        # obtain the true variance by applying a softplus activation: f(x) = log(1 + \e^x). This is done to ensure the output
        # is always positive while maintaining a smooth gradient. The result is scaled by 0.9 and 0.1 is added to the scaled values.
        sigma = 0.1 + 0.9 * torch.nn.functional.softplus(log_sigma)

        # Creates a standard gaussian distribution with mean = mu and variance = sigma
        dist = torch.distributions.normal.Normal(loc=mu, scale=sigma)

        return dist, mu, sigma
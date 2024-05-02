import torch
import torch.nn as nn

class DeterministicEncoder(nn.Module):

    def __init__(self, sizes):

        super(DeterministicEncoder, self).__init__()

        # Initialize an empty container used to store layers of the neural network.
        self.linears = nn.ModuleList()

        for i in range(len(sizes)-1):
            # The nn.Linear() call creates a linear transformation layer. Aka fully connected layer aka dense layer.
            # Mathematically given as output = input x weight^T + bias
            # where input is the input tensor to the layer, weight is the learned weight matrix of the layer,
            # and bias is a learned bias vector of the layer.
            self.linears.append(nn.Linear(sizes[i], sizes[i + 1]))


    def forward(self, context_x, context_y):

        """
        Encode training set as one vector representation

        :param context_x: batch_size x set_size x feature dim
        :param context_y: batch_x x set_size x 1
        :return:
            Representation r
        """

        # This call to .cat() concatenates the tensors 'context_x' and 'context_y' along the last dimension (-1).
        # This essentially puts the x and y data into one tensor with each member along the 3rd dimension being of
        # size 2 i.e. [x64[x3[x, y]]] so the new shape is (64, 3, 2), from (64, 3, 1) for context_x/y respectively.
        encoder_input = torch.cat((context_x, context_y), dim=-1)

        # Extract the tensor shapes (3d)
        batch_size, set_size, filter_size = encoder_input.shape

        # The call to .view() reshapes the tensor from 3d to 2d, with the first dimension (rows) being equal to
        # batch_size x set_size. The -1 parameter tells pytorch to infer the size of this dimension automatically 
        x = encoder_input.view(batch_size * set_size, -1)

        # Make a forward pass through all the hidden layers of the neural network. self.linears[:-1] ensures
        # we do not pass the result of penultimate layer to the output layer just yet.
        for i, linear in enumerate(self.linears[:-1]):
            x = torch.relu(linear(x))

        # Pass through the output layer.
        x = self.linears[-1](x)
        # Reshape output
        x = x.view(batch_size, set_size, -1)
        # Compute the mean along the second dimension
        representation = x.mean(dim=1)
        return representation



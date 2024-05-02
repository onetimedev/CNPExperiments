import torch
import torch.nn as nn
from DeterministicDecoder import DeterministicDecoder as Decoder
from DeterministicEncoder import DeterministicEncoder as Encoder
class DeterministicModel(nn.Module):

    def __init__(self, encoder_sizes, decoder_sizes):
        super(DeterministicModel, self).__init__()
        # Initialise the encoder and decoder neural networks.
        self._encoder = Encoder(encoder_sizes)
        self._decoder = Decoder(decoder_sizes)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def forward(self, query, target_y=None):

        (context_x, context_y), target_x = query
        # pass the context set to the encoder to obtain a representation
        representation = self._encoder(context_x, context_y)

        # pass the representation R to the decoder to obtain a mean prediction and associated uncertainty.
        dist, mu, sigma = self._decoder(representation, target_x)

        log_p = None if target_y is None else dist.log_prob(target_y)

        return log_p, mu, sigma
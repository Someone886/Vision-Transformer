import flax.linen as nn

import jax
import jax.numpy as jnp

from transformer_attention import *

class FeedForward(nn.Module):
    """ Encoder FeedForward network, with a single hidden layer that uses gelu non linearity
         :param filter_size: Hidden layer size
         :param hidden_size: Output size
         :param dropout: Dropout rate to be applied
    """
    filter_size: int
    hidden_size: int
    dropout: float

    @nn.compact
    def __call__(self, x, train=True):
        x = nn.Dense(self.filter_size, use_bias=True, kernel_init=nn.initializers.xavier_uniform())(x)
        x = nn.gelu(x)
        x = nn.Dense(self.hidden_size, use_bias=True, kernel_init=nn.initializers.xavier_uniform())(x)
        
        if self.dropout > 0.0:
            x = nn.Dropout(rate=self.dropout)(x, deterministic=not train)
        return x

class EncoderBlock(nn.Module):
    """ Transformer Encoder Block, that does:
            z_l_temp = MSA(LN(z_l-1)) + z_l-1
            z_l = MLP(LN(z_l_temp)) + z_l_temp
            return z_l

         :param MSALayerConfig: Parameters for MSA (Multi-headed self attention) Block
         :param filter_size: Hidden layer size
         :param hidden_size: Output size
         :param dropout: Dropout rate to be applied
    """
    MSAConfig: MSALayerConfig
    filter_size: int
    hidden_size: int
    dropout: float = 0.0

    def setup(self):
        self.layerNorm1 = nn.LayerNorm()
        self.layerNorm2 = nn.LayerNorm()
        self.multiHeadAttention = MultiHeadAttention(config=self.MSAConfig)
        self.feedForward = FeedForward(self.filter_size, self.hidden_size, self.dropout)

    def __call__(self, x, train=True):
        out = x

        #################################
        ### z_l_temp = MSA(LN(z_l-1)) + z_l-1
        ### z_l = MLP(LN(z_l_temp)) + z_l_temp
        ### return z_l
        ################################

        # TODO: Your code here

        #################################

        return out

class TransformerEncoder(nn.Module):
    """ Transformer Encoder
         :param n_layer: Number of encoder blocks
         :param MSALayerConfig: Parameters for MSA (Multi-headed self attention) Block
         :param filter_size: Hidden layer size
         :param hidden_size: Output size
         :param dropout: Dropout rate to be applied
    """
    n_layer: int

    MSAConfig: MSALayerConfig
    filter_size: int
    hidden_size: int
    dropout: float = 0.0

    @nn.compact
    def __call__(self, x, train=True):
        out = x
        for _ in range(self.n_layer):
            # Pass the input through encoder blocks            
            out = ... # TODO: Your code here
        return out
        


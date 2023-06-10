import flax.linen as nn
from flax.linen.linear import Dense
from flax.linen.module import compact
from flax.linen.module import Module


import jax
import jax.numpy as jnp

class MSALayerConfig:
    """ Helper class that holds MSA params
         :param n_heads: Number of heads for multi-headed attention
         :param qk_dim: Dimension for keys and queries
         :param v_dim: Dimension for values
         :param out_dim: Dimension of final output
    """
    n_heads: int
    qk_dim: int
    v_dim: int
    out_dim: int

    def __init__(self, n_heads: int, qk_dim: int, v_dim: int, out_dim: int):
        self.n_heads = n_heads
        self.qk_dim = qk_dim
        self.v_dim = v_dim
        self.out_dim = out_dim

def dot_product_attention(queries, keys, values):
    """Computes dot product attention weights given queries, keys, values.
        :param queries: Tensor with shape [heads (optional), n_queries, depth_k]
        :param keys:    Tensor with shape [heads (optional), n_keyval, depth_k]
        :param values:  Tensor with shape [heads (optional), n_keyval, depth_v]
        :param mask:    Tensor with shape [n_queries, n_queries]

        :return output: Tensor with shape [heads (optional), n_queries, depth_v]
    """
    key_dim = jnp.array(keys.shape[-1], dtype=jnp.float32)
    similarity = jnp.einsum('...qd,...kd->...qk', queries, keys)/jnp.sqrt(key_dim)

    weights = jax.nn.softmax(similarity, axis=-1)
    return weights @ values

class MultiHeadAttention(Module):
    """ Multi-Head Attention Block
         :param config: Configuration parameters for attention block
    """
    config: MSALayerConfig

    @compact
    def __call__(self, input_q, input_kv):
        """ Forward method for multi-headed attention
            :param input_q: Queries (SeqLen_q, Channels_q)
            :param input_kv: Keys/Values (SeqLen_k/v, Channels_k/v)

            :return output: Attention output (SeqLen_q, out_dim)
        """
        assert self.config.qk_dim % self.config.n_heads == 0 and self.config.v_dim % self.config.n_heads == 0, "invalid q_channels or kv_channels"

        q = Dense(self.config.qk_dim, name = 'Query linear',use_bias=False, kernel_init=nn.initializers.xavier_uniform())(input_q)
        k = Dense(self.config.qk_dim, name = 'Key linear',use_bias=False, kernel_init=nn.initializers.xavier_uniform())(input_kv)
        v = Dense(self.config.v_dim, name = 'Value linear',use_bias=False, kernel_init=nn.initializers.xavier_uniform())(input_kv)

        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        attn_out = dot_product_attention(q, k, v)
        out = self._combine_heads(attn_out)
        out = Dense(self.config.out_dim, name = 'Out linear',use_bias=False, kernel_init=nn.initializers.xavier_uniform())(out)
        return out

    def _split_heads(self, tensor):
        """ Split tensor into multiple heads for multi-headed attention
             :param tensor: Tensor of shape (L, C)
             :return output: Tensor of shape(H, L, C/H)
        """
        L, C = tensor.shape
        H = self.config.n_heads

        output = jnp.reshape(tensor, (L, H, int(C/H)))
        output = jnp.transpose(output, axes=(1,0,2))
        return output

    def _combine_heads(self, tensor):
        """ Re-combine tensor from multiple heads into singular head after multi-headed attention
             :param tensor: Tensor of shape (H, L, C/H)
             :return output: Tensor of shape(L, C)
        """
        H, L, dim = tensor.shape
        C = dim * H
        
        output = jnp.transpose(tensor, axes=(1,0,2))
        output = jnp.reshape(output, (L, C))
        return output
    
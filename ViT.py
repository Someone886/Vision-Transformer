import flax.linen as nn

import jax
import jax.numpy as jnp

from transformer_attention import *
from transformer import *

import matplotlib.pyplot as plt

class ViT(nn.Module):
    """ Vision Transformer Model: https://arxiv.org/pdf/2010.11929.pdf
         
         :param patch_size: Size of the patch
         :param embedding_dim: Embedding dim of the network
         :param num_patches: Number of patches. Make sure that H/patch_size == num_patches and W/patch_size == num_patches
         :param n_classes: Number of output classes
         :param n_heads: Number of attention heads
         :param n_layer: Number of encoder blocks
         :param feedforward_dim: Feedforward dim of encoder blocks
         :param dropout: Dropout rate to be applied
    """

    patch_size: int
    embedding_dim: int
    num_patches: int
    n_classes: int
    n_heads: int
    n_layer: int
    feedforward_dim: int
    dropout: float = 0.0

    def setup(self):
        self.patch_projection = nn.Dense(self.embedding_dim)
        self.cls_token = self.param('cls_token', nn.initializers.lecun_normal(), (1, self.embedding_dim)) * 0.02
        self.position_encoding = self.param('position_encoding', nn.initializers.lecun_normal(),
                                            (self.num_patches * self.num_patches + 1, self.embedding_dim)) * 0.02

        config = MSALayerConfig(self.n_heads, self.embedding_dim, self.embedding_dim, self.embedding_dim)
        self.transformer_encoder = TransformerEncoder(self.n_layer, config, self.feedforward_dim, self.embedding_dim, self.dropout)
        self.norm = nn.LayerNorm()
        self.head = nn.Dense(self.n_classes)
    
    def patchify(self, x, patch_size):
        '''
            x: Input image with shape (H, W, C) (jnp tensor)
            patch_size: Size of image patch (int)

            Return: out: Flattened image with shape (num_patches * num_patches, C * patch_size * patch_size)
        '''
        h = x.shape[0] // patch_size
        w = x.shape[1] // patch_size
        c = x.shape[2]
        assert self.num_patches == h, f' {self.num_patches, h} Num patches and patch size incompatible with image height'
        assert self.num_patches == w, 'Num patches and patch size incompatible with image width'
        x = jnp.reshape(x, newshape=(h, patch_size, w, patch_size, c))
        x = jnp.transpose(x, (0, 2, 4, 1, 3))
        x = jnp.reshape(x, newshape=(h*w, c*patch_size*patch_size))
        return x

    def visualize_patching(self, X, num_rows, num_columns, num_channels, figsize=(12, 8)):
        _, ax = plt.subplots(num_rows, num_columns, figsize=figsize)
        for i in range(num_rows * num_columns):
            img = X[i]
            img = img.reshape(num_channels, self.patch_size, self.patch_size)
            img = img.transpose(1,2,0)
            ax[i // num_columns, i % num_columns].imshow(img)
            ax[i // num_columns, i % num_columns].axis("off")

        plt.show()
    
    def __call__(self, x, train=True):
        """ Returns class logits of input image x
            :param x: Image (H, W, C)
        """
        patched_image = self.patchify(x, patch_size = self.patch_size)
        linear_embedding = self.patch_projection(patched_image)
        cls_prepended = jnp.concatenate((self.cls_token, linear_embedding), axis=0)
        x_with_position_embeddings = cls_prepended + self.position_encoding
        x = x_with_position_embeddings

        x = self.transformer_encoder(x, train=train)
        x = self.norm(x)
        
        # x (- [SeqLen, Dim].
        # MLP on class token, x0 = x[0, :]
        x = self.head(x[0, :])
        return x

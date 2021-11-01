from labml_nn.transformers.mha import PrepareForMultiHeadAttention
import math
from typing import Optional

import torch
from torch import nn

from labml_helpers.module import Module
from labml_nn.transformers.feed_forward import FeedForward
from labml_nn.transformers.mha import PrepareForMultiHeadAttention
from labml_nn.utils import clone_module_list


class FeedbackAttention(Module):

    """Feedback Attention
    This module computes recurrent attention similar to attention from original transformers
    paper.

    """

    def __init__(self, heads: int, d_model: int, dropout_prob: float = 0.1, *,
                 is_kv_precomputed: bool = False):
        """
        * 'heads' is the number of attention heads
        * `d_model` is the number of features in the transformer
        * `dropout_prob` is the attention dropout probability
        * `is_kv_precomputed` is whether key, value tensors are already calculated
        """

        super().__init__()

        # Number of features per head
        self.d_k = d_model // heads
        #
        self.heads = heads

        # These transform the `query` multi-headed attention.
        self.query = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=False)
        # These transform the `key` and `value` for multi-headed attention.
        if not is_kv_precomputed:
            self.key = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=False)
            self.value = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=True)
        # Keys and values are already calculated
        else:
            self.key = None
            self.value = None

        # Output layer
        self.output = nn.Linear(d_model, d_model)
        # Dropout
        self.dropout = nn.Dropout(dropout_prob)
        # Scaling factor before the softmax
        self.scale = 1 / math.sqrt(self.d_k)

        # Softmax for attention along the time dimension of `key`
        self.softmax = nn.Softmax(dim=0)

        # Number of relative positions
        self.P = 2 ** 12

        # Relative positional embeddings for key relative to the query.
        self.key_pos_embeddings = nn.Parameter(torch.zeros((self.P, heads, self.d_k)), requires_grad=True)
        # Relative positional embedding bias for key relative to the query.
        self.key_pos_bias = nn.Parameter(torch.zeros((self.P, heads)), requires_grad=True)
        # Positional embeddings for the query is independent of the position of the query
        self.query_pos_bias = nn.Parameter(torch.zeros((heads, self.d_k)), requires_grad=True)

        # We store attentions so that it can be used for logging, or other computations if needed
        self.attn = None

    def get_scores(self, query: torch.Tensor, key: torch.Tensor):
        r"""
        ### Get attention scores
        We use relative positional encodings for attention, similar
        to [relative multi-head attention form Transformer-XL paper](../relative_mha.html).
        Attention from current step's query to key in step $j$ (relative to current step) is,
        \begin{align}
        A_{j} &= Q^\top K_j \\
            &= lin_q(X^q + P_q)^\top lin_k(X^k_j + P_j) \\
            &= (Q + U^Q)^\top(K_j + U^K_j) \\
            &= \underset{\textcolor{lightgreen}{A}}{Q^\top K_j} +
               \underset{\textcolor{lightgreen}{B}}{Q^\top U^K_j} +
               \underset{\textcolor{lightgreen}{C}}{{U^Q}^\top K_j} +
               \underset{\textcolor{lightgreen}{D}}{{U^Q}^\top U^K_j}
        \end{align}
        where $Q, K_j$, are linear transformations of
         original embeddings $X^q, X^k_j$
         and $U^Q, U^K_j$ are linear transformations of
         positional encodings $P_q, P_j$.
        We replace term $\textcolor{lightgreen}{D}$ with $S_j$.
        """

        # $U^K_j$
        key_pos_emb = self.key_pos_embeddings[-key.shape[0]:]
        # $U^Q$
        query_pos_bias = self.query_pos_bias[None, :, :]
        # $S_j$
        key_pos_bias = self.key_pos_bias[-key.shape[0]:]

        # $\underset{\textcolor{lightgreen}{A}}{Q^\top K_j} + \underset{\textcolor{lightgreen}{C}}{{U^Q}^\top K_j}$
        ac = torch.einsum('bhd,jbhd->jbh', query + query_pos_bias, key)
        # $\underset{\textcolor{lightgreen}{B}}{Q^\top U^K_j} + \underset{\textcolor{lightgreen}{D}}{S_j}$
        bd = torch.einsum('bhd,jhd->jbh', query, key_pos_emb) + key_pos_bias[:, None, :]

        # $A_j$
        return ac + bd

    def forward(self, *,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor):
        """
        * `query` has shape `[batch_size, d_model]`
        * `key` and `value` has shape `[seq_len, batch_size, d_model]`
        """

        # Prepare `query`, `key` and `value` for attention computation
        # `key` and `value`  will then have shape `[seq_len, batch_size, heads, d_k]`
        # and `query` will have shape `[batch_size, heads, d_k]`
        query = self.query(query)
        if self.key:
            key = self.key(key)
        if self.value:
            value = self.value(value)

        # Compute attention scores.
        # Results in a tensor of shape `[seq_len, batch_size, heads]`
        scores = self.get_scores(query, key)

        # Scale scores $\frac{1}{\sqrt{d_k}}$
        scores *= self.scale

        # Softmax
        attn = self.softmax(scores)

        # Apply dropout
        attn = self.dropout(attn)

        # Multiply by the values
        x = torch.einsum("jbh,jbhd->bhd", attn, value)

        # Concatenate multiple heads
        x = x.reshape(x.shape[0], -1)

        # Output layer
        return self.output(x)


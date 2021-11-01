import math
from typing import Optional

import torch
from torch import nn

from labml_helpers.module import Module
from labml_nn.transformers.feed_forward import FeedForward
from labml_nn.transformers.mha import PrepareForMultiHeadAttention
from labml_nn.utils import clone_module_list

from feedback_transformer_layer import FeedbackTransformerLayer
from stack import Stack


class FeedbackTransformerKV(Module):
    """
    ## Updated Feedback Transformer Module
    This is the updated feedback transformer module that caches the keys and values.
    """

    def __init__(self, layer: FeedbackTransformerLayer, n_layers: int, d_model: int, heads: int):
        """
        * `layer` is the feedback transformer layer, which we clone for each layer
        * `n_layers` is the number of layers in the transformer
        * `d_model` is the number of features in the transformer
        * 'heads' is the number of attention heads
        """

        super().__init__()
        # Make copies of the transformer layer
        self.layers = clone_module_list(layer, n_layers)
        # Final normalization layer
        self.norm = nn.LayerNorm([layer.size])
        # Memory vectors are computed as a weighted sum of representations of each layer.
        # This is the weights parameter for that.
        self.weights = nn.Parameter(torch.ones(n_layers + 1), requires_grad=True)
        # Softmax for weights before taking the weighted sum
        self.softmax = nn.Softmax(0)

        # Number of features in a head
        d_k = d_model // heads
        # Module to transform embeddings (memory) to get keys
        self.key = PrepareForMultiHeadAttention(d_model, heads, d_k, bias=False)
        # Module to transform embeddings (memory) to get keys
        self.value = PrepareForMultiHeadAttention(d_model, heads, d_k, bias=False)

        # Memory for stacked keys
        self.mem_key = Stack(512)
        # Memory for stacked values
        self.mem_value = Stack(512)

    def forward(self, x_seq: torch.Tensor):
        """
        * `x_seq` is the input with shape `[seq_len, batch_size, d_model]`
        """

        # Split the input to a list along the sequence axis
        x_seq = torch.unbind(x_seq, dim=0)
        # List to store the outputs
        res = []
        # For each input step
        for step, x in enumerate(x_seq):
            # List to store layer outputs
            layer_outputs = [x]

            # Stack of keys and values
            key_tensor = None
            value_tensor = None
            # Get the keys and values tensors if we are beyond the initial step
            if step > 0:
                key_tensor = self.mem_key.get()
                value_tensor = self.mem_value.get()

            # Run through each layer
            for layer in self.layers:
                # Get layer output
                x = layer(x=x, key=key_tensor, value=value_tensor)
                # Append them to the list of layer outputs
                layer_outputs.append(x)

            # Stack the layer outputs to a tensor
            layer_outputs = torch.stack(layer_outputs)
            # Calculate the memory vector as a weighted sum of layer outputs
            mem = torch.einsum('lbd,l->bd', layer_outputs, self.softmax(self.weights))
            # Calculate the keys from memory and add it to the stack
            self.mem_key.append(step, self.key(mem))
            # Calculate the values from memory and add it to the stack
            self.mem_value.append(step, self.value(mem))
            # Append the output to results
            res.append(x)

        # Stack the output tensors
        res = torch.stack(res)
        # Normalize the output
        return self.norm(res)

    def free(self):
        self.mem_key.free()
        self.mem_value.free()
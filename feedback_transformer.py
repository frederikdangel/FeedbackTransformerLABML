import math
from typing import Optional

import torch
from torch import nn
from labml_helpers.module import Module
from labml_nn.transformers.feed_forward import FeedForward
from labml_nn.transformers.mha import PrepareForMultiHeadAttention
from labml_nn.utils import clone_module_list
from feedback_transformer_layer import FeedbackTransformerLayer


class FeedbackTransformer(Module):
    """
    ## Feedback Transformer Module
    """

    def __init__(self, layer: FeedbackTransformerLayer, n_layers: int):
        """
        * `layer` is the feedback transformer layer, which we clone for each layer
        * `n_layers` is the number of layers in the transformer
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

    def forward(self, x_seq: torch.Tensor):
        """
        * `x_seq` is the input with shape `[seq_len, batch_size, d_model]`
        """

        # Split the input to a list along the sequence axis
        x_seq = torch.unbind(x_seq, dim=0)
        # List to store the outputs
        res = []
        # List to store the memory vectors
        mem = []
        # For each input step
        for x in x_seq:
            # List to store layer outputs
            layer_outputs = [x]

            # If there is memory, stack them into a vector
            mem_tensor = torch.stack(mem) if mem else None

            # Run through each layer
            for layer in self.layers:
                # Get layer output
                x = layer(x=x, key=mem_tensor, value=mem_tensor)
                # Append them to the list of layer outputs
                layer_outputs.append(x)

            # Stack the layer outputs to a tensor
            layer_outputs = torch.stack(layer_outputs)
            # Calculate the memory vector as a weighted sum of layer outputs
            mem.append(torch.einsum('lbd,l->bd', layer_outputs, self.softmax(self.weights)))
            # Append the output to results
            res.append(x)

        # Stack the output tensors
        res = torch.stack(res)
        # Normalize the output
        return self.norm(res)

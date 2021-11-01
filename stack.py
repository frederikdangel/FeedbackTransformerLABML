import math
from typing import Optional

import torch
from torch import nn

from labml_helpers.module import Module
from labml_nn.transformers.feed_forward import FeedForward
from labml_nn.transformers.mha import PrepareForMultiHeadAttention
from labml_nn.utils import clone_module_list


class StackFunction(torch.autograd.Function):
    """
    ### Stack Function implementation
    We implement a custom function instead of appending to a python list
    and then doing `torch.stack`.
    This greatly improves the performance over calling `torch.stack` at
    each step along the sequence.
    Everytime `torch.stack` is called, it creates a new tensor, while
    this method and the accompanying class `Stack` share memory for each step.
    """

    @staticmethod
    def forward(ctx, memory, memory_grad, last, n):
        """
        * `ctx` is the context of the function (which lets us cache stuff)
        * `memory` is the shared memory tensor where we stack and store the values of each step (keys & values)
        * `memory_grad` is the shared memory tensor to store and accumulate gradients of each step
        * `last` is the last value stacked
        * `n` is the number of steps (i.e. size of the stack)
        This returns the stacked tensor for steps upto `n`.
        """

        # Cache accumulated gradients
        ctx._mem_grad = memory_grad
        # Cache the size of the stack
        ctx._n = n
        # Return the stack
        return memory[:n + 1]

    @staticmethod
    def backward(ctx, grad_output):
        """
        * `grad_output` is the gradient with respect to the output of about `forward` function
        This accumulates the gradients in the shared memory tensor and return the
        gradients with respect to the `last` result in the stack.
        """
        # Get the current size of the stack
        n = ctx._n
        # Get the accumulated gradients
        memory_grad = ctx._mem_grad
        # Add the gradients
        memory_grad[:n + 1] += grad_output
        # Return the gradients w.r.t to last value in the stack
        return None, None, memory_grad[n], None


class Stack:
    """
    ### Stack Module
    This uses the stack function defined above, and does the necessary initializations.
    """

    def __init__(self, max_len: int):
        """
        * `max_len` is the maximum size of the stack
        """
        self.max_len = max_len
        self.memory = None
        self.memory_grad = None
        self.last = None
        self.n = -1
        self.last_get_n = -1

    def append(self, n: int, value: torch.Tensor):
        """
        * `n` is the size of the stack
        * `value` is the tensor that needs to be added to the stack
        """

        # You need to get (use) the stack after adding a value.
        # Otherwise this implementation fails
        assert n == 0 or self.last_get_n == n - 1, f"{n}, {self.last_get_n}"

        # Do this without gradients
        with torch.no_grad():
            # Initialize the shared memory tensor to keep the stack
            if self.memory is None or self.memory.shape[1:] != value.shape:
                # This should only happen when the stack is empty
                assert n == 0
                # Create a tensor for the stack
                self.memory = value.new_zeros(self.max_len, *value.shape, requires_grad=False)
                # Create a tensor to accumulate the gradients
                self.memory_grad = value.new_zeros(self.memory.shape, requires_grad=False)
            # The memory is already initialized but we are resetting the stack.
            #
            # This could have been another function like `reset`, but
            # we found this easier to use.
            elif n == 0:
                # Reset accumulated gradients
                self.memory_grad.fill_(0.)

            # Set the value in the correct position of the stack
            self.memory.data[n] = value.detach()
            # Keep track of the stack (for debugging)
            self.n = n

        # Keep track of the last value added to the stack.
        # We need this to be passed on to `StackFunction` in order
        # to get the gradients propagated backwards.
        self.last = value

    def get(self):
        """
        Returns the stack
        """

        # Keep track of the size of the stack when it was used.
        # This is used for a sanity check in `append`.
        self.last_get_n = self.n
        # Take it all through `StackFunction` so that `StackFunction.backwards`
        # is called by PyTorch during backpropagation.
        return StackFunction.apply(self.memory, self.memory_grad, self.last, self.n)

    def free(self):
        """
        To release memory
        """

        self.memory = None
        self.memory_grad = None
        self.last = None


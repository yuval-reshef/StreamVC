from typing import Callable
import torch
from einops import pack, unpack


def auto_batching(input_patterns: tuple[str], output_pattern: str) -> Callable[[Callable], Callable]:
    """
    A decorator for automatic support of multiple/no batching dimensions
    for an object method that only support a single batching dimension.

    Parameters
    ----------
    input_patterns : tuple[str]
        A tuple of patterns for each input tensor. The pattern is a string
        that describes the shape of the tensor. The pattern should have a '*'
        character to represent the batch dimension, and a letter for each
        other dimension. See `einops.pack` documentation for more information.
    output_pattern : str
        A similar pattern for the output tensor. the batch dimensions for the 
        output tensor are taken from the first input tensor.

    Returns
    -------
    A decorator

    Example
    -------
    ```python
    @auto_batching(['* c h w', '* c'], '* h w')
    def forward(self, x, y):
        return torch.einsum('bchw, bc -> bhw', x, y)
    ```
    """
    def decorator(fn: Callable) -> Callable:
        def wrapper(self, *args, **kwargs):
            new_args = []
            batches = []
            for i, arg in enumerate(args):
                if len(input_patterns) <= i:
                    new_args.append(arg)
                    continue

                assert isinstance(
                    arg, torch.Tensor), f"Argument {i} is not a tensor"
                new_arg, batch = pack([arg], input_patterns[i])
                new_args.append(new_arg)
                batches.append(batch)

            output = fn(self, *new_args, **kwargs)
            [output] = unpack(output, batches[0], output_pattern)
            return output
        return wrapper
    return decorator

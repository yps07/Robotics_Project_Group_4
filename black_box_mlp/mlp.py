"""Black-box multilayer perceptron for inverse arm dynamics."""

from collections.abc import Sequence

import torch
from torch import nn


_ACTIVATIONS = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "silu": nn.SiLU,
    "gelu": nn.GELU,
}


class MLPInverseDynamics(nn.Module):
    """Map normalized state features to normalized arm torques.

    The default 13-dimensional input uses the dataset's periodic encoding of
    the unlimited root angle. Output normalization is handled by the dataset
    scaler; this module deliberately has no output activation.
    """

    def __init__(
        self,
        input_dim=13,
        output_dim=4,
        hidden_sizes: Sequence[int] = (256, 256, 128),
        activation="tanh",
        dropout=0.0,
    ):
        super().__init__()
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("input_dim and output_dim must be positive.")
        if not hidden_sizes or any(int(width) <= 0 for width in hidden_sizes):
            raise ValueError("hidden_sizes must contain positive widths.")
        if activation not in _ACTIVATIONS:
            raise ValueError(
                f"Unsupported activation {activation!r}; "
                f"choose from {sorted(_ACTIVATIONS)}."
            )
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1).")

        layers = []
        previous = input_dim
        for width in hidden_sizes:
            layers.append(nn.Linear(previous, int(width)))
            layers.append(_ACTIVATIONS[activation]())
            if dropout:
                layers.append(nn.Dropout(dropout))
            previous = int(width)
        layers.append(nn.Linear(previous, output_dim))
        self.network = nn.Sequential(*layers)
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.hidden_sizes = tuple(int(width) for width in hidden_sizes)
        self.activation_name = activation
        self.dropout = float(dropout)

    def forward(self, x):
        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected {self.input_dim} input features, got {x.shape[-1]}."
            )
        return self.network(x)

    def get_config(self):
        return {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "hidden_sizes": list(self.hidden_sizes),
            "activation": self.activation_name,
            "dropout": self.dropout,
        }


# A short alias is useful for scripts and keeps older notebook imports simple.
MLP = MLPInverseDynamics

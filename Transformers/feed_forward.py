import torch
import torch.nn as nn
from labml_helpers.module import Module


class FeedForwar(nn.Module):
    """
    description:FFN module
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        activation=nn.ReLU(),
        is_gated: bool = False,
        bias1: bool = True,
        bias2: bool = True,
        bias_gate: bool = True,
    ):
        """
        description:
        param {int} d_model: the number of features in a token embedding
        param {int} d_ff: the number of features in the hidden layer
        param {float} dropout: the dropout probability
        param {*} activation:
        param {bool} is_gated
        param {bool} bias1
        param {bool} bias2
        param {bool} bias_gate: specified whether the fully connected layer for the gate should have a learnable bias
        return {*}
        """

        super().__init__()
        self.layer1 = nn.Linear(d_model, d_ff, bias=bias1)
        self.layer2 = nn.Linear(d_ff, d_model, bias=bias2)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.is_gated = is_gated
        if is_gated:
            self.linear_v = nn.Linear(d_model, d_ff, bias=bias_gate)

    def forward(self, x: torch.Tensor):
        # $f(x W_1 + b_1)$
        g = self.activation(self.layer1(x))
        if self.is_gated:
            x = g * self.linear_v(x)
        else:
            x = g
        x = self.dropout(x)
        return self.layer2(x)

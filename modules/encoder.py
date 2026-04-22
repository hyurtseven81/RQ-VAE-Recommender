
from torch import Tensor, nn

from modules.normalize import L2NormalizationLayer

_ACTIVATIONS = {"silu": nn.SiLU, "relu": nn.ReLU, "gelu": nn.GELU}


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        out_dim: int,
        dropout: float = 0.0,
        normalize: bool = False,
        activation: str = "silu",
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.out_dim = out_dim
        self.dropout = dropout

        try:
            act_cls = _ACTIVATIONS[activation.lower()]
        except KeyError as e:
            raise ValueError(
                f"Unknown MLP activation {activation!r}; "
                f"choose from {sorted(_ACTIVATIONS)}"
            ) from e

        dims = [self.input_dim] + self.hidden_dims + [self.out_dim]

        self.mlp = nn.Sequential()
        for i, (in_d, out_d) in enumerate(zip(dims[:-1], dims[1:])):
            self.mlp.append(nn.Linear(in_d, out_d, bias=False))
            if i != len(dims) - 2:
                self.mlp.append(act_cls())
                if dropout != 0:
                    self.mlp.append(nn.Dropout(dropout))
        self.mlp.append(L2NormalizationLayer() if normalize else nn.Identity())

    def forward(self, x: Tensor) -> Tensor:
        assert x.shape[-1] == self.input_dim, (
            f"Invalid input dim: Expected {self.input_dim}, found {x.shape[-1]}"
        )
        return self.mlp(x)

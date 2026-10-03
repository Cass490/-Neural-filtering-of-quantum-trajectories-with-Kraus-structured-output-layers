# Sequence backbones: TCN, GRU, LSTM, RNN, ESN, gated linear SSM, Neural ODE and Transformer.
from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


class SequenceBackbone(nn.Module):
    output_dim: int


class GRUBackbone(SequenceBackbone):
    def __init__(self, input_dim: int, hidden_dim: int, layers: int = 2, dropout: float = 0.0):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.core = nn.GRU(
            hidden_dim,
            hidden_dim,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.output_dim = hidden_dim
        self.last_hidden_rms = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        out, _ = self.core(x)
        self.last_hidden_rms = float(out.detach().pow(2).mean().sqrt().cpu())
        return out


class LSTMBackbone(SequenceBackbone):
    def __init__(self, input_dim: int, hidden_dim: int, layers: int = 2, dropout: float = 0.0):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.core = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.output_dim = hidden_dim
        self.last_hidden_rms = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        out, _ = self.core(x)
        self.last_hidden_rms = float(out.detach().pow(2).mean().sqrt().cpu())
        return out


class RNNBackbone(SequenceBackbone):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        layers: int = 2,
        orthogonal_init: bool = False,
        layer_norm: bool = False,
        residual: bool = False,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.core = nn.RNN(hidden_dim, hidden_dim, num_layers=layers, batch_first=True, nonlinearity="tanh")
        self.norm = nn.LayerNorm(hidden_dim) if layer_norm else nn.Identity()
        self.residual = residual
        self.output_dim = hidden_dim
        self.last_hidden_rms = 0.0
        if orthogonal_init:
            for name, param in self.core.named_parameters():
                if "weight_hh" in name:
                    nn.init.orthogonal_(param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.input_proj(x)
        out, _ = self.core(z)
        if self.residual:
            out = out + z
        out = self.norm(out)
        self.last_hidden_rms = float(out.detach().pow(2).mean().sqrt().cpu())
        return out


class CausalConv1d(nn.Conv1d):
    def __init__(self, *args: Any, dilation: int = 1, **kwargs: Any):
        kernel_size = kwargs.get("kernel_size", args[2] if len(args) > 2 else 3)
        self.left_padding = (int(kernel_size) - 1) * dilation
        super().__init__(*args, dilation=dilation, padding=0, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(F.pad(x, (self.left_padding, 0)))


class TCNBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float, gated: bool):
        super().__init__()
        out_channels = 2 * channels if gated else channels
        self.conv1 = CausalConv1d(channels, out_channels, kernel_size=kernel_size, dilation=dilation)
        self.conv2 = CausalConv1d(channels, out_channels, kernel_size=kernel_size, dilation=dilation)
        self.dropout = nn.Dropout(dropout)
        self.gated = gated
        self.norm1 = nn.GroupNorm(1, channels)
        self.norm2 = nn.GroupNorm(1, channels)

    def _activate(self, x: torch.Tensor) -> torch.Tensor:
        if self.gated:
            a, b = x.chunk(2, dim=1)
            return torch.tanh(a) * torch.sigmoid(b)
        return F.gelu(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self._activate(self.conv1(x))
        x = self.norm1(self.dropout(x))
        x = self._activate(self.conv2(x))
        x = self.norm2(self.dropout(x))
        return F.gelu(x + residual)


class TCNBackbone(SequenceBackbone):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        layers: int = 5,
        kernel_size: int = 5,
        dropout: float = 0.1,
        gated: bool = True,
    ):
        super().__init__()
        self.input_proj = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
        self.blocks = nn.ModuleList(
            [TCNBlock(hidden_dim, kernel_size, 2**i, dropout, gated) for i in range(layers)]
        )
        self.output_dim = hidden_dim
        self.last_hidden_rms = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        x = x.transpose(1, 2)
        self.last_hidden_rms = float(x.detach().pow(2).mean().sqrt().cpu())
        return x


class ESNBackbone(SequenceBackbone):

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        spectral_radius: float = 0.9,
        leak_rate: float = 0.5,
        sparsity: float = 0.9,
        seed: int = 0,
    ):
        super().__init__()
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)
        w_in = torch.randn(hidden_dim, input_dim, generator=gen) / math.sqrt(max(1, input_dim))
        w = torch.randn(hidden_dim, hidden_dim, generator=gen)
        mask = torch.rand(hidden_dim, hidden_dim, generator=gen) > sparsity
        w = w * mask
        v = torch.randn(hidden_dim, generator=gen)
        for _ in range(50):
            v = w @ v
            v = v / (v.norm() + 1e-12)
        radius = (w @ v).norm() / (v.norm() + 1e-12)
        w = w * (spectral_radius / (radius + 1e-12))
        self.register_buffer("w_in", w_in)
        self.register_buffer("w_res", w)
        self.register_buffer("bias", 0.1 * torch.randn(hidden_dim, generator=gen))
        self.leak_rate = float(leak_rate)
        self.output_dim = hidden_dim
        self.last_hidden_rms = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        state = torch.zeros(b, self.output_dim, device=x.device, dtype=x.dtype)
        outputs = []
        for i in range(t):
            proposal = torch.tanh(F.linear(x[:, i], self.w_in, self.bias) + F.linear(state, self.w_res))
            state = (1.0 - self.leak_rate) * state + self.leak_rate * proposal
            outputs.append(state)
        out = torch.stack(outputs, dim=1)
        self.last_hidden_rms = float(out.detach().pow(2).mean().sqrt().cpu())
        return out


class GatedLinearSSMBackbone(SequenceBackbone):

    def __init__(self, input_dim: int, hidden_dim: int, state_dim: int | None = None):
        super().__init__()
        state_dim = state_dim or hidden_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.to_gate = nn.Linear(hidden_dim, state_dim)
        self.to_input = nn.Linear(hidden_dim, state_dim)
        self.log_decay = nn.Parameter(torch.zeros(state_dim))
        self.out = nn.Linear(state_dim, hidden_dim)
        self.skip = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.output_dim = hidden_dim
        self.last_hidden_rms = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.input_proj(x)
        state = torch.zeros(x.shape[0], self.to_gate.out_features, device=x.device, dtype=x.dtype)
        decay_base = torch.sigmoid(self.log_decay)
        outputs = []
        for t in range(x.shape[1]):
            gate = torch.sigmoid(self.to_gate(z[:, t]))
            inp = torch.tanh(self.to_input(z[:, t]))
            decay = decay_base.unsqueeze(0) * (1.0 - 0.5 * gate)
            state = decay * state + gate * inp
            outputs.append(self.norm(self.out(state) + self.skip(z[:, t])))
        out = torch.stack(outputs, dim=1)
        self.last_hidden_rms = float(out.detach().pow(2).mean().sqrt().cpu())
        return out


class NeuralODEBackbone(SequenceBackbone):

    def __init__(self, input_dim: int, hidden_dim: int, dt: float = 0.05):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.field = nn.Sequential(
            nn.Linear(2 * hidden_dim, 2 * hidden_dim),
            nn.Tanh(),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.dt = float(dt)
        self.output_dim = hidden_dim
        self.last_hidden_rms = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        forcing = self.input_proj(x)
        state = torch.zeros(x.shape[0], self.output_dim, device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(x.shape[1]):
            derivative = self.field(torch.cat([state, forcing[:, t]], dim=-1))
            state = state + self.dt * derivative
            outputs.append(state)
        out = torch.stack(outputs, dim=1)
        self.last_hidden_rms = float(out.detach().pow(2).mean().sqrt().cpu())
        return out


class CausalTransformerBackbone(SequenceBackbone):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        layers: int = 3,
        nhead: int = 4,
        dropout: float = 0.1,
        max_len: int = 4096,
    ):
        super().__init__()
        if hidden_dim % nhead != 0:
            raise ValueError("hidden_dim must be divisible by nhead")
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        positions = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, hidden_dim, 2, dtype=torch.float32) * (-math.log(10000.0) / hidden_dim))
        pe = torch.zeros(max_len, hidden_dim)
        pe[:, 0::2] = torch.sin(positions * div)
        pe[:, 1::2] = torch.cos(positions * div[: pe[:, 1::2].shape[1]])
        self.register_buffer("positional_encoding", pe, persistent=False)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=4 * hidden_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.core = nn.TransformerEncoder(layer, num_layers=layers)
        self.output_dim = hidden_dim
        self.last_hidden_rms = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] > self.positional_encoding.shape[0]:
            raise ValueError("Sequence exceeds configured Transformer max_len")
        z = self.input_proj(x) + self.positional_encoding[: x.shape[1]].to(x.dtype).unsqueeze(0)
        mask = torch.triu(torch.ones(x.shape[1], x.shape[1], device=x.device, dtype=torch.bool), diagonal=1)
        out = self.core(z, mask=mask)
        self.last_hidden_rms = float(out.detach().pow(2).mean().sqrt().cpu())
        return out


def build_backbone(name: str, input_dim: int, cfg: dict[str, Any], seed: int = 0) -> SequenceBackbone:
    name = name.lower()
    hidden = int(cfg.get("hidden_dim", 128))
    layers = int(cfg.get("layers", 2))
    if name == "gru":
        return GRUBackbone(input_dim, hidden, layers, float(cfg.get("dropout", 0.0)))
    if name == "lstm":
        return LSTMBackbone(input_dim, hidden, layers, float(cfg.get("dropout", 0.0)))
    if name in {"rnn", "rnn_vanilla"}:
        return RNNBackbone(input_dim, hidden, layers)
    if name == "rnn_orthogonal":
        return RNNBackbone(input_dim, hidden, layers, orthogonal_init=True)
    if name == "rnn_layernorm":
        return RNNBackbone(input_dim, hidden, layers, layer_norm=True)
    if name == "rnn_residual":
        return RNNBackbone(input_dim, hidden, layers, orthogonal_init=True, layer_norm=True, residual=True)
    if name == "tcn":
        return TCNBackbone(
            input_dim,
            hidden,
            layers=int(cfg.get("tcn_layers", 5)),
            kernel_size=int(cfg.get("kernel_size", 5)),
            dropout=float(cfg.get("dropout", 0.1)),
            gated=bool(cfg.get("gated", True)),
        )
    if name == "esn":
        return ESNBackbone(
            input_dim,
            hidden,
            spectral_radius=float(cfg.get("spectral_radius", 0.9)),
            leak_rate=float(cfg.get("leak_rate", 0.5)),
            sparsity=float(cfg.get("sparsity", 0.9)),
            seed=seed,
        )
    if name == "ssm":
        return GatedLinearSSMBackbone(input_dim, hidden, state_dim=int(cfg.get("d_state", hidden)))
    if name == "neural_ode":
        return NeuralODEBackbone(input_dim, hidden, dt=float(cfg.get("ode_dt", 0.05)))
    if name == "transformer":
        return CausalTransformerBackbone(
            input_dim,
            hidden,
            layers=int(cfg.get("transformer_layers", 3)),
            nhead=int(cfg.get("nhead", 4)),
            dropout=float(cfg.get("dropout", 0.1)),
            max_len=int(cfg.get("max_len", 4096)),
        )
    raise ValueError(f"Unknown backbone: {name}")

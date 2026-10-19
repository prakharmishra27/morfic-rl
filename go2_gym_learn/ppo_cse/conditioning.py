"""Morphology-conditioned actor and critic modules."""

import torch
import torch.nn as nn


class FiLM(nn.Module):
    """Generate residual feature scales and shifts."""

    def __init__(self, cond_dim, hidden_dim, scale=0.1):
        super().__init__()
        self.scale = scale
        self.net = nn.Sequential(
            nn.Linear(cond_dim, 128),
            nn.ELU(),
            nn.Linear(128, 2 * hidden_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, cond):
        gamma, beta = (self.net(cond) * self.scale).chunk(2, dim=-1)
        return gamma, beta


class CriticFiLM(nn.Module):
    """Apply FiLM before activation at every critic hidden layer."""

    def __init__(
        self, input_dim, hidden_dims, cond_dim,
        activation=nn.ELU, film_scale=0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        self.films = nn.ModuleList()

        last = input_dim
        for width in hidden_dims:
            self.layers.append(nn.Linear(last, width))
            self.films.append(FiLM(cond_dim, width, scale=film_scale))
            last = width

        self.act = activation()
        self.v_head = nn.Linear(last, 1)

    def forward(self, x, cond):
        h = x
        for linear, film in zip(self.layers, self.films):
            h = linear(h)
            gamma, beta = film(cond)
            h = self.act(h * (1.0 + gamma) + beta)
        return self.v_head(h)


class ActorFiLM(nn.Module):
    """Apply FiLM before activation at every actor hidden layer."""

    def __init__(
        self, input_dim, hidden_dims, cond_dim, num_actions,
        activation=nn.ELU, film_scale=0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        self.films = nn.ModuleList()

        last = input_dim
        for width in hidden_dims:
            self.layers.append(nn.Linear(last, width))
            self.films.append(FiLM(cond_dim, width, scale=film_scale))
            last = width

        self.act = activation()
        self.action_head = nn.Linear(last, num_actions)

    def forward(self, x, cond):
        h = x
        for linear, film in zip(self.layers, self.films):
            h = linear(h)
            gamma, beta = film(cond)
            h = self.act(h * (1.0 + gamma) + beta)
        return self.action_head(h)


class CriticAdaLN(nn.Module):
    """Apply LayerNorm followed by residual FiLM."""

    def __init__(
        self, input_dim, hidden_dims, cond_dim,
        activation=nn.ELU, film_scale=0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.films = nn.ModuleList()

        last = input_dim
        for width in hidden_dims:
            self.layers.append(nn.Linear(last, width))
            self.norms.append(nn.LayerNorm(width))
            self.films.append(FiLM(cond_dim, width, scale=film_scale))
            last = width

        self.act = activation()
        self.v_head = nn.Linear(last, 1)

    def forward(self, x, cond):
        h = x
        for linear, norm, film in zip(
            self.layers, self.norms, self.films
        ):
            h = norm(linear(h))
            gamma, beta = film(cond)
            h = self.act(h * (1.0 + gamma) + beta)
        return self.v_head(h)


class HyperLinearAdapter(nn.Module):
    """Generate a scaled low-rank adapter plus conditional bias."""

    def __init__(
        self, cond_dim, in_dim, out_dim, rank=16, scale=0.05,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.rank = rank
        self.scale = scale

        self.to_A = nn.Linear(cond_dim, rank * in_dim)
        self.to_B = nn.Linear(cond_dim, out_dim * rank)
        self.to_bias = nn.Linear(cond_dim, out_dim)

        nn.init.normal_(self.to_A.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.to_A.bias)
        nn.init.zeros_(self.to_B.weight)
        nn.init.zeros_(self.to_B.bias)
        nn.init.zeros_(self.to_bias.weight)
        nn.init.zeros_(self.to_bias.bias)

    def forward(self, h, cond):
        batch_size = h.shape[0]
        a = self.to_A(cond).view(
            batch_size, self.rank, self.in_dim
        )
        b = self.to_B(cond).view(
            batch_size, self.out_dim, self.rank
        )
        bias = self.to_bias(cond)

        hidden = torch.bmm(a, h.unsqueeze(-1)).squeeze(-1)
        out = torch.bmm(b, hidden.unsqueeze(-1)).squeeze(-1)
        return self.scale * (out + bias)


class CriticHyperNet(nn.Module):
    """Apply low-rank residual adapters at every critic hidden layer."""

    def __init__(
        self, input_dim, hidden_dims, cond_dim,
        activation=nn.ELU, rank=16, hyper_scale=0.05,
        use_layernorm=True,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        self.adapters = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.use_layernorm = use_layernorm

        last = input_dim
        for width in hidden_dims:
            self.layers.append(nn.Linear(last, width))
            self.adapters.append(
                HyperLinearAdapter(
                    cond_dim=cond_dim,
                    in_dim=last,
                    out_dim=width,
                    rank=rank,
                    scale=hyper_scale,
                )
            )
            self.norms.append(nn.LayerNorm(width))
            last = width

        self.act = activation()
        self.v_head = nn.Linear(last, 1)

    def forward(self, x, cond):
        h = x
        for linear, adapter, norm in zip(
            self.layers, self.adapters, self.norms
        ):
            h = linear(h) + adapter(h, cond)
            if self.use_layernorm:
                h = norm(h)
            h = self.act(h)
        return self.v_head(h)
"""Model definitions for Push-T imitation policies."""

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

import torch
from torch import nn
from torch.nn import functional as F


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    def is_state_legal(self, state: torch.Tensor) -> bool:
        # State must be of (B, 5)
        assert state.dim() == 2
        assert state.size(1) == self.state_dim

        return True

    def is_action_legal(self, action_chunk: torch.Tensor) -> bool:
        # Action_chunk must be of (B, self.chunk_size, 2)
        assert action_chunk.dim() == 3
        assert action_chunk.size(1) == self.chunk_size
        assert action_chunk.size(2) == self.action_dim

        return True

    @abc.abstractmethod
    def compute_loss(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        """Compute training loss for a batch."""

    @abc.abstractmethod
    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,  # only applicable for flow policy
    ) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""


class MSEPolicy(BasePolicy):
    """Predicts action chunks with an MSE loss."""

    ### TODO: IMPLEMENT MSEPolicy HERE ###
    # Model input is state, of shape (B, 5), and model output is action, of shape(B, chunk_size, 2).
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        self.input_layer = nn.Linear(state_dim, hidden_dims[0])
        self.output_layer = nn.Linear(hidden_dims[-1], action_dim * chunk_size)
        self.activation = nn.ReLU()
        self.middle_layers: list[nn.Module] = []

        for i in range(1, len(hidden_dims), 1):
            self.middle_layers.append(nn.Linear(hidden_dims[i - 1], hidden_dims[i]))
            self.middle_layers.append(self.activation)

        self.model = nn.Sequential(
            self.input_layer,
            self.activation,
            *self.middle_layers,
            self.output_layer
        )

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:

        self.is_state_legal(state)
        self.is_action_legal(action_chunk)

        pred = self.model(state) # (B, self.chunk_size * self.action_dim)
        pred = pred.view(-1, self.chunk_size, self.action_dim) # (B, self.chunk_size, self.action_dim)

        loss = F.mse_loss(pred, action_chunk)

        return loss

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:

        self.is_state_legal(state)
        
        pred = self.model(state)
        return pred.view(-1, self.chunk_size, self.action_dim) # (B, self.chunk_size, self.action_dim)


class FlowMatchingPolicy(BasePolicy):
    """Predicts action chunks with a flow matching loss."""

    ### TODO: IMPLEMENT FlowMatchingPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)

        # Same structure of MSEpolicy
        self.activation = nn.ReLU()
        self.layers = []

        in_dim = state_dim + action_dim * chunk_size + 1
        out_dim = None
        for hidden_dim in hidden_dims:
            out_dim = hidden_dim
            self.layers.append(nn.Linear(in_dim, out_dim))
            self.layers.append(self.activation)
            in_dim = hidden_dim

        self.layers.append(nn.Linear(in_dim, action_dim * chunk_size))

        self.model = nn.Sequential(*self.layers)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
    
        self.is_state_legal(state) # (B, state_dim)
        self.is_action_legal(action_chunk) # (B, chunk_size, action_dim)

        B = action_chunk.size(0)

        x_1 = action_chunk.view(-1, self.chunk_size * self.action_dim) # (B, D) (D = chunk_size * action_dim)
        # 1. 生成噪声
        x_0 = torch.randn_like(x_1, device=state.device)
        t = torch.rand((B, 1), device=state.device)

        # 2. 构造x_t
        x_t = (1 - t) * x_0 + t * x_1 # (B, D)
        v_target = x_1 - x_0

        # 3. 输入网络
        pred = self.model(torch.concat((state, x_t, t), dim=-1))
        loss = nn.functional.mse_loss(pred, v_target)

        return loss
        
    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        
        self.is_state_legal(state) # (B, state_dim)

        B = state.size(0)
        D = self.action_dim * self.chunk_size
        x_0 = torch.randn(B, D, device=state.device)
        x_t_1 = x_0

        delta_t = 1 / num_steps
        for i in range(num_steps):
            t = i / num_steps
            B_t = torch.ones((B, 1), device=state.device) * t
            x_t = x_t_1 + delta_t * self.model(torch.cat((state, x_t_1, B_t), dim=-1))

            x_t_1 = x_t

        x_t = x_t.view(B, self.chunk_size, self.action_dim)

        return x_t


PolicyType: TypeAlias = Literal["mse", "flow"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    hidden_dims: tuple[int, ...] = (128, 128),
) -> BasePolicy:
    if policy_type == "mse":
        return MSEPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    if policy_type == "flow":
        return FlowMatchingPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    raise ValueError(f"Unknown policy type: {policy_type}")

from typing import Sequence, Callable, Tuple, Optional

import torch
from torch import nn

import numpy as np

from infrastructure import pytorch_util as ptu


class DQNAgent(nn.Module):
    def __init__(
        self,
        observation_shape: Sequence[int],
        num_actions: int,
        make_critic: Callable[[Tuple[int, ...], int], nn.Module],
        make_optimizer: Callable[[torch.nn.ParameterList], torch.optim.Optimizer],
        make_lr_schedule: Callable[
            [torch.optim.Optimizer], torch.optim.lr_scheduler._LRScheduler
        ],
        discount: float,
        target_update_period: int,
        use_double_q: bool = False,
        clip_grad_norm: Optional[float] = None,
    ):
        super().__init__()

        self.critic = make_critic(observation_shape, num_actions)
        self.target_critic = make_critic(observation_shape, num_actions)
        self.critic_optimizer = make_optimizer(self.critic.parameters())
        self.lr_scheduler = make_lr_schedule(self.critic_optimizer)

        self.observation_shape = observation_shape
        self.num_actions = num_actions
        self.discount = discount
        self.target_update_period = target_update_period
        self.clip_grad_norm = clip_grad_norm
        self.use_double_q = use_double_q

        self.critic_loss = nn.MSELoss()

        self.update_target_critic()

    @torch.no_grad()
    def get_action(self, observation: np.ndarray, epsilon: float = 0.0) -> int:
        """
        Epsilon-greedy action selection (default epsilon=0 for deterministic/greedy policy).
        """
        observation = ptu.from_numpy(np.asarray(observation))[None]

        # TODO(Section 2.4): get the action from the critic using an epsilon-greedy strategy
        qa_values = self.critic(observation) # logits.shape = (B, action_dim)
        greedy = torch.argmax(qa_values, dim=-1, keepdim=True)

        n_action = qa_values.size(-1)

        probs = torch.full_like(qa_values, fill_value=epsilon / n_action, device=qa_values.device)
        B = qa_values.size(0)
        greedy_prob = torch.full(size=(B, 1), fill_value=1 - (n_action - 1) * epsilon / n_action, device=qa_values.device)
        probs.scatter_(dim=-1, index=greedy, src=greedy_prob)

        dist = torch.distributions.Categorical(probs=probs)

        action = dist.sample()
        # ENDTODO

        return ptu.to_numpy(action).squeeze(0).item()

    def update_critic(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
    ) -> dict:
        """Update the DQN critic, and return stats for logging."""
        (batch_size,) = reward.shape

        # Compute target values
        with torch.no_grad():
            # TODO(Section 2.4): compute target values

            next_qa_values = self.target_critic(next_obs)

            if self.use_double_q:
                # TODO(Section 2.5): implement double-Q target action selection
                next_action = torch.argmax(self.critic(next_obs), dim=-1, keepdim=True)
            else:
                next_action = torch.argmax(next_qa_values, dim=-1, keepdim=True)

            next_q_values = torch.gather(next_qa_values, dim=-1, index=next_action).squeeze(1)
            assert next_q_values.shape == (batch_size,), next_q_values.shape

            target_values = reward + self.discount * next_q_values * (1 - done.float())
            assert target_values.shape == (batch_size,), target_values.shape
            # ENDTODO

        # TODO(Section 2.4): train the critic with the target values
        qa_values = self.critic(obs)
        q_values = torch.gather(qa_values, dim=-1, index=action.long().unsqueeze(1)).squeeze(1)
        loss = self.critic_loss(q_values, target_values)
        # ENDTODO

        self.critic_optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad.clip_grad_norm_(
            self.critic.parameters(), self.clip_grad_norm or float("inf")
        )
        self.critic_optimizer.step()

        self.lr_scheduler.step()

        return {
            "critic_loss": loss.item(),
            "q_values": q_values.mean().item(),
            "target_values": target_values.mean().item(),
            "grad_norm": grad_norm.item(),
        }

    def update_target_critic(self):
        self.target_critic.load_state_dict(self.critic.state_dict())

    def update(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
        step: int,
    ) -> dict:
        """
        Update the DQN agent, including both the critic and target.
        """
        # TODO(Section 2.4): update the critic, and the target if needed
        critic_stats = self.update_critic(obs, action, reward, next_obs, done)
        if step % self.target_update_period == 0:
            self.update_target_critic()
        # Hint: if step % self.target_update_period == 0: ...
        # ENDTODO

        return critic_stats

"""
I have implemented and reviewed this module structure.
"""

# I have implemented this module-level note to keep the flow human-readable.
from __future__ import annotations
import math
import pickle
import random
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    _TORCH = True
except ImportError:
    _TORCH = False
    print("[DQN] WARNING: PyTorch not found. DQN requires PyTorch.")

from .replay_buffer import PrioritizedReplayBuffer, UniformReplayBuffer
from .n_step_buffer import NStepBuffer


class _QNetwork(nn.Module):


    # I have implemented this callable with parameters: self, state_size, action_size, hidden.
    def __init__(self, state_size: int, action_size: int,
                 hidden: list[int] = None):


        """
        I have implemented this function with a clearer note.
        Parameters used here: self, state_size, action_size, hidden.
        """
        super().__init__()
        hidden = hidden or [256, 128]
        layers = []
        in_size = state_size
        for h in hidden:
            layers.extend([nn.Linear(in_size, h), nn.ReLU()])
            in_size = h
        layers.append(nn.Linear(in_size, action_size))
        self.net = nn.Sequential(*layers)
        self._init_weights()


    # I have implemented this callable with parameters: self.
    def _init_weights(self) -> None:


        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)


    # I have implemented this callable with parameters: self, x.
    def forward(self, x: "torch.Tensor") -> "torch.Tensor":


        """
        I have implemented this function with a clearer note.
        Parameters used here: self, x.
        """
        return self.net(x)


class AdvancedDQNAgent:


    # I have implemented this callable with parameters: self, env, hidden, lr, gamma, epsilon_start, epsilon_min, decay_steps, batch_size, buffer_size, target_update, n_step, per_alpha, per_beta_start, per_beta_steps, use_per, grad_clip.
    def __init__(
        self,
        env,
        hidden: list[int] = None,
        lr: float = 5e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.05,
        decay_steps: int = 50_000,
        batch_size: int = 64,
        buffer_size: int = 50_000,
        target_update: int = 500,
        n_step: int = 3,
        per_alpha: float = 0.6,
        per_beta_start: float = 0.4,
        per_beta_steps: int = 100_000,
        use_per: bool = True,
        grad_clip: float | None = 10.0,
    ):


        if not _TORCH:
            raise RuntimeError("PyTorch required for AdvancedDQNAgent.")

        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_start = epsilon_start
        self.decay_steps = decay_steps
        self.batch_size = batch_size
        self.target_update = target_update
        self.n_step = n_step
        self.grad_clip = grad_clip
        self.gamma_n = gamma ** n_step
        self.step_count = 0


        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")


        hidden = hidden or [256, 128]
        self.online = _QNetwork(env.STATE_SIZE, env.N_ACTIONS, hidden).to(self.device)
        self.target_net = _QNetwork(env.STATE_SIZE, env.N_ACTIONS, hidden).to(self.device)
        self.target_net.load_state_dict(self.online.state_dict())
        self.target_net.eval()


        self.optimiser = optim.Adam(self.online.parameters(), lr=lr)


        if use_per:
            self.buffer = PrioritizedReplayBuffer(
                buffer_size, alpha=per_alpha,
                beta_start=per_beta_start,
                beta_end=1.0,
                beta_steps=per_beta_steps,
            )
        else:
            self.buffer = UniformReplayBuffer(buffer_size)
        self._use_per = use_per


        self.n_step_buf = NStepBuffer(n=n_step, gamma=gamma)


        self.episode_rewards: list[float] = []
        self.losses: list[float] = []
        self.win_rates: list[float] = []


    @staticmethod


    # I have implemented this callable with parameters: tensor.
    def _to_numpy(tensor) -> np.ndarray:


        """
        I have implemented this function with a clearer note.
        Parameters used here: tensor.
        """
        t = tensor.detach().cpu()
        try:
            return t.numpy()
        except RuntimeError:
            return np.array(t.tolist(), dtype=np.float32)


    # I have implemented this callable with parameters: self, state, legal_actions.
    def choose_action(self, state: np.ndarray, legal_actions: list[int]) -> int:


        if random.random() < self.epsilon:
            return random.choice(legal_actions)
        return self._greedy(state, legal_actions)


    # I have implemented this callable with parameters: self, state, legal_actions.
    def _greedy(self, state: np.ndarray, legal_actions: list[int]) -> int:


        """
        I have implemented this function with a clearer note.
        Parameters used here: self, state, legal_actions.
        """
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_vals = self._to_numpy(self.online(s).squeeze(0))
        mask = np.full(self.env.N_ACTIONS, -np.inf)
        mask[legal_actions] = q_vals[legal_actions]
        return int(np.argmax(mask))


    # I have implemented this callable with parameters: self.
    def _cosine_epsilon(self) -> None:


        frac = min(self.step_count / max(self.decay_steps, 1), 1.0)
        self.epsilon = (self.epsilon_min +
                        0.5 * (self.epsilon_start - self.epsilon_min) *
                        (1.0 + math.cos(math.pi * frac)))


    # I have implemented this callable with parameters: self.
    def _train_step(self) -> float | None:


        """
        I have implemented this function with a clearer note.
        Parameters used here: self.
        """
        if len(self.buffer) < self.batch_size:
            return None

        (states, actions, rewards, next_states, dones,
         next_legals, is_weights, indices) = self.buffer.sample(self.batch_size)

        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)
        weights_t = torch.FloatTensor(is_weights).to(self.device)


        q_pred_all = self.online(states_t)
        q_pred = q_pred_all.gather(1, actions_t.unsqueeze(1)).squeeze(1)


        with torch.no_grad():

            q_online_next = self._to_numpy(self.online(next_states_t))

            q_target_next = self._to_numpy(self.target_net(next_states_t))

        td_targets = np.zeros(self.batch_size, dtype=np.float32)
        for i in range(self.batch_size):
            if dones[i]:
                td_targets[i] = rewards[i]
            else:
                nl = next_legals[i]
                if nl:

                    q_on = q_online_next[i].copy()
                    mask = np.full(self.env.N_ACTIONS, -np.inf)
                    mask[nl] = q_on[nl]
                    best_a = int(np.argmax(mask))

                    td_targets[i] = (rewards[i] +
                                     self.gamma_n * q_target_next[i][best_a])
                else:
                    td_targets[i] = rewards[i]

        targets_t = torch.FloatTensor(td_targets).to(self.device)


        td_errors = self._to_numpy((targets_t - q_pred).detach())

        loss = (weights_t * (targets_t - q_pred) ** 2).mean()


        self.optimiser.zero_grad()
        loss.backward()
        if self.grad_clip is not None:
            nn.utils.clip_grad_norm_(self.online.parameters(), self.grad_clip)
        self.optimiser.step()


        if self._use_per and indices is not None:
            self.buffer.update_priorities(indices, np.abs(td_errors))

        return float(loss.item())


    # I have implemented this callable with parameters: self, n_episodes, eval_every, eval_episodes, reward_shaping, shaping_weight, shaping_clip, callback, stop_flag.
    def train(
        self,
        n_episodes: int = 50_000,
        eval_every: int = 2_000,
        eval_episodes: int = 200,
        reward_shaping: bool = False,
        shaping_weight: float = 0.01,
        shaping_clip: float = 0.1,
        callback=None,
        stop_flag=None,
    ) -> dict:


        for ep in range(1, n_episodes + 1):
            if stop_flag and stop_flag.is_set():
                break

            state = self.env.reset()
            total_r = 0.0
            self.n_step_buf.clear()
            prev_heuristic = 0.0 if reward_shaping else 0.0

            while True:
                legal = self.env.get_legal_actions()
                action = self.choose_action(state, legal)

                if reward_shaping:
                    h_before = self.env.get_board_heuristic()

                next_state, reward, done, _ = self.env.step(action)
                next_legal = self.env.get_legal_actions()


                if reward_shaping and not done:
                    h_after = self.env.get_board_heuristic()
                    shaped = np.clip(shaping_weight * (h_after - h_before),
                                     -shaping_clip, shaping_clip)
                    reward += float(shaped)


                ready = self.n_step_buf.push(
                    state, action, reward, next_state, done, next_legal)
                for transition in ready:
                    self.buffer.push(*transition)


                loss = self._train_step()
                if loss is not None:
                    self.losses.append(loss)

                self.step_count += 1
                self._cosine_epsilon()


                if self.step_count % self.target_update == 0:
                    self.target_net.load_state_dict(self.online.state_dict())

                total_r += reward
                state = next_state

                if done:

                    for transition in self.n_step_buf.flush():
                        self.buffer.push(*transition)
                    break

            self.episode_rewards.append(total_r)

            if ep % eval_every == 0:
                wr = self.evaluate(eval_episodes)
                self.win_rates.append(wr)
                avg_loss = (float(np.mean(self.losses[-500:]))
                            if self.losses else 0.0)
                if callback:
                    callback(ep, wr, self.epsilon, avg_loss)

        return {
            "episode_rewards": self.episode_rewards,
            "losses": self.losses,
            "win_rates": self.win_rates,
        }


    # I have implemented this callable with parameters: self, n_episodes.
    def evaluate(self, n_episodes: int = 200) -> float:


        """
        I have implemented this function with a clearer note.
        Parameters used here: self, n_episodes.
        """
        saved_eps = self.epsilon
        self.epsilon = 0.0
        wins = 0
        for _ in range(n_episodes):
            state = self.env.reset()
            while True:
                legal = self.env.get_legal_actions()
                action = self._greedy(state, legal)
                state, reward, done, _ = self.env.step(action)
                if done:
                    if reward >= 1.0:
                        wins += 1
                    break
        self.epsilon = saved_eps
        return wins / n_episodes


    # I have implemented this callable with parameters: self, state, legal_actions.
    def predict(self, state: np.ndarray, legal_actions: list[int]) -> int:


        return self._greedy(state, legal_actions)


    # I have implemented this callable with parameters: self, path.
    def save(self, path: str) -> None:


        """
        I have implemented this function with a clearer note.
        Parameters used here: self, path.
        """
        data = {
            "online_state_dict": self.online.state_dict(),
            "target_state_dict": self.target_net.state_dict(),
            "epsilon": self.epsilon,
            "step_count": self.step_count,
            "episode_rewards": self.episode_rewards,
            "losses": self.losses,
            "win_rates": self.win_rates,
            "state_size": self.env.STATE_SIZE,
            "action_size": self.env.N_ACTIONS,
        }
        torch.save(data, path)


    # I have implemented this callable with parameters: self, path.
    def load(self, path: str) -> None:


        data = torch.load(path, map_location=self.device)
        self.online.load_state_dict(data["online_state_dict"])
        self.target_net.load_state_dict(data["target_state_dict"])
        self.epsilon = data.get("epsilon", self.epsilon_min)
        self.step_count = data.get("step_count", 0)
        self.episode_rewards = data.get("episode_rewards", [])
        self.losses = data.get("losses", [])
        self.win_rates = data.get("win_rates", [])
        self.online.eval()
        self.target_net.eval()


    # I have implemented this callable with parameters: self, eps.
    def set_epsilon(self, eps: float) -> None:


        """
        I have implemented this function with a clearer note.
        Parameters used here: self, eps.
        """
        self.epsilon = eps


    # I have implemented this callable with parameters: self.
    def info(self) -> dict:


        return {
            "agent": "AdvancedDQN",
            "state_size": self.env.STATE_SIZE,
            "action_size": self.env.N_ACTIONS,
            "epsilon": self.epsilon,
            "step_count": self.step_count,
            "buffer_size": len(self.buffer),
            "episodes_trained": len(self.episode_rewards),
            "avg_loss_recent": (float(np.mean(self.losses[-200:]))
                                if len(self.losses) >= 200 else
                                float(np.mean(self.losses)) if self.losses else 0.0),
            "n_step": self.n_step,
            "use_per": self._use_per,
            "double_dqn": True,
        }

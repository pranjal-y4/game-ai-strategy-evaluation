from __future__ import annotations
import random
import pickle
from collections import defaultdict, deque


class AdvancedQLearning:


    def __init__(
        self,
        env,
        lr: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.05,
        n_step: int = 3,
    ):
        self.env = env
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.n_step = n_step


        self.q_table: dict = defaultdict(lambda: defaultdict(float))


        self.episode_rewards: list[float] = []
        self.win_rates: list[float] = []


    def get_q(self, state: tuple, action: int) -> float:
        return self.q_table[state][action]

    def best_q(self, state: tuple, legal_actions: list[int]) -> float:
        if not legal_actions:
            return 0.0
        return max(self.q_table[state][a] for a in legal_actions)

    def update(self, state: tuple, action: int, G: float,
               next_state: tuple, done: bool, next_legal: list[int],
               gamma_n: float) -> None:


        q_cur = self.q_table[state][action]
        if done or not next_legal:
            target = G
        else:
            target = G + gamma_n * self.best_q(next_state, next_legal)
        self.q_table[state][action] = q_cur + self.lr * (target - q_cur)


    def choose_action(self, state: tuple, legal_actions: list[int]) -> int:
        if not legal_actions:
            raise ValueError("No legal actions available.")
        if random.random() < self.epsilon:
            return random.choice(legal_actions)
        return max(legal_actions, key=lambda a: self.q_table[state][a])

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min,
                           self.epsilon * self.epsilon_decay)


    def train(
        self,
        n_episodes: int = 50_000,
        eval_every: int = 2_000,
        eval_episodes: int = 200,
        callback=None,
        stop_flag=None,
    ) -> dict:


        gamma_n = self.gamma ** self.n_step
        n_buf_s: deque = deque()
        n_buf_ns: deque = deque()

        for ep in range(1, n_episodes + 1):
            if stop_flag and stop_flag.is_set():
                break

            state_arr = self.env.reset()
            state = self.env.encode_state()
            total_r = 0.0
            n_buf_s.clear()
            n_buf_ns.clear()
            ep_transitions = []


            while True:
                legal = self.env.get_legal_actions()
                action = self.choose_action(state, legal)
                _, reward, done, _ = self.env.step(action)
                next_state = self.env.encode_state()
                next_legal = self.env.get_legal_actions()

                ep_transitions.append(
                    (state, action, reward, next_state, done, next_legal))
                total_r += reward
                state = next_state
                if done:
                    break


            T = len(ep_transitions)
            for t in range(T):
                s_t, a_t, _, _, _, _ = ep_transitions[t]


                G = 0.0
                actual_n = 0
                for i in range(self.n_step):
                    if t + i >= T:
                        break
                    _, _, r_i, _, done_i, _ = ep_transitions[t + i]
                    G += (self.gamma ** i) * r_i
                    actual_n = i + 1
                    if done_i:
                        break


                idx_n = t + actual_n - 1
                _, _, _, ns_n, done_n, legal_n = ep_transitions[idx_n]
                gn = self.gamma ** actual_n
                self.update(s_t, a_t, G, ns_n, done_n, legal_n, gn)

            self.decay_epsilon()
            self.episode_rewards.append(total_r)

            if ep % eval_every == 0:
                wr = self.evaluate(eval_episodes)
                self.win_rates.append(wr)
                if callback:
                    callback(ep, wr, self.epsilon)

        return {
            "episode_rewards": self.episode_rewards,
            "win_rates": self.win_rates,
            "q_table_size": sum(len(v) for v in self.q_table.values()),
        }


    def evaluate(self, n_episodes: int = 200) -> float:

        saved = self.epsilon
        self.epsilon = 0.0
        wins = 0
        for _ in range(n_episodes):
            self.env.reset()
            state = self.env.encode_state()
            while True:
                legal = self.env.get_legal_actions()
                action = self.choose_action(state, legal)
                _, reward, done, _ = self.env.step(action)
                state = self.env.encode_state()
                if done:
                    if reward >= 1.0:
                        wins += 1
                    break
        self.epsilon = saved
        return wins / n_episodes

    def predict(self, state: tuple, legal_actions: list[int]) -> int:

        if not legal_actions:
            return -1
        return max(legal_actions, key=lambda a: self.q_table[state][a])


    def save(self, path: str) -> None:
        data = {
            "q_table": {k: dict(v) for k, v in self.q_table.items()},
            "epsilon": self.epsilon,
            "episode_rewards": self.episode_rewards,
            "win_rates": self.win_rates,
            "n_step": self.n_step,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.q_table = defaultdict(lambda: defaultdict(float))
        for state, actions in data["q_table"].items():
            for a, q in actions.items():
                self.q_table[state][a] = q
        self.epsilon = data.get("epsilon", self.epsilon_min)
        self.episode_rewards = data.get("episode_rewards", [])
        self.win_rates = data.get("win_rates", [])

    def info(self) -> dict:
        return {
            "agent": "AdvancedQLearning",
            "lr": self.lr,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "n_step": self.n_step,
            "q_table_states": len(self.q_table),
            "q_table_entries": sum(len(v) for v in self.q_table.values()),
            "episodes_trained": len(self.episode_rewards),
        }

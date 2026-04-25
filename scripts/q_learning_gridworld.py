"""Reproducible Gridworld Q-learning experiments for the take-home report.

This is an original, clean-room implementation based on the assignment's
published environment description and Q-learning update equation. It does not
copy code from the referenced GitHub notebook.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple
import argparse
import csv
import random

State = Tuple[int, int]
Action = str
ACTIONS: tuple[Action, ...] = ("UP", "DOWN", "LEFT", "RIGHT")
DELTAS: dict[Action, State] = {
    "UP": (-1, 0),
    "DOWN": (1, 0),
    "LEFT": (0, -1),
    "RIGHT": (0, 1),
}


@dataclass(frozen=True)
class GridworldConfig:
    height: int = 5
    width: int = 5
    gold: State = (0, 3)
    bomb: State = (1, 3)
    slip_probability: float = 0.0
    max_steps: int = 200


class Gridworld:
    def __init__(self, config: GridworldConfig, rng: random.Random):
        self.config = config
        self.rng = rng
        self.current_state = self.reset()

    def reset(self) -> State:
        self.current_state = (self.config.height - 1, self.rng.randrange(self.config.width))
        return self.current_state

    def _move(self, state: State, action: Action) -> State:
        dr, dc = DELTAS[action]
        nr = min(max(state[0] + dr, 0), self.config.height - 1)
        nc = min(max(state[1] + dc, 0), self.config.width - 1)
        return nr, nc

    def step(self, action: Action) -> tuple[State, float, bool]:
        if self.rng.random() < self.config.slip_probability:
            alternatives = [candidate for candidate in ACTIONS if candidate != action]
            action = self.rng.choice(alternatives)

        next_state = self._move(self.current_state, action)
        self.current_state = next_state

        reward = -1.0
        done = False
        if next_state == self.config.gold:
            reward += 10.0
            done = True
        elif next_state == self.config.bomb:
            reward -= 10.0
            done = True
        return next_state, reward, done


class QAgent:
    def __init__(
        self,
        states: Iterable[State],
        epsilon: float,
        alpha: float,
        gamma: float,
        initial_q: float,
        rng: random.Random,
    ):
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.rng = rng
        self.q: Dict[State, Dict[Action, float]] = {
            state: {action: initial_q for action in ACTIONS} for state in states
        }

    def choose_action(self, state: State) -> Action:
        if self.rng.random() < self.epsilon:
            return self.rng.choice(ACTIONS)
        values = self.q[state]
        best = max(values.values())
        return self.rng.choice([action for action, value in values.items() if value == best])

    def learn(self, old_state: State, reward: float, new_state: State, action: Action) -> None:
        current = self.q[old_state][action]
        target = reward + self.gamma * max(self.q[new_state].values())
        self.q[old_state][action] = (1 - self.alpha) * current + self.alpha * target


def all_states(config: GridworldConfig) -> list[State]:
    return [(r, c) for r in range(config.height) for c in range(config.width)]


def train_once(
    *,
    seed: int,
    episodes: int,
    epsilon: float,
    alpha: float,
    gamma: float,
    initial_q: float,
    slip_probability: float,
) -> tuple[QAgent, list[float], list[int]]:
    rng = random.Random(seed)
    config = GridworldConfig(slip_probability=slip_probability)
    env = Gridworld(config, rng)
    agent = QAgent(all_states(config), epsilon, alpha, gamma, initial_q, rng)
    rewards: list[float] = []
    steps: list[int] = []

    for _ in range(episodes):
        state = env.reset()
        total = 0.0
        for step in range(1, config.max_steps + 1):
            action = agent.choose_action(state)
            new_state, reward, done = env.step(action)
            agent.learn(state, reward, new_state, action)
            total += reward
            state = new_state
            if done:
                break
        rewards.append(total)
        steps.append(step)
    return agent, rewards, steps


def greedy_policy(agent: QAgent, config: GridworldConfig) -> list[str]:
    arrows = {"UP": "^", "DOWN": "v", "LEFT": "<", "RIGHT": ">"}
    rows: list[str] = []
    for r in range(config.height):
        cells: list[str] = []
        for c in range(config.width):
            state = (r, c)
            if state == config.gold:
                cells.append("G")
            elif state == config.bomb:
                cells.append("B")
            else:
                action = max(agent.q[state], key=agent.q[state].get)
                cells.append(arrows[action])
        rows.append(" ".join(cells))
    return rows


def summarize(values: list[float], tail: int = 100) -> float:
    window = values[-tail:]
    return sum(window) / len(window)


def run_suite(output: str) -> None:
    settings = [
        {"name": "baseline", "epsilon": 0.05, "alpha": 0.10, "gamma": 1.00, "initial_q": 0.0},
        {"name": "more_exploration", "epsilon": 0.20, "alpha": 0.10, "gamma": 1.00, "initial_q": 0.0},
        {"name": "optimistic_init", "epsilon": 0.05, "alpha": 0.10, "gamma": 1.00, "initial_q": 5.0},
        {"name": "faster_learning", "epsilon": 0.05, "alpha": 0.30, "gamma": 1.00, "initial_q": 0.0},
    ]
    config = GridworldConfig()
    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "setting", "epsilon", "alpha", "gamma", "initial_q",
            "mean_reward_last100", "mean_steps_last100", "greedy_policy",
        ])
        for setting in settings:
            agent, rewards, steps = train_once(
                seed=20260425,
                episodes=1000,
                slip_probability=0.0,
                **{k: v for k, v in setting.items() if k != "name"},
            )
            writer.writerow([
                setting["name"], setting["epsilon"], setting["alpha"],
                setting["gamma"], setting["initial_q"],
                f"{summarize(rewards):.2f}", f"{summarize(steps):.2f}",
                " / ".join(greedy_policy(agent, config)),
            ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="docs/gridworld_results.csv")
    args = parser.parse_args()
    run_suite(args.output)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

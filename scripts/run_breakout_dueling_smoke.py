"""Short, reproducible Breakout DQN vs Dueling DQN smoke experiment.

This is intentionally a small local run, not the full multi-million-frame Atari
benchmark used in papers. It verifies that the modified architecture can be
trained under the same preprocessing, replay buffer, epsilon-greedy behavior,
and target-network mechanics as the original DQN baseline.
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import deque
from pathlib import Path

import numpy as np
import tensorflow as tf
import keras
from keras import layers, ops

import ale_py  # noqa: F401 - importing registers Atari envs in Gymnasium.
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation

from dqn_dueling_variant import create_dueling_q_model


ENV_ID = "BreakoutNoFrameskip-v4"


def create_original_q_model(num_actions: int = 4) -> keras.Model:
    inputs = layers.Input(shape=(4, 84, 84), name="stacked_frames")
    x = layers.Lambda(
        lambda tensor: ops.transpose(tensor, (0, 2, 3, 1)),
        output_shape=(84, 84, 4),
        name="channels_last",
    )(inputs)
    x = layers.Conv2D(32, 8, strides=4, activation="relu", name="conv_8x8")(x)
    x = layers.Conv2D(64, 4, strides=2, activation="relu", name="conv_4x4")(x)
    x = layers.Conv2D(64, 3, strides=1, activation="relu", name="conv_3x3")(x)
    x = layers.Flatten(name="flatten_features")(x)
    x = layers.Dense(512, activation="relu", name="shared_dense")(x)
    q_values = layers.Dense(num_actions, activation="linear", name="q_values")(x)
    return keras.Model(inputs=inputs, outputs=q_values, name="original_dqn_breakout")


def make_env(seed: int) -> gym.Env:
    env = gym.make(ENV_ID)
    env = AtariPreprocessing(
        env,
        frame_skip=4,
        screen_size=84,
        terminal_on_life_loss=False,
        grayscale_obs=True,
        grayscale_newaxis=False,
        scale_obs=False,
    )
    env = FrameStackObservation(env, stack_size=4)
    env.action_space.seed(seed)
    return env


def choose_action(model: keras.Model, state: np.ndarray, epsilon: float, rng: random.Random, num_actions: int) -> int:
    if rng.random() < epsilon:
        return rng.randrange(num_actions)
    q_values = model(np.expand_dims(state, axis=0), training=False).numpy()[0]
    return int(np.argmax(q_values))


def train_model(model_name: str, steps: int, seed: int, output_rows: list[dict[str, object]]) -> dict[str, object]:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    rng = random.Random(seed)

    env = make_env(seed)
    num_actions = int(env.action_space.n)
    if model_name == "Original DQN":
        model = create_original_q_model(num_actions)
        target = create_original_q_model(num_actions)
    else:
        model = create_dueling_q_model(num_actions)
        target = create_dueling_q_model(num_actions)
    target.set_weights(model.get_weights())

    optimizer = keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)
    loss_fn = keras.losses.Huber()
    replay: deque[tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=5000)

    gamma = 0.99
    batch_size = 16
    warmup = min(200, max(10, steps // 5))
    train_every = 4
    target_update = 1000
    epsilon_start = 1.0
    epsilon_end = 0.10

    state, _ = env.reset(seed=seed)
    state = np.asarray(state, dtype=np.float32)
    episode_reward = 0.0
    episode_rewards: list[float] = []
    losses: list[float] = []

    for step in range(1, steps + 1):
        epsilon = max(epsilon_end, epsilon_start - (epsilon_start - epsilon_end) * step / steps)
        action = choose_action(model, state, epsilon, rng, num_actions)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = bool(terminated or truncated)
        next_state = np.asarray(next_state, dtype=np.float32)
        replay.append((state, int(action), float(reward), next_state, done))
        episode_reward += float(reward)

        if len(replay) >= warmup and step % train_every == 0:
            batch = rng.sample(replay, batch_size)
            states = np.stack([item[0] for item in batch]).astype(np.float32)
            actions = np.asarray([item[1] for item in batch], dtype=np.int32)
            rewards = np.asarray([item[2] for item in batch], dtype=np.float32)
            next_states = np.stack([item[3] for item in batch]).astype(np.float32)
            dones = np.asarray([item[4] for item in batch], dtype=np.float32)

            next_q = target(next_states, training=False)
            max_next_q = tf.reduce_max(next_q, axis=1)
            targets = rewards + gamma * (1.0 - dones) * max_next_q

            with tf.GradientTape() as tape:
                q_values = model(states, training=True)
                action_mask = tf.one_hot(actions, num_actions)
                selected_q = tf.reduce_sum(q_values * action_mask, axis=1)
                loss = loss_fn(targets, selected_q)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            losses.append(float(loss.numpy()))

        if step % target_update == 0:
            target.set_weights(model.get_weights())

        if done:
            episode_rewards.append(episode_reward)
            output_rows.append({
                "model": model_name,
                "step": step,
                "episode": len(episode_rewards),
                "episode_reward": episode_reward,
                "running_reward_last_10": float(np.mean(episode_rewards[-10:])),
            })
            state, _ = env.reset()
            state = np.asarray(state, dtype=np.float32)
            episode_reward = 0.0
        else:
            state = next_state

    if episode_reward or not episode_rewards:
        episode_rewards.append(episode_reward)
        output_rows.append({
            "model": model_name,
            "step": steps,
            "episode": len(episode_rewards),
            "episode_reward": episode_reward,
            "running_reward_last_10": float(np.mean(episode_rewards[-10:])),
        })
    env.close()

    return {
        "model": model_name,
        "steps": steps,
        "episodes": len(episode_rewards),
        "final_running_reward": float(np.mean(episode_rewards[-10:])),
        "mean_reward_last_5": float(np.mean(episode_rewards[-5:])),
        "best_reward": float(np.max(episode_rewards)),
        "mean_loss": float(np.mean(losses)) if losses else 0.0,
        "parameter_count": int(model.count_params()),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def render_curve(path: Path, episode_rows: list[dict[str, object]]) -> None:
    width, height = 900, 420
    margin = 58
    models = sorted({str(row["model"]) for row in episode_rows})
    rewards = [float(row["running_reward_last_10"]) for row in episode_rows]
    y_min = min(0.0, min(rewards))
    y_max = max(1.0, max(rewards))
    max_step = max(int(row["step"]) for row in episode_rows)
    colors = {"Original DQN": "#b94f45", "Dueling DQN": "#226f68"}

    def sx(step: int) -> float:
        return margin + (width - 2 * margin) * step / max_step

    def sy(value: float) -> float:
        return height - margin - (height - 2 * margin) * (value - y_min) / (y_max - y_min)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fffaf1"/>',
        '<text x="450" y="36" text-anchor="middle" font-family="Georgia, Times New Roman, serif" font-size="24" font-weight="700" fill="#172026">Breakout Smoke Run: Running Reward</text>',
        f'<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#6c6a62"/>',
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="#6c6a62"/>',
        f'<text x="{width/2}" y="{height-18}" text-anchor="middle" font-family="Georgia" font-size="14" fill="#6c6a62">environment steps</text>',
        f'<text x="20" y="{height/2}" text-anchor="middle" transform="rotate(-90 20 {height/2})" font-family="Georgia" font-size="14" fill="#6c6a62">running reward</text>',
    ]
    for model in models:
        rows = [row for row in episode_rows if row["model"] == model]
        points = " ".join(f'{sx(int(row["step"])):.1f},{sy(float(row["running_reward_last_10"])):.1f}' for row in rows)
        if points:
            parts.append(f'<polyline points="{points}" fill="none" stroke="{colors.get(model, "#40515b")}" stroke-width="3"/>')
        lx = width - margin - 170
        ly = 72 + models.index(model) * 24
        parts.append(f'<rect x="{lx}" y="{ly-12}" width="14" height="14" fill="{colors.get(model, "#40515b")}"/>')
        parts.append(f'<text x="{lx+22}" y="{ly}" font-family="Georgia" font-size="14" fill="#172026">{model}</text>')
    parts.append('</svg>\n')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=20260427)
    parser.add_argument("--summary", default="docs/breakout_dueling_smoke_summary.csv")
    parser.add_argument("--episodes", default="docs/breakout_dueling_smoke_episodes.csv")
    parser.add_argument("--curve", default="docs/images/breakout_dueling_smoke_curve.svg")
    args = parser.parse_args()

    episode_rows: list[dict[str, object]] = []
    summaries = [
        train_model("Original DQN", args.steps, args.seed, episode_rows),
        train_model("Dueling DQN", args.steps, args.seed, episode_rows),
    ]
    write_csv(Path(args.summary), summaries)
    write_csv(Path(args.episodes), episode_rows)
    render_curve(Path(args.curve), episode_rows)
    print(f"wrote {args.summary}")
    print(f"wrote {args.episodes}")
    print(f"wrote {args.curve}")
    for row in summaries:
        print(row)


if __name__ == "__main__":
    main()

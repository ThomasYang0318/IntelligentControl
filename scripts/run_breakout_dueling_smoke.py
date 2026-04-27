"""Short, reproducible Breakout DQN vs Dueling DQN smoke experiment.

This is intentionally a small local run, not the full multi-million-frame Atari
benchmark used in papers. It verifies that the modified architecture can be
trained under the same preprocessing, replay buffer, epsilon-greedy behavior,
and target-network mechanics as the original DQN baseline.
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from collections import deque
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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


def render_curve(path: Path, episode_rows: list[dict[str, object]], summaries: list[dict[str, object]]) -> None:
    """Render the smoke-run curve with matplotlib.

    The figure is generated from the CSV-style result rows produced by the run,
    not manually drawn SVG strings.
    """
    models = ["Original DQN", "Dueling DQN"]
    colors = {"Original DQN": "#b94f45", "Dueling DQN": "#226f68"}
    summary_by_model = {str(row["model"]): row for row in summaries}

    fig, (ax, table_ax) = plt.subplots(
        1,
        2,
        figsize=(13.2, 6.4),
        gridspec_kw={"width_ratios": [2.65, 1.05]},
    )
    fig.patch.set_facecolor("#fffaf1")
    ax.set_facecolor("#fffdf7")
    table_ax.set_facecolor("#fffdf7")

    max_step = max(int(row["step"]) for row in episode_rows)
    for model in models:
        rows = [row for row in episode_rows if row["model"] == model]
        steps = [int(row["step"]) for row in rows]
        running = [float(row["running_reward_last_10"]) for row in rows]
        color = colors[model]

        ax.plot(steps, running, marker="o", linewidth=2.8, markersize=6, color=color, label=model)
        for step, run_value in zip(steps, running):
            ax.annotate(
                f"{run_value:.2f}",
                (step, run_value),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=8,
                color=color,
                fontweight="bold",
            )
        if steps:
            ax.annotate(
                f"final {running[-1]:.2f}",
                (steps[-1], running[-1]),
                textcoords="offset points",
                xytext=(10 if model == "Dueling DQN" else -10, -16),
                ha="left" if model == "Dueling DQN" else "right",
                fontsize=10,
                color=color,
                fontweight="bold",
            )

    ax.set_title("Breakout Smoke Run: Original DQN vs Dueling DQN", fontsize=17, fontweight="bold", pad=18)
    ax.text(
        0.5,
        1.02,
        f"Actual run, {max_step} environment steps per model, same preprocessing/seed/replay/target mechanics",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=9.5,
        color="#6c6a62",
    )
    ax.set_xlabel("environment steps")
    ax.set_ylabel("running reward")
    ax.grid(True, color="#eadcc4", linewidth=0.9)
    ax.legend(frameon=False, loc="upper right")
    for spine in ax.spines.values():
        spine.set_color("#d7c7aa")
    ax.set_xlim(0, max_step * 1.08)

    table_ax.axis("off")
    table_ax.set_title("Numeric Results", fontsize=15, fontweight="bold", pad=12)
    rows_for_table = []
    row_colors = []
    for model in models:
        summary = summary_by_model[model]
        rows_for_table.extend([
            [model, ""],
            ["final running", f"{float(summary['final_running_reward']):.3f}"],
            ["mean last 5", f"{float(summary['mean_reward_last_5']):.3f}"],
            ["best reward", f"{float(summary['best_reward']):.3f}"],
            ["mean loss", f"{float(summary['mean_loss']):.3f}"],
            ["params", f"{int(summary['parameter_count']):,}"],
            ["", ""],
        ])
        row_colors.extend([colors[model]] + ["#172026"] * 5 + ["#172026"])

    table = table_ax.table(
        cellText=rows_for_table,
        colLabels=["metric", "value"],
        cellLoc="left",
        colLoc="left",
        loc="center",
        colWidths=[0.62, 0.38],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1.08, 1.42)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#d7c7aa")
        if row == 0:
            cell.set_facecolor("#f8f3ea")
            cell.set_text_props(weight="bold", color="#172026")
        elif row - 1 < len(row_colors):
            cell.set_facecolor("#fffdf7")
            if rows_for_table[row - 1][1] == "":
                cell.set_text_props(weight="bold", color=row_colors[row - 1])
            elif col == 1:
                cell.set_text_props(weight="bold", color="#172026")
            else:
                cell.set_text_props(color="#6c6a62")

    delta = float(summary_by_model["Dueling DQN"]["final_running_reward"]) - float(summary_by_model["Original DQN"]["final_running_reward"])
    table_ax.text(
        0.5,
        0.04,
        f"Dueling final Δ = {delta:+.2f}",
        transform=table_ax.transAxes,
        ha="center",
        fontsize=12,
        color="#226f68" if delta >= 0 else "#b94f45",
        fontweight="bold",
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=2.0)
    fig.savefig(path, format=path.suffix.lstrip(".") or "svg", facecolor=fig.get_facecolor())
    plt.close(fig)


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
    render_curve(Path(args.curve), episode_rows, summaries)
    print(f"wrote {args.summary}")
    print(f"wrote {args.episodes}")
    print(f"wrote {args.curve}")
    for row in summaries:
        print(row)


if __name__ == "__main__":
    main()

"""Dueling DQN model variant for the Atari Breakout Keras example.

This file contains the network factory that can replace the original
`create_q_model()` from the Keras Breakout example. The training loop, replay
buffer, target network, epsilon-greedy exploration, optimizer, and preprocessing
can remain unchanged so the comparison isolates the architecture change.

Source basis:
- Keras Breakout example: Apache-2.0, keras-team/keras-io.
- Dueling architecture idea: Wang et al., arXiv:1511.06581.
"""
from __future__ import annotations

import os

os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import keras
from keras import layers, ops

from dqn_architecture_spec import NUM_ACTIONS


def create_dueling_q_model(num_actions: int = NUM_ACTIONS) -> keras.Model:
    inputs = layers.Input(shape=(4, 84, 84), name="stacked_frames")

    x = layers.Lambda(
        lambda tensor: ops.transpose(tensor, (0, 2, 3, 1)),
        output_shape=(84, 84, 4),
        name="channels_last",
    )(inputs)

    # CNN backbone: same spirit as the original Keras DQN Breakout model.
    x = layers.Conv2D(32, 8, strides=4, activation="relu", name="conv_8x8")(x)
    x = layers.Conv2D(64, 4, strides=2, activation="relu", name="conv_4x4")(x)
    x = layers.Conv2D(64, 3, strides=1, activation="relu", name="conv_3x3")(x)
    x = layers.Flatten(name="flatten_features")(x)
    x = layers.Dense(512, activation="relu", name="shared_dense")(x)

    # Value stream estimates how good the state itself is: V(s).
    value = layers.Dense(256, activation="relu", name="value_hidden")(x)
    value = layers.Dense(1, activation="linear", name="state_value")(value)

    # Advantage stream estimates how much each action improves over others: A(s,a).
    advantage = layers.Dense(256, activation="relu", name="advantage_hidden")(x)
    advantage = layers.Dense(
        num_actions, activation="linear", name="action_advantage"
    )(advantage)

    advantage_mean = layers.Lambda(
        lambda a: ops.mean(a, axis=1, keepdims=True),
        name="advantage_mean",
    )(advantage)

    centered_advantage = layers.Subtract(name="center_advantage")(
        [advantage, advantage_mean]
    )
    q_values = layers.Add(name="q_values")([value, centered_advantage])

    return keras.Model(inputs=inputs, outputs=q_values, name="dueling_dqn_breakout")


if __name__ == "__main__":
    model = create_dueling_q_model()
    model.summary()

"""Dueling DQN model variant for the Atari Breakout Keras example.

This file contains only the network factory that can replace the original
`create_q_model()` from the Keras example. The training loop, replay buffer,
target network, epsilon-greedy exploration, and optimizer can remain unchanged.

Source basis:
- Keras Breakout example: Apache-2.0, keras-team/keras-io.
- Dueling architecture idea: Wang et al., arXiv:1511.06581.
"""
from __future__ import annotations

import os
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import keras
from keras import layers


def create_dueling_q_model(num_actions: int = 4) -> keras.Model:
    inputs = keras.Input(shape=(4, 84, 84), name="stacked_frames")
    x = layers.Lambda(
        lambda tensor: keras.ops.transpose(tensor, [0, 2, 3, 1]),
        output_shape=(84, 84, 4),
        name="channels_last",
    )(inputs)
    x = layers.Rescaling(1.0 / 255.0, name="pixel_rescale")(x)

    x = layers.Conv2D(32, 8, strides=4, activation="relu", name="conv_8x8")(x)
    x = layers.Conv2D(64, 4, strides=2, activation="relu", name="conv_4x4")(x)
    x = layers.Conv2D(64, 3, strides=1, activation="relu", name="conv_3x3")(x)
    x = layers.Flatten(name="flatten_features")(x)

    value = layers.Dense(512, activation="relu", name="value_hidden")(x)
    value = layers.Dense(1, name="state_value")(value)

    advantage = layers.Dense(512, activation="relu", name="advantage_hidden")(x)
    advantage = layers.Dense(num_actions, name="action_advantage")(advantage)
    advantage_mean = layers.Lambda(
        lambda tensor: keras.ops.mean(tensor, axis=1, keepdims=True),
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

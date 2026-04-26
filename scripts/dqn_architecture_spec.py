"""Shared Dueling DQN architecture specification used by code and figures."""
from __future__ import annotations

NUM_ACTIONS = 4
INPUT_SHAPE = (4, 84, 84)
CHANNELS_LAST_SHAPE = (84, 84, 4)

CNN_BACKBONE = [
    {"type": "Conv2D", "filters": 32, "kernel": 8, "stride": 4, "activation": "relu"},
    {"type": "Conv2D", "filters": 64, "kernel": 4, "stride": 2, "activation": "relu"},
    {"type": "Conv2D", "filters": 64, "kernel": 3, "stride": 1, "activation": "relu"},
    {"type": "Flatten"},
    {"type": "Dense", "units": 512, "activation": "relu", "role": "shared"},
]

VALUE_STREAM = [
    {"type": "Dense", "units": 256, "activation": "relu"},
    {"type": "Dense", "units": 1, "activation": "linear", "name": "V(s)"},
]

ADVANTAGE_STREAM = [
    {"type": "Dense", "units": NUM_ACTIONS, "activation": "linear", "name": "A(s,a)", "preceded_by": {"type": "Dense", "units": 256, "activation": "relu"}},
]

COMBINE_FORMULA = "Q(s,a)=V(s)+(A(s,a)-mean(A))"

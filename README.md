# Intelligent Control Mid-term Takehome 2026

This repository is a learning record for the three take-home questions.

## Files

- `docs/takehome_report.md`: final written answers in Chinese.
- `scripts/q_learning_gridworld.py`: clean-room Gridworld Q-learning implementation and experiment runner.
- `docs/gridworld_results.csv`: reproducible Gridworld experiment summary.
- `docs/gridworld_q_tables.csv`: learned Q-tables for the three settings discussed in Question 1.2.
- `scripts/dqn_dueling_variant.py`: Dueling DQN model variant for the Keras Atari Breakout example.
- `requirements.txt`: optional dependencies for running the Atari model variant.

## Reproduce Gridworld Results

```bash
python3 scripts/q_learning_gridworld.py --output docs/gridworld_results.csv
```

## Legal Use Notes

The Gridworld GitHub repository referenced by the assignment does not visibly provide a license file, so this repo does not copy its notebook code. The included Gridworld script is a clean-room implementation based on the public assignment description and Q-learning equation.

The Keras Breakout example is from `keras-team/keras-io`, which is Apache-2.0 licensed. The Decision Transformer repository is MIT licensed. See `docs/takehome_report.md` for citations.

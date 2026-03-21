# Deep Q-Learning - Atari Agent

## Environment
**BreakoutNoFrameskip-v0** — The agent learns to play Atari Breakout using a DQN.

## Setup

```bash
pip install -r requirements.txt
```

## Training

Each member runs their 10 experiments by specifying their member number:
```bash
python train.py 1 --timesteps 200000   # Member 1
python train.py 2 --timesteps 200000   # Member 2
python train.py 3 --timesteps 200000   # Member 3
```

This reads `experiments_memberN.csv` and trains all 10 experiments. Results are saved to `results_memberN.csv` and models to the `models/` folder.

Monitor training with TensorBoard:
```bash
tensorboard --logdir logs/tensorboard
```

## Playing

Load a trained model and watch the agent play:
```bash
python play.py --model models/<experiment_name>/best_model.zip --env BreakoutNoFrameskip-v0
```

Record a gameplay video:
```bash
python play.py --model models/<experiment_name>/best_model.zip --env BreakoutNoFrameskip-v0 --record
```

## Hyperparameter Tuning Results

### Member 1: [YOUR NAME]

| Experiment | LR | Gamma | Batch Size | Eps Start | Eps End | Eps Fraction | Mean Reward | Noted Behavior |
|---|---|---|---|---|---|---|---|---|
| exp1_baseline | 0.0001 | 0.99 | 32 | 1.0 | 0.01 | 0.1 | | |
| exp2_high_lr | 0.001 | 0.99 | 32 | 1.0 | 0.01 | 0.1 | | |
| exp3_low_lr | 0.00001 | 0.99 | 32 | 1.0 | 0.01 | 0.1 | | |
| exp4_low_gamma | 0.0001 | 0.90 | 32 | 1.0 | 0.01 | 0.1 | | |
| exp5_high_gamma | 0.0001 | 0.999 | 32 | 1.0 | 0.01 | 0.1 | | |
| exp6_large_batch | 0.0001 | 0.99 | 64 | 1.0 | 0.01 | 0.1 | | |
| exp7_small_batch | 0.0001 | 0.99 | 16 | 1.0 | 0.01 | 0.1 | | |
| exp8_slow_decay | 0.0001 | 0.99 | 32 | 1.0 | 0.01 | 0.3 | | |
| exp9_mlp_baseline | 0.0001 | 0.99 | 32 | 1.0 | 0.01 | 0.1 | | |
| exp10_mlp_tuned | 0.0005 | 0.95 | 64 | 1.0 | 0.05 | 0.2 | | |

### Member 2: [YOUR NAME]

| Experiment | LR | Gamma | Batch Size | Eps Start | Eps End | Eps Fraction | Mean Reward | Noted Behavior |
|---|---|---|---|---|---|---|---|---|
| exp1_mlp_baseline | 0.0001 | 0.99 | 32 | 1.0 | 0.01 | 0.1 | | |
| exp2_mlp_high_lr | 0.001 | 0.99 | 32 | 1.0 | 0.01 | 0.1 | | |
| exp3_cnn_mid_lr | 0.0005 | 0.99 | 32 | 1.0 | 0.01 | 0.1 | | |
| exp4_cnn_gamma95 | 0.0001 | 0.95 | 32 | 1.0 | 0.01 | 0.1 | | |
| exp5_large_batch_128 | 0.0001 | 0.99 | 128 | 1.0 | 0.01 | 0.1 | | |
| exp6_combo_high_lr_low_gamma | 0.001 | 0.90 | 32 | 1.0 | 0.01 | 0.1 | | |
| exp7_combo_low_lr_high_gamma | 0.00005 | 0.999 | 64 | 1.0 | 0.01 | 0.1 | | |
| exp8_aggressive_explore | 0.0001 | 0.99 | 32 | 1.0 | 0.05 | 0.5 | | |
| exp9_low_eps_start | 0.0001 | 0.99 | 32 | 0.5 | 0.01 | 0.1 | | |
| exp10_mlp_tuned | 0.0005 | 0.95 | 64 | 1.0 | 0.05 | 0.2 | | |

### Member 3: [YOUR NAME]

| Experiment | LR | Gamma | Batch Size | Eps Start | Eps End | Eps Fraction | Mean Reward | Noted Behavior |
|---|---|---|---|---|---|---|---|---|
| exp1_very_high_lr | 0.005 | 0.99 | 32 | 1.0 | 0.01 | 0.1 | | |
| exp2_extreme_low_gamma | 0.0001 | 0.80 | 32 | 1.0 | 0.01 | 0.1 | | |
| exp3_very_large_batch | 0.0001 | 0.99 | 256 | 1.0 | 0.01 | 0.1 | | |
| exp4_no_explore | 0.0001 | 0.99 | 32 | 0.1 | 0.01 | 0.1 | | |
| exp5_long_explore | 0.0001 | 0.99 | 32 | 1.0 | 0.01 | 0.7 | | |
| exp6_combo_optimal | 0.0003 | 0.97 | 48 | 1.0 | 0.02 | 0.15 | | |
| exp7_mlp_low_gamma | 0.0001 | 0.85 | 32 | 1.0 | 0.01 | 0.1 | | |
| exp8_cnn_balanced | 0.0002 | 0.98 | 64 | 1.0 | 0.05 | 0.2 | | |
| exp9_high_lr_large_batch | 0.001 | 0.99 | 128 | 1.0 | 0.01 | 0.1 | | |
| exp10_final_tuned | 0.00025 | 0.99 | 32 | 1.0 | 0.02 | 0.12 | | |

## Policy Comparison: MLP vs CNN

| Policy | Mean Reward | Training Time | Notes |
|--------|------------|---------------|-------|
| CnnPolicy | | | |
| MlpPolicy | | | |

**Discussion:** _(Fill in after experiments)_

## Gameplay Video

_(Embed or link gameplay recording here)_

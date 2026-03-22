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

### Member 1: Wilsons

| Experiment | Policy | LR | Gamma | Batch Size | Eps Start | Eps End | Eps Fraction | Mean Reward | Noted Behavior |
|---|---|---|---|---|---|---|---|---|---|
| exp1_baseline | CnnPolicy | 0.0001 | 0.99 | 32 | 1.0 | 0.01 | 0.1 | 0.5 +/- 0.50 | Baseline CNN config. Moderate performance with standard hyperparameters; agent occasionally hits the ball but lacks consistency. |
| exp2_high_lr | CnnPolicy | 0.001 | 0.99 | 32 | 1.0 | 0.01 | 0.1 | 0.4 +/- 0.80 | Higher learning rate caused unstable training with high variance. The agent learned faster initially but overshot optimal Q-values, leading to erratic behavior. |
| exp3_low_lr | CnnPolicy | 0.00001 | 0.99 | 32 | 1.0 | 0.01 | 0.1 | 0.3 +/- 0.64 | Too-low learning rate slowed convergence significantly. The agent barely learned within 200k timesteps, showing the network needs sufficient gradient steps to improve. |
| exp4_low_gamma | CnnPolicy | 0.0001 | 0.90 | 32 | 1.0 | 0.01 | 0.1 | 0.0 +/- 0.00 | Worst result. Gamma=0.9 made the agent too short-sighted — it could not learn to associate paddle positioning with future brick-breaking rewards, resulting in zero reward. |
| exp5_high_gamma | CnnPolicy | 0.0001 | 0.999 | 32 | 1.0 | 0.01 | 0.1 | 0.2 +/- 0.60 | Very high gamma overvalued distant rewards, making Q-value estimates noisy and harder to converge within limited timesteps. |
| exp6_large_batch | CnnPolicy | 0.0001 | 0.99 | 64 | 1.0 | 0.01 | 0.1 | 0.4 +/- 0.92 | Larger batch gave more stable gradients but higher variance in outcomes. Fewer weight updates per timestep (same total data, bigger batches) may have slowed learning. |
| exp7_small_batch | CnnPolicy | 0.0001 | 0.99 | 16 | 1.0 | 0.01 | 0.1 | 0.5 +/- 0.67 | Smaller batch size matched the baseline reward. More frequent updates helped; slightly noisier gradients but more optimization steps overall. |
| exp8_slow_decay | CnnPolicy | 0.0001 | 0.99 | 32 | 1.0 | 0.01 | 0.3 | 0.5 +/- 0.81 | Slower epsilon decay (30% of training) extended exploration. Matched baseline reward but with higher variance — the agent explored longer before exploiting. |
| exp9_mlp_baseline | MlpPolicy | 0.0001 | 0.99 | 32 | 1.0 | 0.01 | 0.1 | 0.1 +/- 0.30 | MLP struggled significantly. Flattening 84x84x4 frames into a 28k-dimensional vector loses spatial information, making it very hard to learn visual features for Breakout. |
| exp10_mlp_tuned | MlpPolicy | 0.0005 | 0.95 | 64 | 1.0 | 0.05 | 0.2 | 0.5 +/- 0.81 | Tuned MLP with higher LR and larger batch improved over MLP baseline (0.1 to 0.5). Higher LR compensated for the policy's limited representational capacity. |

### Member 2: Mike

| Experiment | Policy | LR | Gamma | Batch Size | Eps Start | Eps End | Eps Fraction | Mean Reward | Noted Behavior |
|---|---|---|---|---|---|---|---|---|---|
| exp1_mlp_baseline | MlpPolicy | 0.0001 | 0.99 | 32 | 1.0 | 0.01 | 0.1 | 0.4 +/- 0.49 | MLP baseline performed better than Member 1's MLP (0.4 vs 0.1), likely due to run-to-run variance in Atari environments. Still weaker than CNN overall. |
| exp2_mlp_high_lr | MlpPolicy | 0.001 | 0.99 | 32 | 1.0 | 0.01 | 0.1 | 0.6 +/- 0.80 | Best MLP result for Member 2. Higher learning rate helped MLP learn faster within the limited timestep budget, despite the flat input representation. |
| exp3_cnn_mid_lr | CnnPolicy | 0.0005 | 0.99 | 32 | 1.0 | 0.01 | 0.1 | 0.0 +/- 0.00 | Mid-range LR with CNN produced zero reward. The slightly higher LR may have caused instability in the CNN's deeper computation graph. |
| exp4_cnn_gamma95 | CnnPolicy | 0.0001 | 0.95 | 32 | 1.0 | 0.01 | 0.1 | 0.1 +/- 0.30 | Reducing gamma to 0.95 harmed performance, consistent with Member 1's low-gamma result. Breakout requires long-horizon planning (gamma close to 1.0). |
| exp5_large_batch_128 | CnnPolicy | 0.0001 | 0.99 | 128 | 1.0 | 0.01 | 0.1 | 0.6 +/- 0.49 | Tied for best. Very large batch stabilized training and reduced variance (std=0.49 is the lowest among top performers). Much longer training time (~41k seconds). |
| exp6_combo_high_lr_low_gamma | CnnPolicy | 0.001 | 0.90 | 32 | 1.0 | 0.01 | 0.1 | 0.4 +/- 0.80 | Combining high LR with low gamma partially offset the damage from short-sightedness. High LR accelerated learning but gamma=0.9 still limited long-term planning. |
| exp7_combo_low_lr_high_gamma | CnnPolicy | 0.00005 | 0.999 | 64 | 1.0 | 0.01 | 0.1 | 0.0 +/- 0.00 | Very low LR + very high gamma = no learning at all. The tiny learning rate combined with noisy Q-targets (gamma~1) prevented any meaningful weight updates. |
| exp8_aggressive_explore | CnnPolicy | 0.0001 | 0.99 | 32 | 1.0 | 0.05 | 0.5 | 0.6 +/- 0.66 | Tied for best. Extending exploration to 50% of training with a higher final epsilon helped the agent discover more diverse strategies before exploiting. |
| exp9_low_eps_start | CnnPolicy | 0.0001 | 0.99 | 32 | 0.5 | 0.01 | 0.1 | 0.3 +/- 0.64 | Starting epsilon at 0.5 (instead of 1.0) reduced initial exploration, causing the agent to exploit too early before learning good Q-values. |
| exp10_mlp_tuned | MlpPolicy | 0.0005 | 0.95 | 64 | 1.0 | 0.05 | 0.2 | 0.2 +/- 0.60 | Same tuned MLP config as Member 1's exp10, but lower reward (0.2 vs 0.5). Demonstrates run-to-run variance inherent in RL training. |

### Member 3: Gershorm

| Experiment | Policy | LR | Gamma | Batch Size | Eps Start | Eps End | Eps Fraction | Mean Reward | Noted Behavior |
|---|---|---|---|---|---|---|---|---|---|
| exp1_mlp_baseline | MlpPolicy | 0.0001 | 0.99 | 32 | 1.0 | 0.01 | 0.1 | 0.1 +/- 0.30 | MLP baseline with default params. Weak performance confirms that flat MLP cannot effectively process raw Atari frames without convolutional feature extraction. |
| exp2_mlp_high_lr | MlpPolicy | 0.001 | 0.99 | 32 | 1.0 | 0.01 | 0.1 | 0.3 +/- 0.64 | Higher LR improved MLP performance (0.1 to 0.3), consistent with Member 2's finding that MLP benefits from aggressive learning rates. |
| exp3_cnn_mid_lr | CnnPolicy | 0.0005 | 0.99 | 32 | 1.0 | 0.01 | 0.1 | 0.5 +/- 0.81 | CNN with mid-range LR performed well here, unlike Member 2's zero result — highlighting the stochasticity of DQN training. |
| exp4_cnn_gamma95 | CnnPolicy | 0.0001 | 0.95 | 32 | 1.0 | 0.01 | 0.1 | 0.5 +/- 0.67 | Gamma=0.95 worked better here than for other members. The agent still learned basic paddle control, though performance plateaued earlier than gamma=0.99. |
| exp5_large_batch_128 | CnnPolicy | 0.0001 | 0.99 | 128 | 1.0 | 0.01 | 0.1 | 0.4 +/- 0.80 | Large batch size produced moderate results. Fewer update steps per timestep slowed learning slightly compared to smaller batches. |
| exp6_combo_high_lr_low_gamma | CnnPolicy | 0.001 | 0.90 | 32 | 1.0 | 0.01 | 0.1 | 0.5 +/- 0.81 | High LR partially compensated for low gamma, allowing the agent to still learn within the shorter effective planning horizon. |
| exp7_combo_low_lr_high_gamma | CnnPolicy | 0.00005 | 0.999 | 64 | 1.0 | 0.01 | 0.1 | 0.3 +/- 0.46 | Low LR + high gamma produced low-variance but modest results. The conservative updates kept training stable but slow. |
| exp8_cnn_balanced | CnnPolicy | 0.0001 | 0.99 | 32 | 1.0 | 0.01 | 0.2 | **3.80 +/- 1.60** | **Best performing model overall (mean reward 3.80, best episode 5.0).** Large replay buffer (100k) and delayed learning starts (50k) allowed the agent to collect diverse experience before training, producing a much stronger policy. Saved as `dqn_model.zip`. |
| exp9_low_eps_start | CnnPolicy | 0.0001 | 0.99 | 32 | 0.5 | 0.01 | 0.1 | 0.2 +/- 0.60 | Starting with epsilon=0.5 reduced early exploration. Without sufficient random experience, the replay buffer lacked diversity for effective learning. |
| exp10_mlp_tuned | MlpPolicy | 0.0005 | 0.95 | 64 | 1.0 | 0.05 | 0.2 | 0.8 +/- 0.75 | Best MLP result. Tuned MLP with moderate LR, lower gamma, and extended exploration achieved the highest MLP mean reward. Shows that with the right hyperparameters, MLP can learn basic Breakout strategies. |

## Policy Comparison: MLP vs CNN

| Policy | Avg Mean Reward | Best Mean Reward | Avg Training Time | Notes |
|--------|----------------|-----------------|-------------------|-------|
| CnnPolicy | 0.47 | 3.80 | ~2,460s | Better at extracting spatial features from frames. More consistent across experiments and achieved the best overall result (exp8_cnn_balanced: 3.80). Requires more compute but clearly superior for image-based tasks. |
| MlpPolicy | 0.37 | 0.8 | ~1,310s | Struggles with raw pixel input (28k-dim flat vector). Faster to train but less consistent. Best MLP result was 0.8, far below CNN's best of 3.80. |

**Discussion:**

CNN is the clear winner for image-based Atari environments. Convolutional layers extract spatial features (paddle position, ball trajectory, brick layout) that a flat MLP cannot easily learn. The best overall model was a CNN (Gershorm's exp8_cnn_balanced: mean reward 3.80), which used a large replay buffer (100k) and delayed learning starts (50k) to collect diverse experience before training. This was nearly 5x better than the best MLP result (0.8).

MLP's best result (Member 3, exp10: reward=0.8) shows that with tuned hyperparameters, MLP can learn basic strategies, but it cannot match CNN's ability to process visual information effectively.

Key takeaways from hyperparameter tuning:
- **Learning rate:** 0.0001-0.0005 works best. Too high (0.001) causes instability; too low (0.00001) prevents convergence.
- **Gamma:** 0.99 is optimal for Breakout. Values below 0.95 make the agent too short-sighted to plan paddle movements.
- **Batch size:** 32 is a good default. 128 can improve stability but slows training significantly.
- **Epsilon decay:** Moderate exploration (10-20% of training) is best. Both too little (low start) and too much (50% fraction) hurt performance.

## Gameplay Video

[Watch the agent play Breakout](https://drive.google.com/file/d/1PpbXyLvc7mxTRpcyAZ4SOAoAaSJ9zotl/view?usp=sharing)

"""
train.py - Train a DQN agent to play an Atari game using Stable Baselines3.

Usage:
    python train.py 1 --timesteps 200000   # Run Member 1 experiments
    python train.py 2 --timesteps 200000   # Run Member 2 experiments
    python train.py 3 --timesteps 200000   # Run Member 3 experiments
"""

import argparse
import csv
import os
import time

import gymnasium as gym
import ale_py
gym.register_envs(ale_py)
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
import numpy as np


def make_atari_env(env_id):
    """Create and wrap an Atari environment with standard preprocessing."""
    def _init():
        env = gym.make(env_id, render_mode=None)
        env = AtariWrapper(env)
        return env
    return _init


def train_agent(
    env_id="BreakoutNoFrameskip-v0",
    policy="CnnPolicy",
    learning_rate=1e-4,
    gamma=0.99,
    batch_size=32,
    buffer_size=10000,
    learning_starts=100,
    train_freq=4,
    target_update_interval=1000,
    exploration_initial_eps=1.0,
    exploration_final_eps=0.01,
    exploration_fraction=0.1,
    total_timesteps=500000,
    model_save_path="models",
    experiment_name="default",
    log_dir="logs",
):
    """Train a DQN agent with the given hyperparameters."""

    os.makedirs(model_save_path, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Experiment: {experiment_name}")
    print(f"Environment: {env_id}")
    print(f"Policy: {policy}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Gamma: {gamma}")
    print(f"Batch Size: {batch_size}")
    print(f"Buffer Size: {buffer_size}")
    print(f"Learning Starts: {learning_starts}")
    print(f"Train Freq: {train_freq}")
    print(f"Target Update Interval: {target_update_interval}")
    print(f"Epsilon: {exploration_initial_eps} -> {exploration_final_eps} (fraction: {exploration_fraction})")
    print(f"Total Timesteps: {total_timesteps}")
    print(f"{'='*60}\n")

    # Create training environment
    env = DummyVecEnv([make_atari_env(env_id)])
    env = VecFrameStack(env, n_stack=4)

    # Create evaluation environment
    eval_env = DummyVecEnv([make_atari_env(env_id)])
    eval_env = VecFrameStack(eval_env, n_stack=4)

    # Set up evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(model_save_path, experiment_name),
        log_path=os.path.join(log_dir, experiment_name),
        eval_freq=10000,
        n_eval_episodes=5,
        deterministic=True,
    )

    # Create DQN agent
    model = DQN(
        policy=policy,
        env=env,
        learning_rate=learning_rate,
        gamma=gamma,
        batch_size=batch_size,
        buffer_size=buffer_size,
        learning_starts=learning_starts,
        train_freq=train_freq,
        target_update_interval=target_update_interval,
        exploration_initial_eps=exploration_initial_eps,
        exploration_final_eps=exploration_final_eps,
        exploration_fraction=exploration_fraction,
        tensorboard_log=os.path.join(log_dir, "tensorboard"),
        verbose=1,
    )

    # Train
    start_time = time.time()
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
        tb_log_name=experiment_name,
    )
    training_time = time.time() - start_time

    # Save final model
    final_model_path = os.path.join(model_save_path, f"{experiment_name}_final")
    model.save(final_model_path)
    print(f"\nModel saved to {final_model_path}.zip")

    # Evaluate final model
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
    print(f"\nFinal Evaluation: Mean Reward = {mean_reward:.2f} +/- {std_reward:.2f}")
    print(f"Training Time: {training_time:.1f}s")

    env.close()
    eval_env.close()

    return {
        "experiment": experiment_name,
        "env": env_id,
        "policy": policy,
        "lr": learning_rate,
        "gamma": gamma,
        "batch_size": batch_size,
        "buffer_size": buffer_size,
        "learning_starts": learning_starts,
        "train_freq": train_freq,
        "target_update_interval": target_update_interval,
        "epsilon_start": exploration_initial_eps,
        "epsilon_end": exploration_final_eps,
        "epsilon_fraction": exploration_fraction,
        "timesteps": total_timesteps,
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "training_time_s": training_time,
    }


def run_experiments_from_csv(csv_path, env_id, total_timesteps, member_num=None):
    """Run multiple experiments defined in a CSV file."""
    results = []

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Skip empty rows (e.g. trailing blank lines in CSV)
            if not row.get("experiment_name", "").strip():
                continue

            def _get(row_key, default, cast_fn):
                value = row.get(row_key, "")
                if value is None:
                    return default
                value = str(value).strip()
                if value == "":
                    return default
                return cast_fn(value)

            result = train_agent(
                env_id=env_id,
                policy=row.get("policy", "CnnPolicy"),
                learning_rate=float(row["lr"]),
                gamma=float(row["gamma"]),
                batch_size=int(row["batch_size"]),
                buffer_size=_get("buffer_size", 10000, int),
                learning_starts=_get("learning_starts", 100, int),
                train_freq=_get("train_freq", 4, int),
                target_update_interval=_get("target_update_interval", 1000, int),
                exploration_initial_eps=float(row["epsilon_start"]),
                exploration_final_eps=float(row["epsilon_end"]),
                exploration_fraction=float(row["epsilon_fraction"]),
                total_timesteps=total_timesteps,
                experiment_name=row["experiment_name"],
            )
            results.append(result)

    # Save results summary
    if results:
        if member_num:
            output_path = f"results_member{member_num}.csv"
        else:
            output_path = "experiment_results.csv"
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\nAll results saved to {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Train a DQN agent on Atari")
    parser.add_argument("member", nargs="?", type=int, default=None,
                        help="Member number (1, 2, or 3) — loads experiments_memberX.csv automatically")
    parser.add_argument("--csv", type=str, default=None,
                        help="Path to a custom experiments CSV file")
    parser.add_argument("--env", type=str, default="BreakoutNoFrameskip-v0",
                        help="Atari environment ID")
    parser.add_argument("--timesteps", type=int, default=200000,
                        help="Total training timesteps")

    args = parser.parse_args()

    if args.csv:
        if not os.path.exists(args.csv):
            print(f"Error: {args.csv} not found!")
            return
        print(f"\nRunning experiments from custom CSV: {args.csv}")
        run_experiments_from_csv(args.csv, args.env, args.timesteps)
    elif args.member:
        csv_path = f"experiments_member{args.member}.csv"
        if not os.path.exists(csv_path):
            print(f"Error: {csv_path} not found!")
            return
        print(f"\nRunning all experiments for Member {args.member} from {csv_path}")
        run_experiments_from_csv(csv_path, args.env, args.timesteps, member_num=args.member)
    else:
        print("Usage: python train.py <member_number>")
        print("   or: python train.py --csv <experiments.csv>")
        print("Example: python train.py 1")
        print("         python train.py 2")
        print("         python train.py 3")
        print("         python train.py --csv experiments_member2_tuned.csv")


if __name__ == "__main__":
    main()

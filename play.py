"""
play.py - Load a trained DQN model and watch it play an Atari game.

Usage:
    python play.py --model models/experiment_final.zip --env BreakoutNoFrameskip-v0
    python play.py --model models/experiment_final.zip --env BreakoutNoFrameskip-v0 --record
"""

import argparse
import os
import time

import gymnasium as gym
import ale_py
gym.register_envs(ale_py)
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
import numpy as np


def make_play_env(env_id, record_video=False, video_dir="videos"):
    """Create an Atari environment for playing/evaluation."""
    def _init():
        render_mode = "rgb_array" if record_video else "human"
        env = gym.make(env_id, render_mode=render_mode)
        if record_video:
            env = gym.wrappers.RecordVideo(env, video_dir, episode_trigger=lambda x: True)
        env = AtariWrapper(env)
        return env
    return _init


def play(model_path, env_id, num_episodes=5, record_video=False, video_dir="videos"):
    """Load a trained model and play episodes using greedy policy (deterministic=True)."""

    if record_video:
        os.makedirs(video_dir, exist_ok=True)

    print(f"\nLoading model from: {model_path}")
    print(f"Environment: {env_id}")
    print(f"Episodes: {num_episodes}")
    print(f"Policy: Greedy (deterministic)\n")

    # Create environment
    env = DummyVecEnv([make_play_env(env_id, record_video, video_dir)])
    env = VecFrameStack(env, n_stack=4)

    # Load trained model
    model = DQN.load(model_path, env=env)

    episode_rewards = []
    episode_lengths = []

    for episode in range(num_episodes):
        obs = env.reset()
        total_reward = 0
        steps = 0
        done = False

        while not done:
            # Greedy policy: deterministic=True selects action with highest Q-value
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward[0]
            steps += 1

            if not record_video:
                env.render()
                time.sleep(0.03)  # Slow down for human viewing

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        print(f"Episode {episode + 1}: Reward = {total_reward:.1f}, Steps = {steps}")

        if not record_video:
            time.sleep(1)  # Pause between episodes

    env.close()

    # Summary
    print(f"\n{'='*40}")
    print(f"Results over {num_episodes} episodes:")
    print(f"  Mean Reward:  {np.mean(episode_rewards):.2f} +/- {np.std(episode_rewards):.2f}")
    print(f"  Mean Length:  {np.mean(episode_lengths):.1f}")
    print(f"  Best Reward:  {np.max(episode_rewards):.1f}")
    print(f"  Worst Reward: {np.min(episode_rewards):.1f}")
    print(f"{'='*40}")

    if record_video:
        print(f"\nVideos saved to: {video_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Play Atari with a trained DQN agent")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to trained model (.zip)")
    parser.add_argument("--env", type=str, default="BreakoutNoFrameskip-v0",
                        help="Atari environment ID")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Number of episodes to play")
    parser.add_argument("--record", action="store_true",
                        help="Record video of gameplay")
    parser.add_argument("--video-dir", type=str, default="videos",
                        help="Directory to save recorded videos")

    args = parser.parse_args()

    play(
        model_path=args.model,
        env_id=args.env,
        num_episodes=args.episodes,
        record_video=args.record,
        video_dir=args.video_dir,
    )


if __name__ == "__main__":
    main()

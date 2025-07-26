import argparse
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from pathlib import Path
import pandas as pd
from typing import Dict


class RolloutAnalyzer:
    def __init__(self, path: str):
        """
        Initialize analyzer with path to directory containing .npy files

        Args:
            data_path: Path to directory containing rollout_buffer_*.npy files
        """
        self.data_path = path + "logs/"
        self.save_path = path
        self.batch_data = {}
        self.summary_stats = {}

    def load_all_batches(self) -> Dict[int, Dict]:
        """Load all rollout buffer files and return combined data"""
        # Find all rollout buffer files
        pattern = os.path.join(self.data_path, "rollout_buffer_*.npy")
        files = glob.glob(pattern)

        if not files:
            print(f"No rollout buffer files found in {self.data_path}")
            return {}

        print(f"Found {len(files)} rollout buffer files")

        # Sort files by batch number
        files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))

        for file_path in files:
            # Extract batch number from filename
            batch_num = int(Path(file_path).stem.split("_")[-1])

            # Load data
            data = np.load(file_path, allow_pickle=True).item()
            self.batch_data[batch_num] = data

            print(f"Loaded batch {batch_num}: {len(data['rewards'])} timesteps")

        return self.batch_data

    def compute_summary_stats(self) -> Dict:
        """Compute summary statistics across all batches"""
        if not self.batch_data:
            self.load_all_batches()

        all_rewards = []
        all_values = []
        all_actions = []
        all_logprobs = []
        success_rates = []
        episode_lengths = []

        batch_summaries = {}

        for batch_num, data in self.batch_data.items():
            rewards = data["rewards"]
            values = data["state_values"]
            actions = data["actions"]
            logprobs = data["logprobs"]
            terminals = data["is_terminals"]

            # Collect all data
            all_rewards.extend(rewards)
            all_values.extend(values)
            all_actions.extend(actions.flatten() if actions.ndim > 1 else actions)
            all_logprobs.extend(logprobs)

            # Calculate episode statistics
            episode_rewards = []
            episode_lengths_batch = []
            current_episode_reward = 0
            current_episode_length = 0

            for i, (reward, terminal) in enumerate(zip(rewards, terminals)):
                current_episode_reward += reward
                current_episode_length += 1

                if terminal:
                    episode_rewards.append(current_episode_reward)
                    episode_lengths_batch.append(current_episode_length)
                    current_episode_reward = 0
                    current_episode_length = 0

            # Success rate (assuming reward > 0 means success)
            success_rate = (
                sum(1 for r in episode_rewards if r > 0) / len(episode_rewards)
                if episode_rewards
                else 0
            )
            success_rates.append(success_rate)
            episode_lengths.extend(episode_lengths_batch)

            # Batch summary
            batch_summaries[batch_num] = {
                "mean_reward": np.mean(rewards),
                "std_reward": np.std(rewards),
                "mean_value": np.mean(values),
                "std_value": np.std(values),
                "mean_episode_reward": (
                    np.mean(episode_rewards) if episode_rewards else 0
                ),
                "success_rate": success_rate,
                "num_episodes": len(episode_rewards),
                "mean_episode_length": (
                    np.mean(episode_lengths_batch) if episode_lengths_batch else 0
                ),
            }

        # Overall statistics
        self.summary_stats = {
            "batch_summaries": batch_summaries,
            "overall": {
                "total_timesteps": len(all_rewards),
                "total_batches": len(self.batch_data),
                "mean_reward": np.mean(all_rewards),
                "std_reward": np.std(all_rewards),
                "min_reward": np.min(all_rewards),
                "max_reward": np.max(all_rewards),
                "mean_value": np.mean(all_values),
                "std_value": np.std(all_values),
                "mean_success_rate": np.mean(success_rates),
                "action_distribution": np.bincount(all_actions),
                "mean_episode_length": (
                    np.mean(episode_lengths) if episode_lengths else 0
                ),
            },
        }

        return self.summary_stats

    def print_analysis(self):
        """Print comprehensive analysis of the rollout data"""
        if not self.summary_stats:
            self.compute_summary_stats()

        print("=" * 80)
        print("ROLLOUT BUFFER ANALYSIS")
        print("=" * 80)

        overall = self.summary_stats["overall"]
        print(f"📊 OVERALL STATISTICS:")
        print(f"   Total timesteps: {overall['total_timesteps']:,}")
        print(f"   Total batches: {overall['total_batches']}")
        print(f"   Mean episode length: {overall['mean_episode_length']:.1f}")
        print()

        print(f"💰 REWARD STATISTICS:")
        print(f"   Mean reward: {overall['mean_reward']:.3f}")
        print(f"   Std reward: {overall['std_reward']:.3f}")
        print(f"   Min reward: {overall['min_reward']:.3f}")
        print(f"   Max reward: {overall['max_reward']:.3f}")
        print()

        print(f"🎯 VALUE FUNCTION:")
        print(f"   Mean value: {overall['mean_value']:.3f}")
        print(f"   Std value: {overall['std_value']:.3f}")
        print()

        print(f"🎮 ACTION DISTRIBUTION:")
        action_dist = overall["action_distribution"]
        for i, count in enumerate(action_dist):
            percentage = (count / overall["total_timesteps"]) * 100
            print(f"   Action {i}: {count:,} times ({percentage:.1f}%)")
        print()

        print(f"🏆 SUCCESS RATE:")
        print(f"   Mean success rate: {overall['mean_success_rate']:.1%}")
        print()

        print("📈 BATCH-BY-BATCH PROGRESS:")
        print("Batch | Mean Reward | Mean Value | Success Rate | Episodes | Avg Length")
        print("-" * 70)

        for batch_num in sorted(self.summary_stats["batch_summaries"].keys()):
            stats = self.summary_stats["batch_summaries"][batch_num]
            print(
                f"{batch_num:5d} | {stats['mean_reward']:11.3f} | {stats['mean_value']:10.3f} | "
                f"{stats['success_rate']:11.1%} | {stats['num_episodes']:8d} | {stats['mean_episode_length']:10.1f}"
            )

        print("\n" + "=" * 80)

        # Learning indicators
        print("🔍 LEARNING INDICATORS:")
        batches = sorted(self.summary_stats["batch_summaries"].keys())
        if len(batches) >= 2:
            first_batch = self.summary_stats["batch_summaries"][batches[0]]
            last_batch = self.summary_stats["batch_summaries"][batches[-1]]

            reward_change = last_batch["mean_reward"] - first_batch["mean_reward"]
            value_change = last_batch["mean_value"] - first_batch["mean_value"]
            success_change = last_batch["success_rate"] - first_batch["success_rate"]

            print(f"   Reward change: {reward_change:+.3f} (first to last batch)")
            print(f"   Value change: {value_change:+.3f} (first to last batch)")
            print(
                f"   Success rate change: {success_change:+.1%} (first to last batch)"
            )

            if reward_change > 0.01:
                print("   ✅ Rewards are improving!")
            elif abs(reward_change) < 0.01:
                print("   ⚠️  Rewards are stagnant")
            else:
                print("   ❌ Rewards are decreasing")

        # Check for common issues
        print("\n🚨 POTENTIAL ISSUES:")

        # Check if all rewards are the same
        if overall["std_reward"] < 1e-6:
            print("   ❌ All rewards are identical - sparse reward problem?")

        # Check if values are learning
        if overall["std_value"] < 1e-6:
            print("   ❌ Value function is not learning (all values identical)")

        # Check action distribution
        max_action_pct = np.max(action_dist) / overall["total_timesteps"]
        if max_action_pct > 0.8:
            print(
                f"   ⚠️  Policy is very deterministic ({max_action_pct:.1%} on one action)"
            )
        elif max_action_pct < 0.3:
            print(f"   ⚠️  Policy is very random (max action only {max_action_pct:.1%})")

        # Check success rate
        if overall["mean_success_rate"] < 0.01:
            print("   ⚠️  Very low success rate - environment might be too hard")

        print("=" * 80)

    def plot_training_progress(self, name: str):
        """Create comprehensive plots of training progress"""
        file_path = self.save_path + name
        if not self.summary_stats:
            self.compute_summary_stats()

        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle("Rollout Buffer Analysis", fontsize=16)

        # Extract data for plotting
        batches = sorted(self.summary_stats["batch_summaries"].keys())
        batch_rewards = [
            self.summary_stats["batch_summaries"][b]["mean_reward"] for b in batches
        ]
        batch_values = [
            self.summary_stats["batch_summaries"][b]["mean_value"] for b in batches
        ]
        batch_success = [
            self.summary_stats["batch_summaries"][b]["success_rate"] for b in batches
        ]
        episode_rewards = [
            self.summary_stats["batch_summaries"][b]["mean_episode_reward"]
            for b in batches
        ]
        episode_lengths = [
            self.summary_stats["batch_summaries"][b]["mean_episode_length"]
            for b in batches
        ]

        # Plot 1: Mean reward per batch
        axes[0, 0].plot(batches, batch_rewards, "b-o", markersize=4)
        axes[0, 0].set_title("Mean Reward per Batch")
        axes[0, 0].set_xlabel("Batch")
        axes[0, 0].set_ylabel("Mean Reward")
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Mean value per batch
        axes[0, 1].plot(batches, batch_values, "g-o", markersize=4)
        axes[0, 1].set_title("Mean Value Function per Batch")
        axes[0, 1].set_xlabel("Batch")
        axes[0, 1].set_ylabel("Mean Value")
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Success rate per batch
        axes[0, 2].plot(batches, batch_success, "r-o", markersize=4)
        axes[0, 2].set_title("Success Rate per Batch")
        axes[0, 2].set_xlabel("Batch")
        axes[0, 2].set_ylabel("Success Rate")
        axes[0, 2].grid(True, alpha=0.3)
        axes[0, 2].set_ylim(0, 1)

        # Plot 4: Episode rewards
        axes[1, 0].plot(batches, episode_rewards, "m-o", markersize=4)
        axes[1, 0].set_title("Mean Episode Reward per Batch")
        axes[1, 0].set_xlabel("Batch")
        axes[1, 0].set_ylabel("Mean Episode Reward")
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 5: Episode lengths
        axes[1, 1].plot(batches, episode_lengths, "c-o", markersize=4)
        axes[1, 1].set_title("Mean Episode Length per Batch")
        axes[1, 1].set_xlabel("Batch")
        axes[1, 1].set_ylabel("Mean Episode Length")
        axes[1, 1].grid(True, alpha=0.3)

        # Plot 6: Action distribution (last batch)
        if batches:
            last_batch_data = self.batch_data[batches[-1]]
            actions = (
                last_batch_data["actions"].flatten()
                if last_batch_data["actions"].ndim > 1
                else last_batch_data["actions"]
            )
            action_counts = np.bincount(actions)
            axes[1, 2].bar(range(len(action_counts)), action_counts)
            axes[1, 2].set_title(f"Action Distribution (Batch {batches[-1]})")
            axes[1, 2].set_xlabel("Action")
            axes[1, 2].set_ylabel("Count")
            axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()

        plt.savefig(file_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {file_path}")

    def export_summary_csv(self, name: str):
        """Export summary statistics to CSV for further analysis"""
        file_path = self.save_path + name
        if not self.summary_stats:
            self.compute_summary_stats()

        # Create DataFrame from batch summaries
        df = pd.DataFrame.from_dict(
            self.summary_stats["batch_summaries"], orient="index"
        )
        df.index.name = "batch"
        df.to_csv(file_path)
        print(f"Summary exported to {file_path}")


def entry_point():
    parser = argparse.ArgumentParser(description="Analyze rollout results.")
    parser.add_argument(
        "--agent",
        type=str,
        required=True,
        help="Path to the directory containing rollout logs",
    )

    args = parser.parse_args()
    default = f"results/{args.agent}/"
    analyzer = RolloutAnalyzer(default)
    analyzer.load_all_batches()
    analyzer.print_analysis()
    analyzer.plot_training_progress(name="training_plots.png")
    analyzer.export_summary_csv(name="training_summary.csv")


if __name__ == "__main__":
    entry_point()

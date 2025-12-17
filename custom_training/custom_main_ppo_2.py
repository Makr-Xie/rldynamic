#!/usr/bin/env python3
"""
Custom PPO Trainer with Reward Tracking (No Gradient Analysis)

This module provides a CustomRayPPOTracker that records correctness,
format, and length rewards for each step without computing gradients.
"""

import sys
import os

sys.path.insert(0, '/home/qian.niu/Takoai/Medical_Reasoning/Mark/rldynamics')

import json
from datetime import datetime
import verl.trainer.ppo.ray_trainer
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer import main_ppo
from utils.rewards import compute_component_scores


# ========== CustomRayPPOTracker Class ==========
class CustomRayPPOTracker(RayPPOTrainer):
    """
    Custom PPO Trainer that tracks reward components without gradient analysis.

    This class records correctness, format, and length rewards for each
    training step and saves them to a single JSON file.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the custom tracker."""
        print("Initializing CustomRayPPOTracker...")
        super().__init__(*args, **kwargs)

        # Initialize tracking components
        self.reward_tracking_counter = 0
        self.reward_history = {
            "metadata": {
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "model": "qwen3_4b",
                "dataset": "gsm8k"
            },
            "steps": []
        }

        # Create save directory
        self.save_dir = "/home/qian.niu/Takoai/Medical_Reasoning/Mark/rldynamics/reward_tracking_math500"
        os.makedirs(self.save_dir, exist_ok=True)

        print(f"✅ CustomRayPPOTracker initialized successfully!")
        print(f"📂 Rewards will be saved to: {self.save_dir}/all_rewards.json")


    def _wrap_update_actor(self):
        """Wrap the update_actor method to track reward components."""
        print("Wrapping update_actor for reward tracking...")

        original_update_actor = self.actor_rollout_wg.update_actor

        def wrapped_update_actor(batch):
            print(f"\n======== [Step {self.reward_tracking_counter}] Reward tracking triggered! ========")

            # 1. Extract data from batch
            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
            sample_gts = [item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in batch]

            # 2. Compute component rewards and actual lengths
            correctness_rewards = []
            format_rewards = []
            length_rewards = []
            actual_lengths = []

            for response_text, gt in zip(outputs, sample_gts):
                scores = compute_component_scores(
                    data_source="gsm8k",
                    solution_str=response_text,
                    ground_truth=gt if gt is not None else "",
                    extra_info=None
                )
                correctness_rewards.append(scores['correctness_binary'])
                format_rewards.append(scores['format_binary'])
                length_rewards.append(scores['length_binary'])
                actual_lengths.append(len(response_text))  # Record actual character length

            # 3. Build current step data
            step_data = {
                "step": self.reward_tracking_counter,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "num_samples": len(correctness_rewards),
                "statistics": {
                    "avg_correctness": sum(correctness_rewards) / len(correctness_rewards),
                    "avg_format": sum(format_rewards) / len(format_rewards),
                    "avg_length_reward": sum(length_rewards) / len(length_rewards),
                    "avg_actual_length": sum(actual_lengths) / len(actual_lengths),
                    "correct_count": int(sum(correctness_rewards)),
                    "format_valid_count": sum(1 for f in format_rewards if f > 0),
                    "total_samples": len(correctness_rewards)
                },
                "samples": [
                    {
                        "idx": i,
                        "correctness": correctness_rewards[i],
                        "format": format_rewards[i],
                        "length_reward": length_rewards[i],
                        "actual_length": actual_lengths[i]
                    }
                    for i in range(len(correctness_rewards))
                ]
            }

            # 4. Append to history and save
            self.reward_history["steps"].append(step_data)
            self._save_rewards()

            # 5. Print statistics
            print(f"\n📊 [Step {self.reward_tracking_counter}] Reward Statistics:")
            print(f"  Correctness: {step_data['statistics']['avg_correctness']:.3f} "
                  f"({step_data['statistics']['correct_count']}/{step_data['statistics']['total_samples']})")
            print(f"  Format:      {step_data['statistics']['avg_format']:.3f} "
                  f"(valid: {step_data['statistics']['format_valid_count']})")
            print(f"  Length Reward: {step_data['statistics']['avg_length_reward']:.3f}")
            print(f"  Actual Length: {step_data['statistics']['avg_actual_length']:.1f} chars")

            self.reward_tracking_counter += 1

            # 6. Execute normal update
            return original_update_actor(batch)

        self.actor_rollout_wg.update_actor = wrapped_update_actor


    def _save_rewards(self):
        """Save all reward history to a single JSON file."""
        save_path = os.path.join(self.save_dir, "all_rewards.json")
        try:
            # Update last_saved timestamp
            self.reward_history["metadata"]["last_saved"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            with open(save_path, 'w') as f:
                json.dump(self.reward_history, f, indent=2)

            current_time = datetime.now().strftime("%H:%M:%S")
            print(f"  💾 Saved to: {save_path} at {current_time}")
        except Exception as e:
            print(f"  ❌ Failed to save rewards: {e}")


    def fit(self, *args, **kwargs):
        """Override fit method to add reward tracking logic."""
        print("🚀 Starting training with CustomRayPPOTracker...")

        # Wrap the update_actor method before training starts
        self._wrap_update_actor()

        # Run the original fit method
        result = super().fit(*args, **kwargs)

        # Final save to ensure all data is persisted
        print("\n" + "="*80)
        print("✅ Training completed! Final save of reward data...")
        self._save_rewards()
        print(f"📊 Total steps tracked: {self.reward_tracking_counter}")
        print("="*80 + "\n")

        return result


# Monkey patch: Replace the original RayPPOTrainer with our custom version
print("🔄 Applying monkey patch...")
verl.trainer.ppo.ray_trainer.RayPPOTrainer = CustomRayPPOTracker
main_ppo.RayPPOTrainer = CustomRayPPOTracker
print("✅ Monkey patch applied successfully!")


if __name__ == "__main__":
    """Entry point that uses the original main function with our custom trainer."""
    print("🎯 Starting custom PPO training with reward tracking...")
    from verl.trainer.main_ppo import main
    main()

#!/usr/bin/env python3
"""
Custom PPO Trainer with Gradient Similarity Analysis

This module provides a clean CustomRayPPOTrainer that inherits from RayPPOTrainer
and can be extended with gradient analysis functionality.
"""

import sys
import os


import torch
import numpy as np
import threading
from typing import Dict, Any
import verl.trainer.ppo.ray_trainer
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer import main_ppo
from utils.rewards import compute_component_scores

# ========== Token-Level Reward Conversion Functions ==========
def convert_correctness_to_token_level(correctness_reward, response_mask):
    """Convert correctness reward to token-level (last token only)."""
    token_rewards = torch.zeros_like(response_mask, dtype=torch.float32)
    last_indices = response_mask.sum(dim=-1).long() - 1
    for i, idx in enumerate(last_indices):
        if idx >= 0:
            token_rewards[i, idx] = correctness_reward[i]
    return token_rewards


def convert_format_to_token_level(format_reward, response_tokens, response_mask, tokenizer):
    """Convert format reward to token-level (#### + answer tokens)."""
    if not isinstance(format_reward, torch.Tensor):
        format_reward = torch.tensor(format_reward, dtype=torch.float32)
    
    token_rewards = torch.zeros_like(response_mask, dtype=torch.float32)
    
    # Tokenize "####" to get its token ID(s)
    hash_tokens = tokenizer.encode("####", add_special_tokens=False)
    
    for i in range(len(format_reward)):
        if format_reward[i] == 0:
            continue  # No format reward, skip
        
        # Find "####" pattern in response tokens
        response_ids = response_tokens[i].cpu().tolist()
        
        # Search for "####" pattern
        format_start = -1
        for j in range(len(response_ids) - len(hash_tokens) + 1):
            if response_ids[j:j+len(hash_tokens)] == hash_tokens:
                format_start = j
                break
        
        if format_start >= 0:
            # Find the end of response (last valid token)
            last_idx = int(response_mask[i].sum()) - 1
            
            # Distribute format reward across "####" and answer tokens
            num_format_tokens = last_idx - format_start + 1
            if num_format_tokens > 0:
                token_rewards[i, format_start:last_idx+1] = format_reward[i] / num_format_tokens
    
    return token_rewards
    

def convert_length_to_token_level(length_reward, response_mask):
    """Convert length reward to token-level (average across all tokens)."""
    token_rewards = length_reward.unsqueeze(-1) * response_mask
    num_tokens = response_mask.sum(dim=-1, keepdim=True)
    token_rewards = token_rewards / num_tokens.clamp(min=1)
    return token_rewards



# ========== CustomRayPPOTrainer Class ==========
class CustomRayPPOTrainer(RayPPOTrainer):
    """
    Custom PPO Trainer that extends RayPPOTrainer.

    This class can be used to inject custom functionality into the training loop
    while maintaining compatibility with the original veRL framework.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the custom trainer."""
        print("Initializing CustomRayPPOTrainer...")
        super().__init__(*args, **kwargs)

        # Initialize custom components
        self.gradient_analysis_counter = 0
        self.similarity_history = []
        self.save_threads = []  # Track all background save threads

        print("✅ CustomRayPPOTrainer initialized successfully!")

    
    def _wrap_update_actor(self):
        print("Wrapping update_actor for gradient similarity analysis...")

        original_update_actor = self.actor_rollout_wg.update_actor

        def wrapped_update_actor(batch):
            print(f"\n ======== [Update {self.gradient_analysis_counter}] Gradient analysis triggered! ========")
            
            ## ========= Extract Data ==========
            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
            sample_gts = [item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in batch]
            response_mask = batch.batch["response_mask"]
            response_tokens = batch.batch["responses"]

            ## ========= Compute Component Rewards ==========
            correctness_rewards = []
            format_rewards = []
            length_rewards = []

            for response_text, gt in zip(outputs, sample_gts):
                scores = compute_component_scores(
                    data_source="gsm8k",
                    solution_str=response_text,
                    ground_truth=gt if gt is not None else "",  # 使用实际的GT
                    extra_info=None
                )
                correctness_rewards.append(scores['correctness_binary'])
                format_rewards.append(scores['format_binary'])
                length_rewards.append(scores['length_binary'])

            # ========== 添加这些测试输出 ==========
            print(f"✅ Component rewards computed!")
            print(f"  Total samples: {len(correctness_rewards)}")
            print(f"  Correctness: avg={sum(correctness_rewards)/len(correctness_rewards):.3f}, "
                f"correct_count={sum(correctness_rewards)}/{len(correctness_rewards)}")
            print(f"  Format: avg={sum(format_rewards)/len(format_rewards):.3f}, "
                f"valid_count={sum(format_rewards)}/{len(format_rewards)}")
            print(f"  Length: avg={sum(length_rewards)/len(length_rewards):.3f}, "
                f"valid_count={sum(length_rewards)}/{len(length_rewards)}")
            
            # ========================================

            ## ========= Convert to Token-Level Rewards ==========
            correctness_tensor = torch.tensor(correctness_rewards, dtype=torch.float32, device=response_mask.device)
            format_tensor = torch.tensor(format_rewards, dtype=torch.float32, device=response_mask.device)
            length_tensor = torch.tensor(length_rewards, dtype=torch.float32, device=response_mask.device)

            ## 暂时全部都用last token reward
            token_level_correctness = convert_correctness_to_token_level(correctness_tensor, response_mask)
            # token_level_format = convert_format_to_token_level(format_tensor, response_tokens, response_mask, self.tokenizer)
            token_level_format = convert_correctness_to_token_level(format_tensor, response_mask)
            # token_level_length = convert_length_to_token_level(length_tensor, response_mask)
            token_level_length = convert_correctness_to_token_level(length_tensor, response_mask)
            

            # ========== Compute Component-Specific Advantages ==========
            print(f"\n🔄 Computing component-specific advantages...")
            from verl.trainer.ppo.ray_trainer import compute_advantage
            import copy

            component_advantages = {}
            component_names = ['correctness', 'format', 'length']
            component_rewards = {
                'correctness': token_level_correctness,
                'format': token_level_format,
                'length': token_level_length
            }

            for component in component_names:
                temp_batch = copy.deepcopy(batch)
                temp_batch.batch["token_level_rewards"] = component_rewards[component]

                temp_batch = compute_advantage(
                    temp_batch,
                    adv_estimator=self.config.algorithm.adv_estimator,
                    gamma=self.config.algorithm.gamma,
                    lam=self.config.algorithm.lam,
                    num_repeat=self.config.actor_rollout_ref.rollout.n,
                    norm_adv_by_std_in_grpo=self.config.algorithm.get("norm_adv_by_std_in_grpo", True),
                    config=self.config.algorithm
                )

                # Store advantages
                component_advantages[component] = temp_batch.batch["advantages"]
                # print(f"  ✅ {component}: advantages computed (shape={component_advantages[component].shape})")

            print(f"✅ Component-specific advantages computed!")

            # ========== Compute Gradients for Each Component ==========
            print("\n🔄 Computing component-specific gradients...")
            component_gradients = {}
            component_gradients_by_layer = {}

            for component_name in ['correctness', 'format', 'length']:
                print(f"  Computing gradients for: {component_name}")
                batch_copy = copy.deepcopy(batch)
                batch_copy.batch["advantages"] = component_advantages[component_name]
                batch_copy.meta_info["compute_gradients_only"] = True

                gradient_results = original_update_actor(batch_copy)

                for result in gradient_results:
                    grad_vector = result.non_tensor_batch.get("gradient_vector")
                    grad_dict = result.non_tensor_batch.get("gradient_dict")
                    rank = result.non_tensor_batch.get("rank", -1)

                    if grad_vector is not None and rank == 0:
                        component_gradients[component_name] = grad_vector
                        if grad_dict is not None and len(grad_dict) > 0:
                            component_gradients_by_layer[component_name] = grad_dict
                        print(f"  ✅ {component_name}: gradient vector size = {len(grad_vector)}")
                        break
            
            # ========== Compute Gradient Similarity ==========
            print(f"\n📊 Gradient collection results:")
            print(f"  Total components collected: {len(component_gradients)}")
            for comp_name, grad_vec in component_gradients.items():
                print(f"  {comp_name}: vector size = {len(grad_vec)}, norm = {np.linalg.norm(grad_vec):.6f}")

            # 计算余弦相似度（如果收集到了所有三个组件）
            if len(component_gradients) == 3:
                print(f"\n📐 Computing gradient cosine similarity...")

                # 转换为 numpy array（如果还不是的话）
                grad_arrays = {}
                for name, grad_vec in component_gradients.items():
                    if isinstance(grad_vec, list):
                        grad_arrays[name] = np.array(grad_vec)
                    else:
                        grad_arrays[name] = grad_vec

                # 计算余弦相似度
                def cosine_similarity(v1, v2):
                    dot_product = np.dot(v1, v2)
                    norm1 = np.linalg.norm(v1)
                    norm2 = np.linalg.norm(v2)
                    return dot_product / (norm1 * norm2)

                sim_correct_format = cosine_similarity(grad_arrays['correctness'], grad_arrays['format'])
                sim_correct_length = cosine_similarity(grad_arrays['correctness'], grad_arrays['length'])
                sim_format_length = cosine_similarity(grad_arrays['format'], grad_arrays['length'])

                print(f"\n📊 Gradient Similarity Matrix:")
                print(f"  Correctness vs Format:  {sim_correct_format:.6f}")
                print(f"  Correctness vs Length:  {sim_correct_length:.6f}")
                print(f"  Format vs Length:       {sim_format_length:.6f}")
            else:
                print(f"⚠️ Warning: Only collected {len(component_gradients)}/3 components, skipping similarity computation")

            # ========== Save Gradients by Layer (Async) ==========
            if len(component_gradients_by_layer) == 3:
                print(f"\n💾 Preparing to save gradients asynchronously...")

                # 创建保存目录
                save_dir = "/home/qian.niu/Takoai/Medical_Reasoning/Mark/rldynamics/gradient_analysis_math500"
                os.makedirs(save_dir, exist_ok=True)

                # 保存文件名：包含 step 信息
                filename = f"{save_dir}/gradients_step_{self.gradient_analysis_counter}.npz"

                # 准备保存的数据
                save_data = {}
                for comp_name, grad_dict in component_gradients_by_layer.items():
                    for param_name, grad_array in grad_dict.items():
                        # key 格式：component_name/param_name
                        save_data[f"{comp_name}/{param_name}"] = grad_array

                # 异步保存函数
                def save_gradients_async(data, file_path):
                    try:
                        np.savez_compressed(file_path, **data)
                        print(f"  ✅ Background save completed: {file_path}")
                    except Exception as e:
                        print(f"  ❌ Background save failed: {e}")

                # 启动后台线程保存
                thread = threading.Thread(target=save_gradients_async, args=(save_data, filename))
                thread.daemon = False  # 不设为 daemon，确保保存完成
                thread.start()
                self.save_threads.append(thread)  # 记录线程

                print(f"  🔄 Saving gradients in background to: {filename}")
                print(f"  📦 Total entries to save: {len(save_data)} ({len(component_gradients_by_layer)} components × {len(grad_dict)} params)")
                print(f"  📝 Active save threads: {len(self.save_threads)}")
            else:
                print(f"⚠️ Warning: Only collected {len(component_gradients_by_layer)}/3 component gradients by layer, skipping file save")

            self.gradient_analysis_counter += 1            

            return original_update_actor(batch)

        self.actor_rollout_wg.update_actor = wrapped_update_actor

    def wait_for_saves(self):
        """Wait for all background save threads to complete."""
        if hasattr(self, 'save_threads') and self.save_threads:
            print(f"\n{'='*80}")
            print(f"⏳ Waiting for {len(self.save_threads)} background save operations to complete...")
            print(f"{'='*80}")
            for i, thread in enumerate(self.save_threads):
                if thread.is_alive():
                    print(f"  Waiting for save thread {i+1}/{len(self.save_threads)}...")
                    thread.join()  # Wait for thread to complete
                    print(f"  ✅ Save thread {i+1}/{len(self.save_threads)} completed")
                else:
                    print(f"  ✅ Save thread {i+1}/{len(self.save_threads)} already completed")
            print(f"\n✅ All {len(self.save_threads)} gradient saves completed successfully!")
            print(f"{'='*80}\n")

    def fit(self, *args, **kwargs):
        """Override fit method to add custom training logic."""
        print("🚀 Starting training with CustomRayPPOTrainer...")

        self._wrap_update_actor()

        try:
            result = super().fit(*args, **kwargs)
        finally:
            # Always wait for saves to complete, even if training errors out
            self.wait_for_saves()

        return result





# Monkey patch: Replace the original RayPPOTrainer with our custom version
print("🔄 Applying monkey patch...")
verl.trainer.ppo.ray_trainer.RayPPOTrainer = CustomRayPPOTrainer
main_ppo.RayPPOTrainer = CustomRayPPOTrainer
print("✅ Monkey patch applied successfully!")


if __name__ == "__main__":
    """Entry point that uses the original main function with our custom trainer."""
    print("🎯 Starting custom PPO training...")
    from verl.trainer.main_ppo import main
    main()
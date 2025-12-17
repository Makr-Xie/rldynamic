#!/usr/bin/env python3
"""
Custom PPO Trainer with Token-Level Perplexity Tracking

This module provides a CustomRayPPOTrainer that records token-level log probabilities
and rewards during validation for analysis.
"""

import sys
import os
import json
import uuid
from collections import defaultdict

sys.path.insert(0, '/home/qian.niu/Takoai/Medical_Reasoning/Mark/rldynamics')

import torch
import numpy as np

import verl.trainer.ppo.ray_trainer
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, compute_response_mask
from verl.trainer.ppo.ray_trainer import pad_dataproto_to_divisor, unpad_dataproto
from verl.trainer.ppo.metric_utils import process_validation_metrics
from verl.trainer import main_ppo
from verl import DataProto


class PPLTrackingTrainer(RayPPOTrainer):
    """
    Custom PPO Trainer that tracks token-level log probabilities during validation.

    For each validation prompt, generates N rollouts and records:
    - Token-level log probabilities
    - Token-level rewards
    - Response tokens
    """

    def __init__(self, *args, **kwargs):
        """Initialize the custom trainer."""
        print("Initializing PPLTrackingTrainer...")
        super().__init__(*args, **kwargs)

        # Directory to save validation results
        self.ppl_output_dir = '/home/qian.niu/Takoai/Medical_Reasoning/Mark/rldynamics/validation_ppl_tracking'
        os.makedirs(self.ppl_output_dir, exist_ok=True)

        print(f"PPL tracking output directory: {self.ppl_output_dir}")
        print("PPLTrackingTrainer initialized successfully!")

    def _validate(self):
        """
        Override validation to track token-level log probabilities.

        For each prompt, generates N rollouts and saves:
        - prompt, response, ground_truth
        - token-level log_probs and rewards
        - total reward
        """
        print(f"\n{'='*60}")
        print(f"PPL-tracking validation at step {self.global_steps}")
        print(f"{'='*60}")

        # ========== Part 1: 原始框架 - 数据收集变量 ==========
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        sample_uids = []

        # 用于 token-level 记录
        all_records = []
        n_rollouts = self.config.actor_rollout_ref.rollout.val_kwargs.n
        print(f"  Generating {n_rollouts} rollouts per prompt")

        # ========== 遍历 validation dataloader ==========
        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            # 分配 UID
            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch["input_ids"]))],
                    dtype=object
                )

            # 保存原始 UID（repeat 之前）
            original_uids = test_batch.non_tensor_batch["uid"].copy()
            original_batch_size = len(original_uids)

            # Repeat：每个 prompt 生成 N 个 rollout
            test_batch = test_batch.repeat(
                repeat_times=n_rollouts, interleave=True
            )

            # 跳过 model-based reward
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            # 存储 inputs
            input_ids = test_batch.batch["input_ids"]
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)
            sample_uids.extend(test_batch.non_tensor_batch["uid"])

            # 存储 ground truths
            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None)
                for item in test_batch
            ]
            sample_gts.extend(ground_truths)

            # ========== 生成 responses ==========
            test_gen_batch = self._get_gen_batch(test_batch)
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }

            # Pad to divisor
            size_divisor = (
                self.actor_rollout_wg.world_size
                if not self.async_rollout_mode
                else self.config.actor_rollout_ref.rollout.agent.num_workers
            )
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)

            # Generate
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            # Unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
            print(f"  Generated {len(test_output_gen_batch.batch['responses'])} responses")

            # 存储 outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            # Union batch
            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            # ========== 计算 response_mask ==========
            if "response_mask" not in test_batch.batch:
                test_batch.batch["response_mask"] = compute_response_mask(test_batch)

            # ========== Part 2: 计算 log probabilities ==========
            print(f"  Computing log probabilities...")

            # 准备 meta_info
            batch_size = len(test_batch.batch["input_ids"])
            test_batch.meta_info["micro_batch_size"] = min(batch_size, 32)
            test_batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
            test_batch.meta_info["use_dynamic_bsz"] = False

            # 调用 compute_log_prob
            log_prob_output = self.actor_rollout_wg.compute_log_prob(test_batch)
            log_probs = log_prob_output.batch["old_log_probs"]  # (batch, response_len)
            print(f"  Log probs shape: {log_probs.shape}")

            # ========== 计算 rewards ==========
            if self.val_reward_fn is None:
                raise ValueError("val_reward_fn must be provided for validation.")

            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]  # (batch, response_len)
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)

            # Collect num_turns
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            # Collect data source
            data_source_lst.append(
                test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0])
            )

            # ========== Part 3: 构建 token-level records ==========
            response_mask = test_batch.batch["response_mask"]
            response_ids = test_batch.batch["responses"]

            for i in range(batch_size):
                # 获取有效 token 数量
                valid_len = int(response_mask[i].sum().item())

                # 提取 token-level 数据（只取有效部分）
                token_ids_i = response_ids[i, :valid_len].cpu().tolist()
                token_log_probs_i = log_probs[i, :valid_len].cpu().tolist()
                token_rewards_i = reward_tensor[i, :valid_len].cpu().tolist()

                # Token IDs 转字符串
                token_strings = [self.tokenizer.decode([tid]) for tid in token_ids_i]

                # 计算 rollout index（interleave=True 时，连续样本来自同一 prompt）
                prompt_idx = i // n_rollouts
                rollout_idx = i % n_rollouts
                original_uid = original_uids[prompt_idx]

                record = {
                    "prompt": input_texts[i],
                    "uid": original_uid,
                    "ground_truth": ground_truths[i],
                    "response": output_texts[i],
                    "response_tokens": token_strings,
                    "token_log_probs": token_log_probs_i,
                    "reward": scores[i],
                    "step": self.global_steps,
                    "rollout_idx": rollout_idx,
                }

                # 添加 extra info（如 acc）
                if "reward_extra_info" in result:
                    for key, lst in result["reward_extra_info"].items():
                        record[key] = lst[i]

                all_records.append(record)

        # ========== Part 4: 保存 JSON ==========
        output_file = os.path.join(self.ppl_output_dir, f"validation_step_{self.global_steps}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_records, f, ensure_ascii=False, indent=2)

        print(f"\n  Saved {len(all_records)} records to {output_file}")

        # 计算并打印统计信息
        if len(all_records) > 0:
            avg_reward = np.mean([r["reward"] for r in all_records])
            avg_ppl = np.mean([
                np.exp(-np.mean(r["token_log_probs"]))
                for r in all_records if len(r["token_log_probs"]) > 0
            ])
            print(f"  Summary: avg_reward={avg_reward:.4f}, avg_ppl={avg_ppl:.4f}")

        # ========== 返回原有 metrics ==========
        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)
        data_src2var2metric2val = process_validation_metrics(data_sources, sample_uids, reward_extra_infos_dict)

        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        print(f"{'='*60}\n")
        return metric_dict


# Monkey patch
print("Applying PPLTrackingTrainer monkey patch...")
verl.trainer.ppo.ray_trainer.RayPPOTrainer = PPLTrackingTrainer
main_ppo.RayPPOTrainer = PPLTrackingTrainer
print("PPLTrackingTrainer monkey patch applied!")


if __name__ == "__main__":
    from verl.trainer.main_ppo import main
    main()

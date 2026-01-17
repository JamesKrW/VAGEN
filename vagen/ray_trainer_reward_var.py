# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
PPO Trainer with Ray-based single controller - Reward Variance Visualization Version.
This trainer computes and visualizes the relationship between entropy and reward variance.
Inherits from RayPPOTrainer and overrides only the necessary methods.
"""

import json
import os
import uuid
from collections import defaultdict
from pprint import pprint
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayWorkerGroup
from verl.trainer.ppo.utils import Role, WorkerType
from verl.utils.torch_functional import masked_mean

from .ray_trainer import (
    RayPPOTrainer,
    ResourcePoolManager,
    compute_response_mask,
)


def compute_group_entropy_and_reward_var(data: DataProto, entropys: torch.Tensor) -> dict:
    """Compute mean entropy and reward variance for each group.

    Args:
        data (DataProto): The data containing batch information.
        entropys (torch.Tensor): Entropy values for each trajectory (batch_size,)

    Returns:
        dict: Dictionary containing:
            - group_entropy: dict mapping group_idx to mean entropy
            - group_reward_var: dict mapping group_idx to reward variance
            - pairs: list of (entropy, reward_var) tuples
    """
    token_level_scores = data.batch["token_level_scores"]
    group_idx = (
        data.non_tensor_batch["group_idx"]
        if "group_idx" in data.non_tensor_batch
        else data.non_tensor_batch["uid"]
    )

    # Compute total reward per sample
    if isinstance(token_level_scores, torch.Tensor):
        total_rewards = token_level_scores.sum(dim=-1).detach().cpu().numpy()
    else:
        total_rewards = np.asarray(token_level_scores).sum(axis=-1)

    # Convert entropy to numpy
    if isinstance(entropys, torch.Tensor):
        entropys_np = entropys.detach().cpu().numpy()
    else:
        entropys_np = np.asarray(entropys)

    # Group rewards and entropys by group_idx
    group_rewards = defaultdict(list)
    group_entropys = defaultdict(list)

    for idx, (reward, entropy) in zip(group_idx, zip(total_rewards, entropys_np)):
        group_key = str(idx)
        group_rewards[group_key].append(float(reward))
        group_entropys[group_key].append(float(entropy))

    # Compute per-group mean entropy and reward variance
    group_entropy = {}
    group_reward_var = {}
    pairs = []

    for group_key in group_rewards.keys():
        rewards = group_rewards[group_key]
        entropys_list = group_entropys[group_key]

        # Mean entropy for the group
        mean_entropy = float(np.mean(entropys_list))
        group_entropy[group_key] = mean_entropy

        # Reward variance for the group (population variance, ddof=0)
        if len(rewards) <= 1:
            reward_var = 0.0
        else:
            reward_var = float(np.var(rewards, ddof=0))
        group_reward_var[group_key] = reward_var

        pairs.append((mean_entropy, reward_var))

    return {
        "group_entropy": group_entropy,
        "group_reward_var": group_reward_var,
        "pairs": pairs
    }


def plot_entropy_reward_var(pairs: list, save_path: str) -> float:
    """Plot scatter plot of entropy vs reward variance and compute R value.

    Args:
        pairs: List of (entropy, reward_var) tuples
        save_path: Path to save the plot

    Returns:
        float: Pearson correlation coefficient (R value)
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy import stats

    entropies = [p[0] for p in pairs]
    reward_vars = [p[1] for p in pairs]

    # Compute Pearson correlation
    if len(entropies) > 1:
        r_value, p_value = stats.pearsonr(entropies, reward_vars)
    else:
        r_value, p_value = 0.0, 1.0

    # Create scatter plot
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(entropies, reward_vars, alpha=0.6, edgecolors='black', linewidth=0.5)

    # Add trend line if there's enough data
    if len(entropies) > 1:
        z = np.polyfit(entropies, reward_vars, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(entropies), max(entropies), 100)
        ax.plot(x_line, p(x_line), "r--", alpha=0.8, label=f'Trend line (R={r_value:.4f})')

    ax.set_xlabel('Mean Entropy per Group', fontsize=12)
    ax.set_ylabel('Reward Variance per Group', fontsize=12)
    ax.set_title(f'Entropy vs Reward Variance (R={r_value:.4f}, p={p_value:.4e})', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Save plot
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved entropy-reward_var plot to {save_path}")
    return r_value


class RayPPOTrainerRewardVar(RayPPOTrainer):
    """Distributed PPO trainer variant for visualizing entropy-reward variance relationship.

    This trainer inherits from RayPPOTrainer and adds functionality to:
    1. Perform rollout and compute entropy for each trajectory
    2. Calculate reward variance per group
    3. Generate scatter plots showing entropy vs reward variance
    4. Compute correlation coefficient (R value)
    """

    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        """Initialize the reward variance visualization trainer."""
        super().__init__(
            config=config,
            tokenizer=tokenizer,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            processor=processor,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
            device_name=device_name,
        )

        # Get reward_var visualization config
        self.visualize_reward_var_entropy = self.config.trainer.get("visualize_reward_var_entropy", False)
        self.rollout_validation_n = self.config.trainer.get("rollout_validation_n", self.config.actor_rollout_ref.rollout.n)

    def _validate_reward_var(self):
        """Validate and compute entropy-reward variance relationship.

        This method:
        1. Performs rollout on validation data with configurable n repeats
        2. Computes entropy for each trajectory
        3. Computes reward for each trajectory
        4. Calculates mean entropy and reward variance per group
        5. Saves pairs to JSON file
        6. Plots scatter plot and computes R value
        """
        all_pairs = []
        all_group_data = {}

        # Determine n for validation rollout
        val_n = self.rollout_validation_n
        print(f"Running reward variance visualization with n={val_n} repeats per sample")

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                )

            # Repeat test batch with configurable n
            test_batch = test_batch.repeat(repeat_times=val_n, interleave=True)

            # Skip model-based reward validation
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                print("Skipping model-based reward validation")
                continue

            test_gen_batch = self._get_gen_batch(test_batch)

            if not self.concat_multi_turn:
                num_traj_per_sample = val_n
                self._assign_group_and_traj_idx(test_gen_batch, num_traj_per_sample)

            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }

            # Pad to be divisible by dp_size
            size_divisor = (
                self.actor_rollout_wg.world_size
                if not self.async_rollout_mode
                else self.config.actor_rollout_ref.rollout.agent.num_workers
            )

            if not self.concat_multi_turn:
                original_uids = set(test_gen_batch.non_tensor_batch["uid"])

            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)

            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            # Unpad
            if self.concat_multi_turn:
                test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
            else:
                valid_indices = [
                    i for i, uid in enumerate(test_output_gen_batch_padded.non_tensor_batch["group_idx"])
                    if uid in original_uids
                ]
                test_output_gen_batch = test_output_gen_batch_padded.select_idxs(valid_indices)

            print("Validation generation end")

            # Align and union, same as train epoch logic
            if self.concat_multi_turn:
                test_batch = test_batch.repeat(repeat_times=val_n, interleave=True)
                test_batch = test_batch.union(test_output_gen_batch)
            else:
                test_batch = self._post_process_no_concat_batch(test_batch, test_output_gen_batch)

            test_batch.meta_info["validate"] = True

            if "response_mask" not in test_batch.batch.keys():
                test_batch.batch["response_mask"] = compute_response_mask(test_batch)

            if self.config.trainer.balance_batch:
                if not self.concat_multi_turn:
                    divisor_size = self.actor_rollout_wg.world_size
                    batch_size = len(test_batch.batch["attention_mask"])
                    test_batch, pad_size = pad_dataproto_to_divisor(test_batch, divisor_size)
                    print(
                        f"Pad {pad_size} samples to make batch size {batch_size} divisible by {divisor_size} dp_workers"
                    )
                self._balance_batch(test_batch, metrics={})

            # Compute log_prob and entropy
            # Note: Worker sets meta_info internally (micro_batch_size, temperature, etc.)
            print("Computing log probabilities and entropy...")
            log_prob_output = self.actor_rollout_wg.compute_log_prob(test_batch)
            entropys = log_prob_output.batch["entropys"]  # (batch_size, response_length)

            # Aggregate entropy per trajectory (mean over response tokens)
            response_mask = test_batch.batch["response_mask"]
            # Mean entropy per trajectory
            traj_entropys = masked_mean(entropys, mask=response_mask, axis=-1)  # (batch_size,)

            # Compute rewards
            if self.val_reward_fn is None:
                raise ValueError("val_reward_fn must be provided for validation.")
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            test_batch.batch["token_level_scores"] = reward_tensor

            # Compute response_mask if not already present
            if "response_mask" not in test_batch.batch:
                test_batch.batch["response_mask"] = response_mask

            # Compute group entropy and reward variance
            group_data = compute_group_entropy_and_reward_var(test_batch, traj_entropys)
            all_pairs.extend(group_data["pairs"])

            # Merge group data
            for k, v in group_data["group_entropy"].items():
                if k not in all_group_data:
                    all_group_data[k] = {"entropy": v, "reward_var": group_data["group_reward_var"][k]}

        # Save results
        experiment_dir = self.config.trainer.default_local_dir
        os.makedirs(experiment_dir, exist_ok=True)

        # Save pairs to JSON
        pairs_file = os.path.join(experiment_dir, f"entropy_reward_var_pairs_step{self.global_steps}.json")
        with open(pairs_file, 'w') as f:
            json.dump({
                "pairs": all_pairs,
                "group_data": all_group_data,
                "config": {
                    "rollout_validation_n": val_n,
                    "global_steps": self.global_steps
                }
            }, f, indent=2)
        print(f"Saved entropy-reward_var pairs to {pairs_file}")

        # Plot and compute R value
        plot_path = os.path.join(experiment_dir, f"entropy_reward_var_plot_step{self.global_steps}.png")
        r_value = plot_entropy_reward_var(all_pairs, plot_path)

        # Print summary
        print(f"\n{'='*50}")
        print(f"Entropy-Reward Variance Analysis Summary")
        print(f"{'='*50}")
        print(f"Number of groups: {len(all_pairs)}")
        print(f"Pearson R value: {r_value:.4f}")
        print(f"Results saved to: {experiment_dir}")
        print(f"{'='*50}\n")

        return {
            "reward_var/r_value": r_value,
            "reward_var/num_groups": len(all_pairs),
            "reward_var/mean_entropy": np.mean([p[0] for p in all_pairs]) if all_pairs else 0.0,
            "reward_var/mean_reward_var": np.mean([p[1] for p in all_pairs]) if all_pairs else 0.0,
        }

    def fit(self):
        """
        Run reward variance visualization.
        This method performs validation rollout, computes entropy and reward variance,
        then generates visualization and exits.
        """
        from omegaconf import OmegaConf
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # Load checkpoint
        self._load_checkpoint()

        # Check if visualize_reward_var_entropy is enabled
        if not self.visualize_reward_var_entropy:
            print("Warning: visualize_reward_var_entropy is not enabled. Enabling it automatically.")
            self.visualize_reward_var_entropy = True

        # Perform reward variance visualization
        print("\n" + "="*60)
        print("Starting Entropy-Reward Variance Visualization")
        print("="*60 + "\n")

        val_metrics = self._validate_reward_var()

        # Log metrics
        logger.log(data=val_metrics, step=self.global_steps)

        pprint(f"Reward Variance Visualization Results: {val_metrics}")

        print("\n" + "="*60)
        print("Entropy-Reward Variance Visualization Complete")
        print("="*60 + "\n")

        return val_metrics

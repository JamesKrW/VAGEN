from .ray_trainer import *
import torch
import sys
import os
import re
from typing import Dict




class RayPlotTrainer(RayPPOTrainer):

    def _plot_per_group_metrics_correlation(self, per_group_metrics: Dict[str, Dict[str, float]]):
        """Plot correlation heatmap and pairwise scatter plots for per-group metrics.

        Args:
            per_group_metrics: Dict mapping metric_name to Dict[group_id, value]
        """
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        import pandas as pd
        from itertools import combinations

        output_dir = self.config.trainer.default_local_dir
        os.makedirs(output_dir, exist_ok=True)

        # Filter out metrics with no data
        valid_metrics = {k: v for k, v in per_group_metrics.items() if v and len(v) > 0}

        if len(valid_metrics) < 2:
            print(f"Warning: Need at least 2 valid metrics for correlation analysis. Got {len(valid_metrics)}.")
            return

        # Get common group_ids across all metrics (intersection)
        all_group_ids = [set(v.keys()) for v in valid_metrics.values()]
        common_group_ids = set.intersection(*all_group_ids) if all_group_ids else set()

        if len(common_group_ids) < 2:
            print(f"Warning: Need at least 2 common groups for correlation. Got {len(common_group_ids)}.")
            return

        # Sort group_ids for consistent ordering
        sorted_group_ids = sorted(common_group_ids)
        metric_names = list(valid_metrics.keys())

        # Build data matrix: rows=groups, cols=metrics
        data_matrix = np.zeros((len(sorted_group_ids), len(metric_names)))
        for j, metric_name in enumerate(metric_names):
            for i, gid in enumerate(sorted_group_ids):
                val = valid_metrics[metric_name].get(gid, np.nan)
                data_matrix[i, j] = val if np.isfinite(val) else np.nan

        # Create DataFrame for easier handling
        df = pd.DataFrame(data_matrix, columns=metric_names, index=sorted_group_ids)

        # Remove rows with any NaN/Inf
        df_clean = df.replace([np.inf, -np.inf], np.nan).dropna()

        if len(df_clean) < 2:
            print(f"Warning: After cleaning NaN/Inf, only {len(df_clean)} groups remain. Need at least 2.")
            return

        print(f"Computing correlations for {len(df_clean)} groups across {len(metric_names)} metrics.")

        # Compute correlation matrix
        corr_matrix = df_clean.corr(method='pearson')

        # Save correlation matrix to CSV
        csv_path = os.path.join(output_dir, "per_group_metrics_corr.csv")
        corr_matrix.to_csv(csv_path)
        print(f"Saved correlation matrix to {csv_path}")

        # Save per-group metrics data to CSV
        data_csv_path = os.path.join(output_dir, "per_group_metrics.csv")
        df_clean.to_csv(data_csv_path)
        print(f"Saved per-group metrics data to {data_csv_path}")

        # Plot correlation heatmap
        self._plot_correlation_heatmap(corr_matrix, metric_names, output_dir)

        # Plot pairwise scatter plots
        self._plot_pairwise_scatter(df_clean, metric_names, corr_matrix, output_dir)

        print(f"All plots saved to {output_dir}")

    def _sanitize_filename(self, name: str) -> str:
        """Sanitize metric name for use in filename."""
        # Replace / with _, remove other special characters
        sanitized = re.sub(r'[/\\]', '_', name)
        sanitized = re.sub(r'[^\w\-_]', '', sanitized)
        return sanitized

    def _plot_correlation_heatmap(self, corr_matrix, metric_names: list, output_dir: str):
        """Plot and save correlation heatmap."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        n_metrics = len(metric_names)
        fig_size = max(8, n_metrics * 1.2)

        fig, ax = plt.subplots(figsize=(fig_size, fig_size))

        # Create heatmap
        im = ax.imshow(corr_matrix.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')

        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.set_ylabel('Pearson r', rotation=-90, va="bottom")

        # Set ticks and labels
        ax.set_xticks(np.arange(n_metrics))
        ax.set_yticks(np.arange(n_metrics))
        ax.set_xticklabels(metric_names, rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(metric_names, fontsize=9)

        # Annotate cells with correlation values
        for i in range(n_metrics):
            for j in range(n_metrics):
                val = corr_matrix.values[i, j]
                if np.isfinite(val):
                    text_color = 'white' if abs(val) > 0.5 else 'black'
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                           color=text_color, fontsize=8)

        ax.set_title('Per-Group Metrics Correlation', fontsize=12, fontweight='bold')

        plt.tight_layout()

        heatmap_path = os.path.join(output_dir, "per_group_metrics_corr.png")
        plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved correlation heatmap to {heatmap_path}")

    def _plot_pairwise_scatter(self, df, metric_names: list, corr_matrix, output_dir: str):
        """Plot pairwise scatter plots for all metric pairs (including both directions)."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        n_groups = len(df)

        for i, metric_x in enumerate(metric_names):
            for j, metric_y in enumerate(metric_names):
                if i == j:
                    continue

                x_vals = df[metric_x].values
                y_vals = df[metric_y].values

                # Get correlation value
                r_val = corr_matrix.loc[metric_x, metric_y]

                # Create scatter plot
                fig, ax = plt.subplots(figsize=(8, 6))

                ax.scatter(x_vals, y_vals, alpha=0.6, edgecolors='black', linewidth=0.5)

                # Add trend line
                if np.isfinite(r_val) and len(x_vals) > 1:
                    z = np.polyfit(x_vals, y_vals, 1)
                    p = np.poly1d(z)
                    x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
                    ax.plot(x_line, p(x_line), 'r--', alpha=0.8, linewidth=2, label='Trend line')

                ax.set_xlabel(metric_x, fontsize=11)
                ax.set_ylabel(metric_y, fontsize=11)
                ax.set_title(f'{metric_x} vs {metric_y}', fontsize=12, fontweight='bold')

                # Add correlation annotation
                r_text = f'r = {r_val:.3f}' if np.isfinite(r_val) else 'r = N/A'
                ax.annotate(f'{r_text}\nn = {n_groups}',
                           xy=(0.05, 0.95), xycoords='axes fraction',
                           fontsize=10, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

                ax.grid(True, alpha=0.3)
                plt.tight_layout()

                # Save with sanitized filename
                x_safe = self._sanitize_filename(metric_x)
                y_safe = self._sanitize_filename(metric_y)
                scatter_path = os.path.join(output_dir, f"per_group_scatter__{x_safe}__vs__{y_safe}.png")
                plt.savefig(scatter_path, dpi=150, bbox_inches='tight')
                plt.close(fig)

        print(f"Saved {len(metric_names) * (len(metric_names) - 1)} pairwise scatter plots")
        
        
    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
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

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # add uid to batch
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                gen_batch = self._get_gen_batch(batch)

                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch_output = gen_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                )
                if not self.concat_multi_turn:
                    # we need to create group_idx, traj_idx for each traj in no-concat mode
                    num_traj_per_sample = self.config.actor_rollout_ref.rollout.n
                    self._assign_group_and_traj_idx(gen_batch_output, num_traj_per_sample)

                is_last_step = self.global_steps >= self.total_training_steps
                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, color="red"):
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch_output)
                        else:
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)

                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        if not self.concat_multi_turn:
                            raise NotImplementedError("REMAX advantage estimation is not supported in no-concat mode yet.")
                        if self.reward_fn is None:
                            raise ValueError("A reward_fn is required for REMAX advantage estimation.")

                        with marked_timer("gen_max", timing_raw, color="purple"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            if not self.async_rollout_mode:
                                gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)
                            else:
                                gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)
                            batch = batch.union(gen_baseline_output)
                            # compute reward model score on batch
                            rm_scores = None
                            if self.use_rm and "rm_scores" not in batch.batch.keys():
                                rm_scores = self.rm_wg.compute_rm_score(batch)
                                batch = batch.union(rm_scores)
                            reward_baseline_tensor, _ = compute_reward(batch, self.reward_fn)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            keys_to_pop = set(gen_baseline_output.batch.keys())
                            if rm_scores is not None:
                                keys_to_pop.update(rm_scores.batch.keys())
                            batch.pop(batch_keys=list(keys_to_pop))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del rm_scores, gen_baseline_batch, gen_baseline_output
                    # repeat to align with repeated responses in rollout
                    if self.concat_multi_turn:
                        batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                        batch = batch.union(gen_batch_output)
                    else:
                        # In no-concat mode, each trajectory has multiple prompt-response pairs.
                        # We need to re-generate batch to align with gen_batch_output.
                        batch = self._post_process_no_concat_batch(batch, gen_batch_output)

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)

                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    if self.config.trainer.balance_batch:
                        if not self.concat_multi_turn: # pad to divisor of dp_size
                            divisor_size = self.actor_rollout_wg.world_size
                            batch_size = len(batch.batch["attention_mask"])
                            batch, pad_size = pad_dataproto_to_divisor(batch, divisor_size)
                            print(f"Pad {pad_size} samples to make batch size {batch_size} divisible by {divisor_size} dp_workers")
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    with marked_timer("reward", timing_raw, color="yellow"):
                        # compute reward model score
                        if self.use_rm and "rm_scores" not in batch.batch.keys():
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(
                                data=batch, config=self.config, tokenizer=self.tokenizer
                            )
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                    # Operating Mode Selection:
                    # - Bypass mode: Sets old_log_probs = rollout_log_probs (2 policies: π_rollout, π_θ)
                    # - Decoupled mode: Recomputes old_log_probs as proximal anchor (3 policies: π_rollout, π_old, π_θ)
                    #   Note: π_old computed once per data batch, serves as stable reference during mini-batch updates
                    rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                    bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
                    if bypass_recomputing_logprobs:  # Use `rollout_log_probs`
                        from verl.trainer.ppo.rollout_corr_helper import apply_rollout_correction

                        apply_rollout_correction(
                            batch=batch,
                            rollout_corr_config=rollout_corr_config,
                            policy_loss_config=self.config.actor_rollout_ref.actor.policy_loss,
                        )
                    else:  # Recompute old_log_probs
                        with marked_timer("old_log_prob", timing_raw, color="blue"):
                            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                            entropys = old_log_prob.batch["entropys"]
                            response_masks = batch.batch["response_mask"]
                            loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                            entropy_agg = agg_loss(
                                loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode
                            )
                            old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                            metrics.update(old_log_prob_metrics)
                            # Keep entropys in batch for per-group metric computation
                            batch.batch["entropys"] = entropys
                            old_log_prob.batch.pop("entropys")
                            batch = batch.union(old_log_prob)
                            if "rollout_log_probs" in batch.batch.keys():
                                # TODO: we may want to add diff of probs too.
                                from verl.utils.debug.metrics import calculate_debug_metrics

                                metrics.update(calculate_debug_metrics(batch))

                    assert "old_log_probs" in batch.batch, f'"old_log_prob" not in {batch.batch.keys()=}'

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # Compute rollout correction: IS weights, rejection sampling, and metrics
                        # Only runs in decoupled mode (computes once per batch using stable π_old)
                        # In bypass mode, this is skipped - actor computes metrics from evolving π_θ vs π_rollout
                        if (
                            rollout_corr_config is not None
                            and "rollout_log_probs" in batch.batch
                            and not bypass_recomputing_logprobs  # Only in decoupled mode
                        ):
                            from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch

                            # Compute IS weights, apply rejection sampling, compute metrics
                            batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)
                            # IS and off-policy metrics already have rollout_corr/ prefix
                            metrics.update(is_metrics)

                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )  # GRPO adv normalization factor

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )

                    if self.config.algorithm.adv_estimator in ["no_concat_gae_last", "no_concat_gae_first"]:
                        batch.batch["value_mask"] = compute_value_mask(batch)

                    # compute custom metrics
                    with marked_timer("custom_metrics", timing_raw, color="magenta"):
                        custom_train_metrics = compute_custom_metrics(batch, prefix="custom_metrics/train")
                        metrics.update(custom_train_metrics)

                    # debug: log batch shapes for plot run
                    try:
                        b = batch.batch
                        nt = batch.non_tensor_batch
                        print(
                            f"[Plot Debug] batch_size={len(batch)} "
                            f"tensor_keys={list(b.keys())} non_tensor_keys={list(nt.keys())}"
                        )
                        for k, v in b.items():
                            if torch.is_tensor(v):
                                print(
                                    f"[Plot Debug][tensor] {k} shape={tuple(v.shape)} "
                                    f"dtype={v.dtype} device={v.device}"
                                )
                            else:
                                print(f"[Plot Debug][tensor] {k} type={type(v)}")
                        for k, v in nt.items():
                            if hasattr(v, "shape"):
                                print(f"[Plot Debug][non_tensor] {k} shape={v.shape} type={type(v)}")
                            else:
                                try:
                                    v_len = len(v)
                                except Exception:
                                    v_len = "n/a"
                                print(f"[Plot Debug][non_tensor] {k} len={v_len} type={type(v)}")
                    except Exception as e:
                        print(f"[Plot Debug] Failed to log batch shapes: {e}")

                    # stop here for plot
                    from .plot_metrics.registry import REGISTERED_METRICS
                    per_group_metrics = {}  # key: metric name, value: Dict[group_id, float]

                    for metric_name, metric_fn in REGISTERED_METRICS.items():
                        try:
                            per_group_metrics[metric_name] = metric_fn(
                                batch,
                                actor_wg=self.actor_rollout_wg,
                                tokenizer=self.tokenizer,
                                processor=self.processor,
                            )
                        except Exception as e:
                            print(f"Warning: Failed to compute metric '{metric_name}': {e}")

                    # plot correlation of per group metrics and save to self.config.trainer.default_local_dir
                    self._plot_per_group_metrics_correlation(per_group_metrics)
                    print("Plotting completed. Exiting now.")
                    sys.exit(0)

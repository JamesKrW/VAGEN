# Registry of all available per-group metrics
from .mutual_inforamtion import compute_group_mi_first_turn
from .plot_metrics import group_reward_variance, group_response_len_mean, group_entropy_mean
REGISTERED_METRICS = {
    "reward/variance": group_reward_variance,
    # "reward/mean": group_reward_mean,
    "actor/entropy": group_entropy_mean,
    "response/length": group_response_len_mean,
    # "advantage/mean": group_advantage_mean,
    # "actor/log_prob": group_old_log_prob_mean,
    "collapse/mi": compute_group_mi_first_turn,
}
"""Bi-level GAE implementation."""

import logging

import torch

from vagen.algorithms._common import AdvantageInputs, advantage_estimator, register_algorithm
from vagen.algorithms._common.packing import _pack

logger = logging.getLogger(__name__)


def _lump_rewards_at_turn_end(packed):
    """Sum each turn's token rewards onto that turn's final token."""
    boundary = packed.boundary() & packed.valid
    last_valid = packed.valid & ~torch.cat(
        [packed.valid[:, 1:], torch.zeros_like(packed.valid[:, :1])], dim=1
    )
    boundary = boundary | last_valid
    turn_of = boundary.long().cumsum(dim=1) - boundary.long()
    masked = torch.where(packed.valid, packed.seq_r, torch.zeros_like(packed.seq_r))
    totals = torch.zeros_like(masked).scatter_add_(1, turn_of, masked)
    return torch.where(
        boundary,
        totals.gather(1, turn_of),
        torch.zeros_like(masked),
    )


@advantage_estimator("bi_level_gae", needs_critic=True, turn_lumped_reward=True)
def compute_bi_level_gae(inputs: AdvantageInputs):
    """The **published** VAGEN Bi-Level GAE, reproduced as released rather than as fixed.

    Ported from the released implementation
    (``compute_bi_level_gae_advantage_return``, commit 4076507 in this repo's own
    history), not from the paper's prose -- §4.2 and Algorithm 2 disagree about whether
    the turn advantage is composed with the last token's delta or replaces it, and the
    code is what produced the published numbers. It replaces.

    Two nested passes:

    1. **Turn level.** GAE over the turn-final tokens only, anchored at each turn's
       *last* model-output token -- ``V(tau_{<=a_t})``, the value of a prefix that already
       contains the action -- using ``high_level_gamma``::

           delta_t = r_t + high_level_gamma * V(eos_{t+1}) - V(eos_t)
           A_t     = delta_t + high_level_gamma * lam * A_{t+1}

    2. **Token level.** The turn's *return* ``A_t + V(eos_t)`` is written back as the
       reward at that token, then token-level GAE runs backward with ``gamma``, and at
       every turn-final token both the bootstrap and the accumulator are zeroed::

           at a turn end:  nextvalue = 0, lastgaelam = 0
                           delta     = (A_t + V) - V = A_t     ->  advantage = A_t
           inside a turn:  the ordinary token recursion, seeded from A_t

    So the turn's last token receives exactly ``A_t`` -- its own delta is discarded, not
    added -- and each turn's inner chain is independent of every other's.

    ★ The afterstate anchor is real but **self-cancelling here**, and this is the most
    important thing to know about the estimator. Pass 1 telescopes to ``A_t = G_t - V(e_t)``
    and pass 2 inside the turn telescopes to ``V(e_t) - V(j)``; the ``V(e_t)`` terms cancel
    and every token receives ``G_t - V(j)``, the correct advantage. Measured on a tabular
    multi-turn MDP with an exact critic and an exact policy gradient:

    * ``high_level_gamma == gamma`` and ``lam == 1``: **identical to token_level_gae**, to
      4e-16. Not approximately -- the same numbers.
    * the outer chain *in isolation* (i.e. lifted out to be a turn-level estimator) is
      catastrophic: relative error 0.949, and **exactly zero gradient** on every
      non-final token of a turn, because ``G_t - E[G_t | s_t, a_t^{1..L-1}]`` is
      conditionally zero-mean given all but the last token. Only the final token -- usually
      a near-deterministic EOS -- keeps any signal.
    * "fixing" the anchor to the turn's first token *without* moving the write-back
      position makes it **worse than the bug**: relative error 0.156 and a gradient 1.92x
      too large, because the value is then subtracted twice. Anchor and deposit position
      must move together, which is what ``turn_level_gae`` does and why the corrected
      first-anchor estimator already exists under that name.

    So do not "correct" the anchor here. The defect is in how the outer chain is
    *described*, not in what the composite computes.

    ★ Therefore ``high_level_gamma`` is the only thing separating this from
    ``token_level_gae``, and at ``high_level_gamma == gamma`` there is nothing left. The
    released config ships ``gamma 1.0 / lam 1.0 / high_level_gamma 0.99`` and the released
    sokoban script uses ``0.9``; measured divergence from token-level GAE on the same
    fixture is 4e-16 at 1.0, 1.8e-2 at 0.99 and 2.1e-1 at 0.9. Running this at
    ``high_level_gamma = gamma`` reproduces nothing.

    The zeroed accumulator is separately why intra-turn reward cannot propagate across a
    turn boundary, and the overwrite is why the token carrying the outcome reward learns
    from the turn advantage rather than its own delta.

    ★ Pass 1 reads a turn's reward only at the turn's last token, so this estimator lumps
    it there itself, via ``_lump_rewards_at_turn_end``. It does not ask the
    environment to pay it that way: where a reward is earned is the environment's business,
    what shape this recursion needs is this function's, and a reward left mid-turn would
    otherwise be invisible to the outer chain and credited by the inner one alone.

    ★ Two clocks, two gammas -- ``gamma`` for tokens, ``+algorithm.high_level_gamma`` for
    turns. Two explicit gammas is what makes this well-defined away from 1.0, and why it
    carries no ``undiscounted`` guard: the released code takes the two separately
    and the paper's Table 23 sets the token one to 1.0. Left unset, ``high_level_gamma``
    follows ``gamma``.
    """
    gamma = float(inputs.config.gamma)
    lam = float(inputs.config.lam)
    high_gamma = float(inputs.param("high_level_gamma", gamma))

    if high_gamma == gamma and lam == 1.0:
        # Not an error -- it is what the released code does with those numbers -- but a run
        # started this way is token_level_gae wearing the baseline's name, and the curves
        # give no hint of it.
        logger.warning(
            "bi_level_gae: high_level_gamma == gamma == %s and lam == 1.0, at which "
            "the two passes telescope and this estimator is IDENTICAL to token_level_gae "
            "(verified to 4e-16). The outer chain's V(afterstate) cancels against the "
            "inner chain exactly. Reproducing the published setting means "
            "high_level_gamma < gamma -- the released config uses 0.99 and the released "
            "sokoban script 0.9.", gamma,
        )

    with torch.no_grad():
        packed = _pack(inputs)
        valid, seq_v = packed.valid, packed.seq_v
        # Pass 1 steps only over turn-final tokens, so fold each turn's reward onto its own
        # boundary first. The environment pays per span; this is where that becomes the
        # one-slot-per-turn shape the outer chain requires.
        seq_r = _lump_rewards_at_turn_end(packed)
        # The released code's `reward_mask`: one position per turn, at its last token.
        turn_end = packed.boundary() & valid
        n_traj, max_len = valid.shape
        zeros = torch.zeros(n_traj, dtype=seq_v.dtype, device=seq_v.device)

        # -- pass 1: turn level, stepping only over turn-final tokens.
        turn_adv = torch.zeros_like(seq_v)
        nextvalue, lastgaelam = zeros.clone(), zeros.clone()
        for t in reversed(range(max_len)):
            live = turn_end[:, t]
            delta = seq_r[:, t] + high_gamma * nextvalue - seq_v[:, t]
            lastgaelam = torch.where(live, delta + high_gamma * lam * lastgaelam, lastgaelam)
            turn_adv[:, t] = torch.where(live, lastgaelam, torch.zeros_like(lastgaelam))
            nextvalue = torch.where(live, seq_v[:, t], nextvalue)

        # The turn's return becomes the reward at its last token; every other position
        # keeps whatever token-level reward it had.
        upd_r = torch.where(turn_end, turn_adv + seq_v, seq_r)

        # -- pass 2: token level, restarting at every turn end.
        seq_adv = torch.zeros_like(seq_v)
        nextvalues, lastgaelam = zeros.clone(), zeros.clone()
        for t in reversed(range(max_len)):
            live = valid[:, t]
            ends = turn_end[:, t]
            nv = torch.where(ends, torch.zeros_like(nextvalues), nextvalues)
            lg = torch.where(ends, torch.zeros_like(lastgaelam), lastgaelam)
            delta = upd_r[:, t] + gamma * nv - seq_v[:, t]
            lastgaelam = torch.where(live, delta + gamma * lam * lg, lastgaelam)
            seq_adv[:, t] = torch.where(live, lastgaelam, torch.zeros_like(lastgaelam))
            nextvalues = torch.where(live, seq_v[:, t], nextvalues)

        return packed.emit(
            advantages=packed.scatter(seq_adv),
            returns=packed.scatter(seq_adv + packed.seq_v),
        )


SPEC = register_algorithm("bi_level_gae", compute_bi_level_gae)

__all__ = ["SPEC", "compute_bi_level_gae"]

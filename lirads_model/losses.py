"""
Ordinal-aware loss(es) for the LR-1..LR-5 ordinal head.

Plain nn.CrossEntropyLoss treats the 5 ordinal ranks as unrelated
categories, so a prediction of LR-1 on a true LR-5 case is penalized
exactly as much as a prediction of LR-4 on that same case. SORDLoss fixes
that by softening the one-hot target into a distribution peaked at the
true rank and decaying with squared rank distance, while keeping the same
weighted-cross-entropy machinery (and the same call signature) as the
nn.CrossEntropyLoss(weight=...) it replaces -- see train.py's
build_ordinal_criterion().
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SORDLoss(nn.Module):
    """Soft Ordinal (SORD) loss (Diaz & Marathe, "Soft Labels for Ordinal
    Regression", CVPR 2019), applied to the num_classes ordinal ranks
    0..num_classes-1.

    For a true rank y, the target distribution over ranks i is
        p_i = softmax_i(-(i - y)**2)
    i.e. a discrete, rank-distance-weighted soft label instead of a
    one-hot vector -- adjacent ranks get partial credit, ranks further
    away get exponentially less. The loss is the cross entropy between
    the model's predicted softmax and this soft target,
    -sum_i p_i * log_softmax(logits)_i, per-sample-weighted by `weight`
    (indexed by the true class, exactly like nn.CrossEntropyLoss(weight=...))
    and averaged the same way (sum of per-sample losses / sum of weights).
    """

    def __init__(self, num_classes: int, weight: Optional[torch.Tensor] = None):
        super().__init__()
        ranks = torch.arange(num_classes, dtype=torch.float32)
        metric = (ranks.unsqueeze(1) - ranks.unsqueeze(0)) ** 2  # (num_classes, num_classes)
        # row y = target distribution over ranks i, for true rank y
        soft_targets = torch.softmax(-metric, dim=1)
        self.register_buffer("soft_targets", soft_targets)
        self.register_buffer("weight", weight if weight is not None else torch.ones(num_classes))

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_probs = torch.log_softmax(logits, dim=1)
        targets = self.soft_targets[target]  # (B, num_classes)
        per_sample = -(targets * log_probs).sum(dim=1)  # (B,)
        sample_weight = self.weight[target]  # (B,)
        return (per_sample * sample_weight).sum() / sample_weight.sum().clamp_min(1e-12)


def corn_loss(
    logits: torch.Tensor, target: torch.Tensor, num_classes: int, weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """CORN (Conditional Ordinal Regression for Neural networks; Shi, Cao &
    Raschka, 2021) loss. `logits` has shape (N, num_classes-1): task k's
    logit is trained -- via plain binary cross entropy -- only on the subset
    of examples whose true rank is >= k, to predict whether that rank is > k.
    Conditioning each task's training set on the previous tasks' outcome
    (rather than training every task on the full batch, as CORAL does) is
    what gives CORN rank-monotonic predictions without needing tied task
    weights, so its predicted ranks are decoded via corn_label_from_logits
    below rather than a plain argmax.

    `weight`, if given, is a (num_classes,) tensor indexed by each example's
    *true* ordinal class (matching nn.CrossEntropyLoss(weight=...)),
    applied to that example's contribution in every task it participates
    in; the total is normalized by the sum of weights actually used, same
    as SORDLoss.
    """
    if weight is None:
        weight = logits.new_ones(num_classes)

    total_loss = logits.new_zeros(())
    total_weight = logits.new_zeros(())
    for k in range(num_classes - 1):
        mask = target > (k - 1)  # true rank >= k
        if not mask.any():
            continue
        task_logits = logits[mask, k]
        task_target = (target[mask] > k).to(task_logits.dtype)
        task_weight = weight[target[mask]]
        per_example = F.binary_cross_entropy_with_logits(task_logits, task_target, reduction="none")
        total_loss = total_loss + (per_example * task_weight).sum()
        total_weight = total_weight + task_weight.sum()
    return total_loss / total_weight.clamp_min(1e-12)


def corn_probas_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Converts CORN's (N, num_classes-1) conditional logits into a proper
    per-class probability distribution (N, num_classes), for use anywhere a
    softmax-style distribution over ranks is needed (see soft_qwk below).
    cum[:, k] = P(rank > k) = prod_{i<=k} sigmoid(logit_i) -- the chained
    conditional probabilities; each class's probability mass is the
    consecutive difference of that cumulative distribution."""
    cum = torch.cumprod(torch.sigmoid(logits), dim=1)  # (N, num_classes-1): P(rank > k)
    first = 1.0 - cum[:, :1]  # P(rank == 0)
    middle = cum[:, :-1] - cum[:, 1:]  # P(rank == k), k = 1..num_classes-2
    last = cum[:, -1:]  # P(rank == num_classes-1)
    return torch.cat([first, middle, last], dim=1)


def corn_label_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Rank-consistent hard decode for a CORN ordinal head (Shi et al.,
    2021): threshold k is considered "exceeded" when the cumulative
    P(rank > k) > 0.5, and the predicted rank is how many thresholds were
    exceeded. Used at inference (see model.decode_prediction) in place of
    the plain argmax used for a softmax ordinal head."""
    cum = torch.cumprod(torch.sigmoid(logits), dim=1)
    return (cum > 0.5).sum(dim=1)


def soft_qwk(probas: torch.Tensor, target: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Differentiable approximation of Quadratic Weighted Kappa (de la Torre
    et al., "Weighted kappa loss function for multi-class classification of
    ordinal data in deep learning", 2018): the observed and expected
    disagreement matrices that QWK is built from are computed using
    predicted class probabilities in place of hard argmax predictions, so
    gradients flow through `probas`. Returns a scalar that matches
    sklearn's cohen_kappa_score(weights="quadratic") in the limit of
    one-hot `probas`, roughly in [-1, 1] where 1 is perfect agreement.
    """
    device = probas.device
    ranks = torch.arange(num_classes, dtype=probas.dtype, device=device)
    weight = (ranks.unsqueeze(1) - ranks.unsqueeze(0)) ** 2
    weight = weight / (num_classes - 1) ** 2

    target_onehot = torch.zeros(probas.shape[0], num_classes, dtype=probas.dtype, device=device)
    target_onehot.scatter_(1, target.unsqueeze(1), 1.0)

    observed = target_onehot.t() @ probas  # (K, K): O_ij = sum_n 1[y_n=i] * p_n(j)

    hist_true = target_onehot.sum(dim=0)  # (K,)
    hist_pred = probas.sum(dim=0)  # (K,)
    n = probas.shape[0]
    expected = torch.outer(hist_true, hist_pred) / n

    numerator = (weight * observed).sum()
    denominator = (weight * expected).sum().clamp_min(1e-12)
    return 1.0 - numerator / denominator


class CornSoftQWKLoss(nn.Module):
    """L = CORN_loss + lambda_qwk * (1 - soft_QWK), for an ordinal head
    trained with the CORN (rank-consistent conditional) parameterization
    (see model.LiRadsNet's ordinal_head_type="corn", which sizes ord_head
    to num_classes-1 outputs instead of the plain softmax head's
    num_classes). corn_loss alone only supervises each of the num_classes-1
    pairwise-threshold decisions independently; the soft_qwk term adds a
    direct, differentiable pressure toward the eventual evaluation metric
    (quadratic weighted kappa) on top of that, using the full predicted
    class distribution reconstructed by corn_probas_from_logits.
    """

    def __init__(self, num_classes: int, lambda_qwk: float = 1.0, weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.num_classes = num_classes
        self.lambda_qwk = lambda_qwk
        self.register_buffer("weight", weight if weight is not None else torch.ones(num_classes))

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        c_loss = corn_loss(logits, target, self.num_classes, weight=self.weight)
        probas = corn_probas_from_logits(logits)
        kappa = soft_qwk(probas, target, self.num_classes)
        return c_loss + self.lambda_qwk * (1.0 - kappa)

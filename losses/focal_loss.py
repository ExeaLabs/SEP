"""Focal Loss for Class-Imbalanced Extinction Risk Classification.

Implements the focal loss from *"Focal Loss for Dense Object Detection"*
(Lin et al., 2017) adapted for multi-class classification. Down-weights
easy examples so the model focuses on hard-to-classify species whose
extinction risk is ambiguous.

This is critical for IUCN data where the majority of assessed species
are Least Concern, creating severe class imbalance.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """Multi-class focal loss with optional label smoothing.

    Focal loss applies a modulating factor ``(1 - p_t)^gamma`` to the
    standard cross-entropy loss, reducing the relative loss for
    well-classified examples and focusing on hard, mis-classified ones.

    .. math::

        FL(p_t) = -\\alpha_t \\, (1 - p_t)^{\\gamma} \\, \\log(p_t)

    Args:
        gamma: Focusing parameter.  ``gamma=0`` recovers standard CE.
            Higher values increase the focus on hard examples.
        alpha: Per-class balancing weights tensor of shape ``(C,)``.
            If ``None``, all classes are weighted equally.
        reduction: Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``.
        label_smoothing: Label smoothing factor in ``[0, 1)``.  When > 0,
            the one-hot target distribution is mixed with a uniform
            distribution over classes.

    Example::

        criterion = FocalLoss(gamma=2.0, alpha=torch.tensor([0.5, 1.0, 1.5, 2.0, 2.5]))
        logits = torch.randn(8, 5)
        targets = torch.randint(0, 5, (8,))
        loss = criterion(logits, targets)
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        if gamma < 0:
            raise ValueError(f"gamma must be non-negative, got {gamma}")
        if reduction not in ("none", "mean", "sum"):
            raise ValueError(
                f"reduction must be 'none', 'mean', or 'sum', got '{reduction}'"
            )
        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError(
                f"label_smoothing must be in [0, 1), got {label_smoothing}"
            )

        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing

        # Register alpha as a buffer so it moves with the model to GPU/CPU
        if alpha is not None:
            if not isinstance(alpha, torch.Tensor):
                alpha = torch.tensor(alpha, dtype=torch.float32)
            self.register_buffer("alpha", alpha.float())
        else:
            self.alpha: Optional[torch.Tensor] = None

        logger.debug(
            "FocalLoss created: gamma=%.2f, label_smoothing=%.3f, "
            "reduction=%s, alpha=%s",
            gamma,
            label_smoothing,
            reduction,
            alpha,
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Args:
            logits: Raw (unnormalised) predictions of shape ``(B, C)``
                where ``C`` is the number of classes.
            targets: Ground-truth class indices of shape ``(B,)`` with
                values in ``[0, C)``.

        Returns:
            Focal loss scalar (if reduction is ``'mean'`` or ``'sum'``)
            or per-sample losses of shape ``(B,)`` (if ``'none'``).

        Raises:
            ValueError: If logits and targets have incompatible shapes.
        """
        if logits.ndim != 2:
            raise ValueError(f"Expected 2-D logits (B, C), got {logits.ndim}-D")
        if targets.ndim != 1:
            raise ValueError(f"Expected 1-D targets (B,), got {targets.ndim}-D")
        if logits.size(0) != targets.size(0):
            raise ValueError(
                f"Batch size mismatch: logits={logits.size(0)}, "
                f"targets={targets.size(0)}"
            )

        # Compute in float32 regardless of autocast context to avoid
        # fp16 underflow in log_softmax / (1-p)^gamma blowing up to NaN.
        logits = logits.float()

        num_classes = logits.size(1)

        # Compute softmax probabilities
        probs = F.softmax(logits, dim=-1)  # (B, C)
        probs = probs.clamp(min=1e-6, max=1.0 - 1e-6)

        # Apply label smoothing to targets
        if self.label_smoothing > 0.0:
            # Create smoothed one-hot targets
            with torch.no_grad():
                smooth_targets = torch.full_like(
                    probs, self.label_smoothing / (num_classes - 1)
                )
                smooth_targets.scatter_(
                    1,
                    targets.unsqueeze(1),
                    1.0 - self.label_smoothing,
                )
        else:
            with torch.no_grad():
                smooth_targets = torch.zeros_like(probs)
                smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0)

        # Log probabilities, computed from the same clamped probs used
        # above so focal_weight and log_probs stay numerically consistent.
        log_probs = torch.log(probs)  # (B, C)

        # Focal modulating factor: (1 - p_t)^gamma
        # For multi-class, compute per-class focal weight
        focal_weight = (1.0 - probs).pow(self.gamma)  # (B, C)

        # Per-class focal cross-entropy
        loss = -focal_weight * smooth_targets * log_probs  # (B, C)

        # Apply alpha weighting
        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)
            # Broadcast alpha across batch: (C,) -> (1, C)
            loss = loss * alpha.unsqueeze(0)

        # Sum over classes -> per-sample loss
        loss = loss.sum(dim=-1)  # (B,)

        # Reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


def test_focal_loss() -> None:
    """Unit test verifying FocalLoss output shape and gradient flow.

    Tests:
        1. Output is a scalar with ``reduction='mean'``.
        2. Output shape is ``(B,)`` with ``reduction='none'``.
        3. Gradients flow back to logits.
        4. Loss is non-negative.
        5. ``gamma=0`` with no alpha/smoothing approximates standard CE.
        6. Label smoothing produces a valid loss.
        7. Per-class alpha weighting works correctly.
    """
    torch.manual_seed(42)
    batch_size, num_classes = 16, 5
    logits = torch.randn(batch_size, num_classes, requires_grad=True)
    targets = torch.randint(0, num_classes, (batch_size,))
    alpha = torch.tensor([0.5, 1.0, 1.5, 2.0, 2.5])

    criterion = FocalLoss(gamma=2.0, alpha=alpha, reduction="mean")
    loss = criterion(logits, targets)
    assert loss.ndim == 0, f"Expected scalar, got shape {loss.shape}"
    print(f"✓ Test 1 passed: scalar loss = {loss.item():.6f}")

    criterion_none = FocalLoss(gamma=2.0, alpha=alpha, reduction="none")
    loss_none = criterion_none(logits, targets)
    assert loss_none.shape == (batch_size,), (
        f"Expected shape ({batch_size},), got {loss_none.shape}"
    )
    print(f"✓ Test 2 passed: per-sample loss shape = {loss_none.shape}")

    loss.backward()
    assert logits.grad is not None, "Gradients did not flow to logits"
    assert not torch.all(logits.grad == 0), "All gradients are zero"
    print(f"✓ Test 3 passed: gradients flow (grad norm = {logits.grad.norm():.6f})")

    assert (loss_none >= 0).all(), "Focal loss should be non-negative"
    print("✓ Test 4 passed: all per-sample losses are non-negative")

    logits_ce = torch.randn(batch_size, num_classes)
    targets_ce = torch.randint(0, num_classes, (batch_size,))
    focal_g0 = FocalLoss(gamma=0.0, alpha=None, reduction="mean")
    ce_ref = nn.CrossEntropyLoss(reduction="mean")
    loss_focal_g0 = focal_g0(logits_ce, targets_ce)
    loss_ce_ref = ce_ref(logits_ce, targets_ce)
    diff = (loss_focal_g0 - loss_ce_ref).abs().item()
    assert diff < 1e-5, f"gamma=0 should match CE, diff={diff}"
    print(f"✓ Test 5 passed: gamma=0 matches CE (diff={diff:.2e})")

    criterion_smooth = FocalLoss(
        gamma=2.0, alpha=alpha, reduction="mean", label_smoothing=0.1
    )
    loss_smooth = criterion_smooth(logits_ce, targets_ce)
    assert loss_smooth.ndim == 0, "Label-smoothed loss should be scalar"
    assert loss_smooth.item() >= 0, "Label-smoothed loss should be non-negative"
    print(f"✓ Test 6 passed: label smoothing loss = {loss_smooth.item():.6f}")

    criterion_no_alpha = FocalLoss(gamma=2.0, alpha=None, reduction="none")
    criterion_with_alpha = FocalLoss(gamma=2.0, alpha=alpha, reduction="none")
    loss_no_a = criterion_no_alpha(logits_ce, targets_ce)
    loss_with_a = criterion_with_alpha(logits_ce, targets_ce)
    assert not torch.allclose(loss_no_a, loss_with_a), (
        "Alpha weighting should change the loss"
    )
    print("✓ Test 7 passed: alpha weighting modifies loss values")

    print("\n✅ All FocalLoss tests passed!")


if __name__ == "__main__":
    test_focal_loss()

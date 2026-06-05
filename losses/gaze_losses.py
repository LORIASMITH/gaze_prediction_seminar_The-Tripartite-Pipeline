"""
Loss functions for the cross-domain gaze estimation pipeline.

angular_loss   : L1-style loss on the angular difference between two gaze directions.
gfal_loss      : GFAL frontalization consistency loss.
                 Given the canonical gaze prediction g₀ and the head-pose rotation R_h:
                   L_canon  = angular(g₀_pred_3d,  g₀_gt_3d)
                   L_rotate = angular(R_h @ g₀_pred_3d,  g_gt_3d)
                   L_gfal   = L_canon + L_rotate
"""

import torch
import torch.nn as nn


# ──────────────────────────────────────────────────────────────────────────────
# Coordinate helpers
# ──────────────────────────────────────────────────────────────────────────────

def gazeto3d(gaze2d: torch.Tensor) -> torch.Tensor:
    """
    Convert (yaw, pitch) to a unit 3-D gaze vector.
    Convention used by GazeHub:  g = [-cos(p)sin(y),  -sin(p),  -cos(p)cos(y)]

    gaze2d : (..., 2)  — [yaw, pitch] in radians
    returns : (..., 3)
    """
    yaw   = gaze2d[..., 0]
    pitch = gaze2d[..., 1]
    x = -torch.cos(pitch) * torch.sin(yaw)
    y = -torch.sin(pitch)
    z = -torch.cos(pitch) * torch.cos(yaw)
    return torch.stack([x, y, z], dim=-1)


def head_pose_rotation(head_pose: torch.Tensor) -> torch.Tensor:
    """
    Build 3×3 rotation matrix from head pose (yaw, pitch).
    R = R_y(yaw) @ R_x(pitch)

    head_pose : (B, 2) — [yaw, pitch] in radians
    returns   : (B, 3, 3)
    """
    yaw, pitch = head_pose[:, 0], head_pose[:, 1]
    cy, sy = torch.cos(yaw),   torch.sin(yaw)
    cp, sp = torch.cos(pitch), torch.sin(pitch)
    B = head_pose.shape[0]
    z = torch.zeros(B, device=head_pose.device)
    o = torch.ones(B,  device=head_pose.device)

    # R_y(yaw)
    Ry = torch.stack([
        torch.stack([ cy,  z, sy], dim=1),
        torch.stack([  z,  o,  z], dim=1),
        torch.stack([-sy,  z, cy], dim=1),
    ], dim=1)                            # (B, 3, 3)

    # R_x(pitch)
    Rx = torch.stack([
        torch.stack([ o,   z,   z], dim=1),
        torch.stack([ z,  cp, -sp], dim=1),
        torch.stack([ z,  sp,  cp], dim=1),
    ], dim=1)                            # (B, 3, 3)

    return torch.bmm(Ry, Rx)            # (B, 3, 3)


# ──────────────────────────────────────────────────────────────────────────────
# Loss functions
# ──────────────────────────────────────────────────────────────────────────────

def _angular(v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
    """
    Mean angular error (radians) between unit-normalised vector batches.
    v1, v2: (B, 3)
    """
    dot = (v1 * v2).sum(dim=-1)
    n1  = v1.norm(dim=-1).clamp(min=1e-7)
    n2  = v2.norm(dim=-1).clamp(min=1e-7)
    cos = (dot / (n1 * n2)).clamp(-1 + 1e-7, 1 - 1e-7)
    return torch.acos(cos).mean()


def angular_loss(pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    """
    Angular loss between two (B,2) gaze predictions in [yaw, pitch] format.
    Returns mean angular error in radians.
    """
    return _angular(gazeto3d(pred), gazeto3d(label))


def gfal_loss(
    g0_pred: torch.Tensor,
    gaze_gt: torch.Tensor,
    head_pose: torch.Tensor,
    lambda_canon: float = 0.5,
    lambda_rotate: float = 0.5,
) -> torch.Tensor:
    """
    GFAL frontalization consistency loss.

    g0_pred   : (B,2) — predicted canonical gaze [yaw₀, pitch₀]
    gaze_gt   : (B,2) — ground-truth image-frame gaze [yaw, pitch]
    head_pose : (B,2) — head pose [yaw_h, pitch_h]

    1. Canonical ground-truth:   g₀_gt_3d = R_h^T @ g_gt_3d
    2. L_canon  = angular(g₀_pred_3d, g₀_gt_3d)
    3. Rotated prediction:       g_rot_3d = R_h @ g₀_pred_3d
    4. L_rotate = angular(g_rot_3d, g_gt_3d)

    Total = lambda_canon * L_canon + lambda_rotate * L_rotate
    """
    R   = head_pose_rotation(head_pose)             # (B, 3, 3)
    g_gt_3d   = gazeto3d(gaze_gt)                  # (B, 3)
    g0_pred_3d = gazeto3d(g0_pred)                 # (B, 3)

    # Ground-truth canonical gaze
    g0_gt_3d = torch.bmm(
        R.transpose(1, 2), g_gt_3d.unsqueeze(-1)
    ).squeeze(-1)                                   # (B, 3)

    # Canonical supervision
    L_canon = _angular(g0_pred_3d, g0_gt_3d)

    # Rotation consistency: rotate predicted canonical → image frame → compare
    g_rot_3d = torch.bmm(R, g0_pred_3d.unsqueeze(-1)).squeeze(-1)  # (B, 3)
    L_rotate = _angular(g_rot_3d, g_gt_3d)

    return lambda_canon * L_canon + lambda_rotate * L_rotate


def compute_loss(
    gaze_pred: torch.Tensor,
    gaze_gt: torch.Tensor,
    g0_pred: torch.Tensor,
    head_pose: torch.Tensor,
    stage: int,
    lambda_gfal: float = 0.5,
) -> tuple:
    """
    Unified loss dispatcher.

    Returns (total_loss, loss_dict) where loss_dict has float entries for logging.
    """
    main = angular_loss(gaze_pred, gaze_gt)
    losses = {'gaze': main.item()}
    total  = main

    if stage >= 2 and g0_pred is not None:
        gf = gfal_loss(g0_pred, gaze_gt, head_pose)
        losses['gfal'] = gf.item()
        total = total + lambda_gfal * gf

    losses['total'] = total.item()
    return total, losses

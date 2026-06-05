"""
Evaluation metrics for gaze estimation.
"""

import math
import torch
import numpy as np


def gazeto3d_torch(gaze2d: torch.Tensor) -> torch.Tensor:
    """(B,2)[yaw,pitch] → (B,3) unit gaze vector."""
    yaw, pitch = gaze2d[..., 0], gaze2d[..., 1]
    x = -torch.cos(pitch) * torch.sin(yaw)
    y = -torch.sin(pitch)
    z = -torch.cos(pitch) * torch.cos(yaw)
    return torch.stack([x, y, z], dim=-1)


def gazeto3d_np(gaze2d: np.ndarray) -> np.ndarray:
    """(...,2)[yaw,pitch] → (...,3) unit gaze vector (NumPy)."""
    yaw, pitch = gaze2d[..., 0], gaze2d[..., 1]
    x = -np.cos(pitch) * np.sin(yaw)
    y = -np.sin(pitch)
    z = -np.cos(pitch) * np.cos(yaw)
    return np.stack([x, y, z], axis=-1)


def angular_error_deg(pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    """
    Angular error in **degrees** between (B,2) gaze predictions.
    Returns a (B,) tensor.
    """
    p3 = gazeto3d_torch(pred)
    l3 = gazeto3d_torch(label)
    dot = (p3 * l3).sum(dim=-1)
    denom = (p3.norm(dim=-1) * l3.norm(dim=-1)).clamp(min=1e-7)
    cos = (dot / denom).clamp(-1 + 1e-7, 1 - 1e-7)
    return torch.acos(cos) * (180.0 / math.pi)


def mean_angular_error_deg(pred: torch.Tensor, label: torch.Tensor) -> float:
    """Mean angular error in degrees (scalar Python float)."""
    return angular_error_deg(pred, label).mean().item()


def angular_error_np(pred: np.ndarray, label: np.ndarray) -> np.ndarray:
    """Angular error in degrees for NumPy arrays (B,2) or (B,3)."""
    if pred.shape[-1] == 2:
        pred  = gazeto3d_np(pred)
        label = gazeto3d_np(label)
    dot   = (pred * label).sum(axis=-1)
    denom = (np.linalg.norm(pred, axis=-1) * np.linalg.norm(label, axis=-1))
    cos   = np.clip(dot / np.clip(denom, 1e-7, None), -1 + 1e-7, 1 - 1e-7)
    return np.degrees(np.arccos(cos))

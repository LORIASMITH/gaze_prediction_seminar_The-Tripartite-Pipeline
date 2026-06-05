"""
UnifiedGazeNet  —  ablation-ready gaze estimation model.

Stage 0 : Baseline   — ResNet18 → GAP → FC → [yaw, pitch]
Stage 1 : +FSCI      — ResNet18 → spatial features → 3 causal tokens
                        (X=head-pose, W=gaze, Z=confound) via transformer
                        decoder → CausalIntervention (EMA-based Ez)
Stage 2 : +GFAL      — Stage 1 + frontalization head on W-token:
                        predict canonical gaze g₀, rotate by head pose → g
Stage 3 : +AGG       — post-training ISOMap + GeodesicProjection calibration
                        (calibrated externally via utils/gpm_utils.py)

The forward signature is the same for all stages:
    gaze_pred, gaze_feat, g0_pred = model(img, head_pose, mode)
  • gaze_pred : (B,2)   — main [yaw,pitch] prediction (degrees → same unit as label)
  • gaze_feat : (B,512) — pre-FC feature used by AGG
  • g0_pred   : (B,2)   — canonical gaze (only for stage ≥ 2; else None)
"""

import math
import sys
import os

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

# ============================================================
# ResNet-18 backbone  (verbatim from AGG / FSCI repos)
# ============================================================

def _conv3x3(in_c, out_c, stride=1):
    return nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False)


class _BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = _conv3x3(inplanes, planes, stride)
        self.bn1   = nn.BatchNorm2d(planes)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = _conv3x3(planes, planes)
        self.bn2   = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample:
            identity = self.downsample(x)
        return self.relu(out + identity)


class _ResNet18(nn.Module):
    """ResNet-18 without the original classifier; outputs spatial feature map."""

    def __init__(self):
        super().__init__()
        self.inplanes = 64
        self.conv1   = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1     = nn.BatchNorm2d(64)
        self.relu    = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1  = self._make_layer(64,  2)
        self.layer2  = self._make_layer(128, 2, stride=2)
        self.layer3  = self._make_layer(256, 2, stride=2)
        self.layer4  = self._make_layer(512, 2, stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        layers = [_BasicBlock(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(_BasicBlock(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x                                  # (B, 512, 7, 7) for 224×224 input


_RESNET18_URL = 'https://download.pytorch.org/models/resnet18-5c106cde.pth'

def _build_resnet18(pretrained: bool) -> _ResNet18:
    m = _ResNet18()
    if pretrained:
        state = model_zoo.load_url(_RESNET18_URL)
        # Drop the original FC keys; load everything else
        state = {k: v for k, v in state.items()
                 if not k.startswith(('fc',))}
        m.load_state_dict(state, strict=False)
    return m


# ============================================================
# FSCI components
# ============================================================

class _SinPos2D(nn.Module):
    """Sinusoidal 2-D positional encoding (appended to flattened spatial features)."""

    def __init__(self, d_model: int, h: int = 7, w: int = 7):
        super().__init__()
        assert d_model % 4 == 0, 'd_model must be divisible by 4'
        half = d_model // 2
        pos_h = torch.arange(h).float().unsqueeze(1)   # (H, 1)
        pos_w = torch.arange(w).float().unsqueeze(1)   # (W, 1)
        div   = torch.exp(torch.arange(0, half, 2).float() * (-math.log(10000) / half))
        pe_h  = torch.zeros(h, half)
        pe_h[:, 0::2] = torch.sin(pos_h * div)
        pe_h[:, 1::2] = torch.cos(pos_h * div)
        pe_w  = torch.zeros(w, half)
        pe_w[:, 0::2] = torch.sin(pos_w * div)
        pe_w[:, 1::2] = torch.cos(pos_w * div)
        # Build (HW, d_model): row-embed replicated W times, col-embed replicated H times
        pe = torch.cat([
            pe_h.unsqueeze(1).expand(-1, w, -1).reshape(h * w, half),
            pe_w.unsqueeze(0).expand(h, -1, -1).reshape(h * w, half),
        ], dim=-1)                                      # (HW, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, HW, D)
        return x + self.pe.unsqueeze(0)


class _CausalTokenDecoder(nn.Module):
    """
    FSCI: 3 learnable tokens (X, W, Z) attend to spatial image features via a
    single-layer Transformer decoder.
      X = head-pose context
      W = gaze content
      Z = confounding / appearance
    """

    def __init__(self, d_model: int = 512, nhead: int = 8, dropout: float = 0.1):
        super().__init__()
        self.tokens = nn.Parameter(
            torch.randn(3, d_model) / math.sqrt(d_model)
        )
        self.decoder = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, img_feats: torch.Tensor) -> torch.Tensor:
        """
        img_feats: (B, HW, D)
        returns  : (B, 3, D)  — [X, W, Z] tokens
        """
        B = img_feats.shape[0]
        tgt = self.tokens.unsqueeze(0).expand(B, -1, -1)    # (B, 3, D)
        out = self.decoder(tgt, img_feats)
        return self.norm(out)


class _CausalIntervention(nn.Module):
    """
    FSCI: fuse X, W, Ez into a gaze feature, then predict gaze.

    gaze_fea = 0.5 * W + 0.5 * (hd_fc(X) + ez_fc(Ez))
    gaze      = gaze_fc(gaze_fea)
    """

    def __init__(self, d_model: int = 512, dropout: float = 0.3):
        super().__init__()
        self.hd_fc = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(inplace=True))
        self.ez_fc = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(inplace=True))
        self.gaze_fc = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 2),
        )

    def forward(self, X, W, Ez):
        """All inputs: (B, D).  Returns (gaze: B×2, gaze_fea: B×D)."""
        gaze_fea = 0.5 * W + 0.5 * (self.hd_fc(X) + self.ez_fc(Ez))
        return self.gaze_fc(gaze_fea), gaze_fea


# ============================================================
# Unified model
# ============================================================

class UnifiedGazeNet(nn.Module):
    """
    Single model class covering all four ablation stages.

    Parameters
    ----------
    stage : int
        0 = Baseline, 1 = +FSCI, 2 = +FSCI+GFAL
        (Stage 3 / +AGG is applied post-training via gpm_utils.py)
    pretrained : bool
        Load ImageNet weights for the ResNet-18 backbone.
    d_model : int
        Feature dimension (512 for ResNet-18).
    ez_momentum : float
        EMA momentum for the confounding-feature running mean (Ez).
    """

    def __init__(
        self,
        stage: int = 0,
        pretrained: bool = True,
        d_model: int = 512,
        ez_momentum: float = 0.9,
        dropout: float = 0.3,
    ):
        super().__init__()
        assert stage in (0, 1, 2), f'stage must be 0, 1 or 2; got {stage}'
        self.stage = stage

        # ── Backbone (shared across all stages) ──────────────────────────────
        self.backbone = _build_resnet18(pretrained)
        self.avgpool  = nn.AdaptiveAvgPool2d((1, 1))

        # ── Stage 0: simple FC head ───────────────────────────────────────────
        if stage == 0:
            self.head = nn.Sequential(
                nn.Linear(d_model, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(256, 2),
            )

        # ── Stage 1/2: FSCI causal components ────────────────────────────────
        if stage >= 1:
            self.pos_embed   = _SinPos2D(d_model)
            self.token_dec   = _CausalTokenDecoder(d_model, nhead=8, dropout=0.1)
            self.causal_int  = _CausalIntervention(d_model, dropout=dropout)
            # EMA buffer for the confounding-feature running mean
            self.register_buffer('Ez',          torch.zeros(1, d_model))
            self.register_buffer('Ez_momentum', torch.tensor(ez_momentum))

        # ── Stage 2: GFAL frontalization head on W-token ─────────────────────
        if stage >= 2:
            self.frontal_head = nn.Sequential(
                nn.Linear(d_model, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 2),
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def reset_ez(self):
        """Call at the start of each epoch to clear the EMA buffer (no-op for stage 0)."""
        if self.stage >= 1:
            self.Ez.zero_()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, img: torch.Tensor, head_pose: torch.Tensor = None, mode: str = 'train'):
        """
        Parameters
        ----------
        img       : (B, 3, H, W)
        head_pose : (B, 2)  — [yaw, pitch] in radians  (needed for stage ≥ 2)
        mode      : 'train' | 'eval'

        Returns
        -------
        gaze_pred : (B, 2)
        gaze_feat : (B, 512)   pre-FC feature for AGG calibration
        g0_pred   : (B, 2) | None   canonical gaze (stage 2 only)
        """
        # Backbone
        feat_map = self.backbone(img)              # (B, 512, 7, 7)

        # ── Stage 0 ─────────────────────────────────────────────────────────
        if self.stage == 0:
            gap = self.avgpool(feat_map).flatten(1)    # (B, 512)
            gaze_pred = self.head(gap)
            return gaze_pred, gap, None

        # ── Stage 1/2: reshape → add pos-embed ──────────────────────────────
        B, C, H, W = feat_map.shape
        img_feats = feat_map.permute(0, 2, 3, 1).reshape(B, H * W, C)  # (B,49,512)
        img_feats = self.pos_embed(img_feats)

        # Causal token decoder
        tokens = self.token_dec(img_feats)          # (B, 3, 512)
        X = tokens[:, 0]                            # head-pose token
        W = tokens[:, 1]                            # gaze token
        Z = tokens[:, 2]                            # confounding token

        # Update Ez (EMA of Z across batches — causal intervention)
        with torch.no_grad():
            z_mean = Z.mean(0, keepdim=True)        # (1, 512)
            if mode == 'train':
                self.Ez = self.Ez_momentum * self.Ez + (1.0 - self.Ez_momentum) * z_mean

        Ez = self.Ez.expand(B, -1)                  # (B, 512)

        gaze_pred, gaze_feat = self.causal_int(X, W, Ez)

        # ── Stage 2: GFAL canonical gaze ────────────────────────────────────
        g0_pred = None
        if self.stage >= 2:
            g0_pred = self.frontal_head(W)          # (B, 2) — canonical [yaw,pitch]

        return gaze_pred, gaze_feat, g0_pred

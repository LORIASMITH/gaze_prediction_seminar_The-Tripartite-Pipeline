"""
AGG – Geodesic Projection Module (GPM) calibration.

Usage pattern:
    # After training a stage-1/2 model, collect source-domain features:
    calibrator = GPMCalibrator()
    calibrator.fit(source_features_np, source_gaze_labels_np)

    # At inference on the target domain:
    gaze_pred_np = calibrator.predict(target_features_np)
    error = angular_error_np(gaze_pred_np, target_labels_np).mean()

This is a thin wrapper around the SphereAlignment code in the official AGG repo.
It falls back to a sklearn ISOMap + least-squares GeodesicProjection.
"""

import os
import sys
import numpy as np
import pickle

# Make the official AGG SphereAlignment importable
_AGG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'Analytical-Gaze-Generalization-framework',
)
if _AGG_DIR not in sys.path:
    sys.path.insert(0, _AGG_DIR)

try:
    from SphereAlignment import ISOMap, FitGaze, GeodesicProjection  # type: ignore
    _AGG_AVAILABLE = True
except ImportError:
    _AGG_AVAILABLE = False
    print('[GPMCalibrator] Warning: AGG SphereAlignment not found; '
          'falling back to linear regression.')

from utils.metrics import angular_error_np


class GPMCalibrator:
    """
    Fits a geodesic projection on top of backbone features extracted from the
    source domain, then applies it to the target domain at inference time.

    Parameters
    ----------
    n_neighbors : int
        k for ISOMap; None → auto (40 % of training samples, capped at 300).
    iso_dim : int
        Number of ISOMap output dimensions (3 by default; the GPM fits a sphere
        in this space).
    """

    def __init__(self, n_neighbors: int = None, iso_dim: int = 3):
        self.n_neighbors = n_neighbors
        self.iso_dim     = iso_dim
        self._iso_fitter = None
        self._gpm_param  = None
        self._model      = None
        self._fallback   = False

    # ------------------------------------------------------------------

    def fit(self, features: np.ndarray, gaze_labels: np.ndarray,
            tgt_features: np.ndarray = None) -> float:
        """
        Fit ISOMap + GeodesicProjection on source-domain data.

        features     : (N_src, D) — source backbone features (labeled)
        gaze_labels  : (N_src, 2) — source [yaw, pitch] in radians
        tgt_features : (N_tgt, D) — target backbone features (unlabeled, optional)
                       When provided, ISOMap is fitted on src+tgt jointly
                       (transductive mode) so the manifold covers both domains.
                       GeodesicProjection is still fitted on source labels only.
        Returns mean fitting error on source in degrees.
        """
        n_src = features.shape[0]
        k = self.n_neighbors if self.n_neighbors is not None else min(300, max(10, int(n_src * 0.4)))

        if not _AGG_AVAILABLE:
            return self._fit_fallback(features, gaze_labels)

        # Transductive: concatenate source + target for ISOMap, remember split
        if tgt_features is not None:
            all_features = np.concatenate([features, tgt_features], axis=0)
            self._n_src = n_src
            mode_str = f'transductive (src={n_src}, tgt={tgt_features.shape[0]})'
        else:
            all_features = features
            self._n_src = None
            mode_str = f'inductive (src={n_src})'

        print(f'[GPM] Fitting ISOMap [{mode_str}] (k={k}, dim={self.iso_dim}) ...')
        pgf_all, self._iso_fitter = ISOMap(all_features, dim=self.iso_dim,
                                           n_neighbors=k, verbose=True)

        # GeodesicProjection is fitted on source points only
        pgf_src = pgf_all[:n_src]
        print('[GPM] Fitting GeodesicProjection (source only) ...')
        self._gpm_param, gaze_pred, self._model = FitGaze(pgf_src, gaze_labels, verbose=True)

        from utils.metrics import angular_error_np as _ae, gazeto3d_np
        labels_3d = gazeto3d_np(gaze_labels)
        fit_err = _ae(gaze_pred, labels_3d).mean()
        print(f'[GPM] Source fitting error: {fit_err:.2f}°')
        return float(fit_err)

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Predict gaze for target-domain features.

        In transductive mode (tgt_features passed to fit), the target embedding
        was already computed during fit — we retrieve it directly from the fitter
        to avoid out-of-sample projection error accumulation.

        features : (N_tgt, D) — must match the tgt_features passed to fit()
        Returns  : (N, 3) 3-D gaze vectors
        """
        if self._fallback:
            return self._predict_fallback(features)

        assert self._iso_fitter is not None, 'Call fit() first.'

        if self._n_src is not None:
            # Transductive: target embedding is the tail of the full fitted embedding
            pgf_tgt = self._iso_fitter.embedding_[self._n_src:]
        else:
            pgf_tgt, _ = ISOMap(features, fitter=self._iso_fitter, verbose=False)

        return self._model(pgf_tgt)   # (N, 3) unit vectors

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({
                'iso_fitter': self._iso_fitter,
                'gpm_param':  self._gpm_param,
                'fallback':   self._fallback,
                'coef':       getattr(self, '_coef', None),
            }, f)

    def load(self, path: str):
        with open(path, 'rb') as f:
            d = pickle.load(f)
        self._iso_fitter = d['iso_fitter']
        self._gpm_param  = d['gpm_param']
        self._fallback   = d['fallback']
        if d['coef'] is not None:
            self._coef = d['coef']
        if self._gpm_param is not None:
            self._model = GeodesicProjection(None, None, self._gpm_param)

    # ------------------------------------------------------------------
    # Fallback: simple PCA + linear regression when AGG code is missing
    # ------------------------------------------------------------------

    def _fit_fallback(self, features, gaze_labels):
        from sklearn.decomposition import PCA
        from sklearn.linear_model import Ridge
        from utils.metrics import gazeto3d_np, angular_error_np

        print('[GPM-fallback] PCA(50) + Ridge regression')
        self._pca = PCA(n_components=min(50, features.shape[1]))
        feat_r = self._pca.fit_transform(features)
        g3d    = gazeto3d_np(gaze_labels)
        self._coef = Ridge(alpha=1.0).fit(feat_r, g3d)
        pred   = self._coef.predict(feat_r)
        fit_err = angular_error_np(pred, g3d).mean()
        print(f'[GPM-fallback] Fitting error: {fit_err:.2f}°')
        self._fallback = True
        return float(fit_err)

    def _predict_fallback(self, features):
        feat_r = self._pca.transform(features)
        return self._coef.predict(feat_r)               # (N, 3)


# ──────────────────────────────────────────────────────────────────────────────
# Helper: collect features + labels from a DataLoader with a trained model
# ──────────────────────────────────────────────────────────────────────────────

def collect_features(model, loader, device):
    """
    Run a model in eval mode over a DataLoader and collect gaze_feat + labels.

    Returns
    -------
    features : np.ndarray  (N, D)
    labels   : np.ndarray  (N, 2)  [yaw, pitch] in radians
    """
    import torch
    model.eval()
    all_feats, all_labels = [], []
    with torch.no_grad():
        for imgs, gazes, head_poses in loader:
            imgs       = imgs.to(device)
            head_poses = head_poses.to(device)
            _, gaze_feat, _ = model(imgs, head_poses, mode='eval')
            all_feats.append(gaze_feat.cpu().numpy())
            all_labels.append(gazes.numpy())
    return np.concatenate(all_feats, axis=0), np.concatenate(all_labels, axis=0)



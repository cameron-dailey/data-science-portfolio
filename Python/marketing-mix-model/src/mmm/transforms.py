"""
Transforms for Marketing Mix Modeling (adstock and saturation).
"""

import numpy as np
import pandas as pd

def geometric_adstock(x: np.ndarray, lam: float) -> np.ndarray:
    """
    Geometric adstock (carryover).
    x : 1D array of spends over time
    lam : carryover rate in [0, 1). Higher => longer memory.
    """
    if not 0 <= lam < 1:
        raise ValueError("lam must be in [0, 1)")
    out = np.zeros_like(x, dtype=float)
    carry = 0.0
    for t, val in enumerate(x):
        carry = val + lam * carry
        out[t] = carry
    return out


def hill_saturation(a: np.ndarray, alpha: float, theta: float) -> np.ndarray:
    """
    Hill (S-curve) saturation.
    alpha : controls curve steepness (>0)
    theta : half-saturation point (>0)
    """
    if alpha <= 0 or theta <= 0:
        raise ValueError("alpha and theta must be > 0")
    a = np.asarray(a, dtype=float)
    a_alpha = np.power(a, alpha)
    return a_alpha / (a_alpha + np.power(theta, alpha))


def adstock_then_hill(x: np.ndarray, lam: float, alpha: float, theta: float) -> np.ndarray:
    """Convenience: apply geometric adstock then hill saturation."""
    return hill_saturation(geometric_adstock(x, lam), alpha, theta)


def apply_channel_transforms(df: pd.DataFrame, channel_cols, lam, alpha, theta, prefix="tr_"):
    """
    Apply adstock+hill to multiple channels.
    lam, alpha, theta can be floats (same for all) or dicts keyed by channel.
    """
    out = df.copy()
    for c in channel_cols:
        lam_c = lam[c] if isinstance(lam, dict) else lam
        alpha_c = alpha[c] if isinstance(alpha, dict) else alpha
        theta_c = theta[c] if isinstance(theta, dict) else theta
        out[f"{prefix}{c}"] = adstock_then_hill(out[c].to_numpy(), lam_c, alpha_c, theta_c)
    return out

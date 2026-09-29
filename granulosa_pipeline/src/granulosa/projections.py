from __future__ import annotations

from typing import Tuple

import numpy as np


def max_projection(stack_zyx: np.ndarray) -> np.ndarray:
    """MIP over Z. Input (Z, Y, X) -> (Y, X)."""
    if stack_zyx.ndim != 3:
        raise ValueError(f"Expected 3D (Z,Y,X), got shape {stack_zyx.shape}")
    return stack_zyx.max(axis=0)


def mean_projection(stack_zyx: np.ndarray) -> np.ndarray:
    if stack_zyx.ndim != 3:
        raise ValueError(f"Expected 3D (Z,Y,X), got shape {stack_zyx.shape}")
    return stack_zyx.mean(axis=0)


def cell_z_range(stack_zyx: np.ndarray, mask_yx: np.ndarray) -> Tuple[int, int]:
    """Estimate the z-range a cell spans by Otsu on per-slice mean DAPI within the mask.

    Returns (z_start, z_end_exclusive). For single-Z inputs returns (0, 1).
    """
    from skimage.filters import threshold_otsu

    Z = stack_zyx.shape[0]
    if Z == 1 or mask_yx.sum() == 0:
        return 0, Z
    per_z = np.array([stack_zyx[z][mask_yx].mean() for z in range(Z)])
    try:
        thr = threshold_otsu(per_z)
    except Exception:
        return 0, Z
    active = np.where(per_z >= thr)[0]
    if len(active) == 0:
        return 0, Z
    return int(active.min()), int(active.max()) + 1

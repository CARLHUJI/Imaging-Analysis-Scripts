from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


@dataclass
class ND2Image:
    """Container for a loaded ND2 image.

    data shape: (Z, C, Y, X). Single-Z acquisitions are promoted to Z=1.
    """
    path: Path
    data: np.ndarray        # (Z, C, Y, X)
    channels: List[str]     # channel names, order matches C axis
    pixel_size_um: float    # XY pixel size in µm
    z_step_um: float        # z spacing in µm; 0.0 if single-Z


def _extract_channel_names(metadata) -> List[str]:
    """Best-effort channel-name extraction from nd2 metadata."""
    names: List[str] = []
    try:
        for ch in metadata.channels:
            nm = getattr(ch.channel, "name", None) or getattr(ch, "name", None)
            names.append(str(nm) if nm else f"ch{len(names)}")
    except Exception:
        pass
    return names


def _extract_pixel_size(vs) -> float:
    """voxel size in µm; nd2 returns (Z, Y, X). Use X (== Y for square pixels)."""
    x = getattr(vs, "x", None)
    if x is None or x <= 0:
        raise ValueError("ND2 metadata missing XY pixel size; cannot proceed.")
    return float(x)


def load_nd2(path: Path) -> ND2Image:
    import nd2  # local import so this module imports even without nd2 installed

    with nd2.ND2File(str(path)) as f:
        arr = f.asarray()  # dims like TZCYX, ZCYX, CYX, etc.
        sizes = f.sizes
        vs = f.voxel_size()
        names = _extract_channel_names(f.metadata)

    # Normalise axis order to (Z, C, Y, X)
    axes = list(sizes.keys())
    # Squeeze non-relevant axes if size 1
    while arr.ndim > 4:
        squeezed = False
        for ax_idx, ax in enumerate(axes):
            if arr.shape[ax_idx] == 1 and ax not in ("Z", "C", "Y", "X"):
                arr = np.squeeze(arr, axis=ax_idx)
                del axes[ax_idx]
                squeezed = True
                break
        if not squeezed:
            break

    # Insert missing Z / C axes as size-1
    if "Z" not in axes:
        arr = arr[np.newaxis, ...]
        axes = ["Z"] + axes
    if "C" not in axes:
        z_idx = axes.index("Z")
        arr = np.expand_dims(arr, axis=z_idx + 1)
        axes.insert(z_idx + 1, "C")

    # Transpose to Z, C, Y, X
    target = ["Z", "C", "Y", "X"]
    perm = [axes.index(a) for a in target]
    arr = np.transpose(arr, perm)

    if not names or len(names) != arr.shape[1]:
        names = [f"ch{i}" for i in range(arr.shape[1])]

    pixel_um = _extract_pixel_size(vs)
    z_step = float(getattr(vs, "z", 0.0) or 0.0) if arr.shape[0] > 1 else 0.0

    return ND2Image(
        path=Path(path),
        data=arr,
        channels=names,
        pixel_size_um=pixel_um,
        z_step_um=z_step,
    )


def channel_index(img: ND2Image, name: str) -> int:
    """Case-insensitive channel lookup. Substring match falls back to exact."""
    lname = name.lower()
    exact = [i for i, c in enumerate(img.channels) if c.lower() == lname]
    if exact:
        return exact[0]
    partial = [i for i, c in enumerate(img.channels) if lname in c.lower()]
    if partial:
        return partial[0]
    raise KeyError(
        f"Channel {name!r} not found in {img.path.name}. "
        f"Available: {img.channels}"
    )


def ensure_channels(img: ND2Image, required: List[str]) -> Dict[str, int]:
    """Resolve each required channel to its index; raise if any missing."""
    missing = []
    idx_map: Dict[str, int] = {}
    for name in required:
        try:
            idx_map[name] = channel_index(img, name)
        except KeyError:
            missing.append(name)
    if missing:
        raise KeyError(
            f"{img.path.name}: missing required channel(s) {missing}. "
            f"Available: {img.channels}"
        )
    return idx_map

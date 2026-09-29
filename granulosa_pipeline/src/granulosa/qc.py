from __future__ import annotations

from typing import List

from .config import Config
from .measurement import CellRow


def apply_qc(rows: List[CellRow], cfg: Config) -> None:
    """Mutate CellRow.qc_pass / qc_reason in place based on config thresholds.

    Each row is checked against every enabled filter; the first failure wins.
    """
    for r in rows:
        reasons = []
        if r.area_um2 < cfg.MIN_AREA_UM2:
            reasons.append(f"area<{cfg.MIN_AREA_UM2}um2")
        if r.area_um2 > cfg.MAX_AREA_UM2:
            reasons.append(f"area>{cfg.MAX_AREA_UM2}um2")
        if r.solidity < cfg.MIN_SOLIDITY:
            reasons.append(f"solidity<{cfg.MIN_SOLIDITY}")
        if r.eccentricity > cfg.MAX_ECCENTRICITY:
            reasons.append(f"eccentricity>{cfg.MAX_ECCENTRICITY}")
        if r.focus_score < cfg.FOCUS_MIN:
            reasons.append(f"focus<{cfg.FOCUS_MIN}")
        if cfg.DAPI_INTENSITY_MAX is not None:
            dapi_key = "DAPI_mip_roi_mean" if "DAPI_mip_roi_mean" in r.intensities \
                else next((k for k in r.intensities if k.lower().startswith("dapi") and k.endswith("_roi_mean")), None)
            if dapi_key and r.intensities[dapi_key] > cfg.DAPI_INTENSITY_MAX:
                reasons.append(f"dapi>{cfg.DAPI_INTENSITY_MAX}")
        if reasons:
            r.qc_pass = False
            r.qc_reason = ";".join(reasons)

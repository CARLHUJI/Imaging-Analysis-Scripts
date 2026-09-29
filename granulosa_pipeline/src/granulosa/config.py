from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional

import yaml


@dataclass
class Config:
    SEG_CHANNEL: str = "dapi+gfp"
    ROI_MODE: str = "whole_cell"
    CYTO_EXPAND_UM: float = 2.0
    MEASURE_CHANNELS: List[str] = field(default_factory=lambda: ["DAPI", "GFP"])

    BACKGROUND_MODE: str = "shared_patch"
    SPECK_PERCENTILE: float = 95.0
    RING_INNER_UM: float = 1.5
    RING_OUTER_UM: float = 5.0
    BG_CELL_BUFFER_UM: float = 2.0

    MIN_AREA_UM2: float = 30.0
    MAX_AREA_UM2: float = 400.0
    MIN_SOLIDITY: float = 0.85
    MAX_ECCENTRICITY: float = 0.90
    DAPI_INTENSITY_MAX: Optional[float] = None
    FOCUS_MIN: float = 20.0

    CELLPOSE_NUCLEI_MODEL: str = "nuclei"
    CELLPOSE_CYTO_MODEL: str = "cyto3"
    CELLPOSE_DIAMETER_UM: Optional[float] = None
    CELLPOSE_FLOW_THRESHOLD: float = 0.4
    CELLPOSE_CELLPROB_THRESHOLD: float = 0.0
    CELLPOSE_GPU: bool = False
    CELLPOSE_MIN_SIZE: int = 15

    RANDOM_SEED: int = 0
    OVERLAY_DPI: int = 150

    def validate(self) -> None:
        if self.SEG_CHANNEL not in ("dapi", "dapi+gfp"):
            raise ValueError(f"SEG_CHANNEL must be 'dapi' or 'dapi+gfp' (got {self.SEG_CHANNEL!r})")
        if self.ROI_MODE not in ("nuclear", "whole_cell"):
            raise ValueError(f"ROI_MODE must be 'nuclear' or 'whole_cell' (got {self.ROI_MODE!r})")
        if self.BACKGROUND_MODE not in ("shared_patch", "ring"):
            raise ValueError(f"BACKGROUND_MODE must be 'shared_patch' or 'ring' (got {self.BACKGROUND_MODE!r})")
        if self.RING_INNER_UM >= self.RING_OUTER_UM:
            raise ValueError("RING_INNER_UM must be < RING_OUTER_UM")
        if not (0 < self.SPECK_PERCENTILE <= 100):
            raise ValueError("SPECK_PERCENTILE must be in (0, 100]")

    def as_dict(self) -> dict:
        return asdict(self)


def load_config(path: Optional[Path]) -> Config:
    if path is None:
        cfg = Config()
    else:
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        cfg = Config(**raw)
    cfg.validate()
    return cfg

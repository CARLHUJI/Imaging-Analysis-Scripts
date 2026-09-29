"""Extra tests for the background-selection module."""
from __future__ import annotations

import numpy as np

from src.granulosa.background import build_background_pool


def test_pool_buffer_respected():
    H = W = 100
    cell = np.zeros((H, W), dtype=np.int32)
    cell[45:55, 45:55] = 1
    gfp = np.full((H, W), 100.0, dtype=np.float32)
    dapi = np.full((H, W), 50.0, dtype=np.float32)

    pool = build_background_pool(
        mip_by_channel={"GFP": gfp, "DAPI": dapi},
        cell_mask=cell,
        pixel_size_um=1.0,   # 1 µm/px
        speck_percentile=99.0,
        cell_buffer_um=5.0,  # 5 px Euclidean buffer
        reference_gfp_channel="GFP",
    )
    # Cell pixels always excluded.
    assert not pool[45:55, 45:55].any()
    # Pixels orthogonally within 5 px of the cell must be excluded.
    # (rows 40..44 and 55..59 across the cell's column span)
    assert not pool[40:45, 45:55].any()
    assert not pool[55:60, 45:55].any()
    assert not pool[45:55, 40:45].any()
    assert not pool[45:55, 55:60].any()
    # Pixels far from the cell must be included.
    assert pool[0:10, 0:10].all()
    assert pool[85:95, 85:95].all()


def test_pool_without_gfp_reference_falls_back_to_dapi_empty():
    H = W = 60
    cell = np.zeros((H, W), dtype=np.int32)
    cell[20:40, 20:40] = 1
    dapi = np.zeros((H, W), dtype=np.float32)

    pool = build_background_pool(
        mip_by_channel={"DAPI": dapi},
        cell_mask=cell,
        pixel_size_um=1.0,
        speck_percentile=95.0,
        cell_buffer_um=0.0,
        reference_gfp_channel=None,
    )
    assert not pool[cell == 1].any()
    assert pool[0:10, 0:10].all()

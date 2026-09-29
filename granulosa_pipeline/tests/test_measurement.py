"""Synthetic-image tests: verify ROI mean/median/integrated, background subtraction,
z-span vs MIP consistency, and shared-patch selection logic."""
from __future__ import annotations

import numpy as np
import pytest

from src.granulosa.background import (
    BackgroundPatch,
    build_background_pool,
    find_shared_patch,
)
from src.granulosa.measurement import measure_cells
from src.granulosa.projections import cell_z_range, max_projection


def _make_image(H=200, W=200, Z=5, cell_val=1000, bg_val=100, speck_val=800):
    """One square cell centered, uniform background, one non-cell GFP speck.

    Returns dapi_stack, gfp_stack, cell_mask, expected values.
    """
    dapi = np.full((Z, H, W), bg_val, dtype=np.uint16)
    gfp = np.full((Z, H, W), bg_val, dtype=np.uint16)

    # Cell: 40x40 square, present in z=1..3 (3 slices)
    cy, cx = H // 2, W // 2
    rr, cc = slice(cy - 20, cy + 20), slice(cx - 20, cx + 20)
    dapi[1:4, rr, cc] = cell_val
    gfp[1:4, rr, cc] = cell_val + 500

    # Non-cell GFP speck outside cell
    gfp[:, 10:14, 10:14] = speck_val

    cell_mask = np.zeros((H, W), dtype=np.int32)
    cell_mask[rr, cc] = 1
    return dapi, gfp, cell_mask


def test_roi_mean_and_integrated_exact():
    dapi_stack, gfp_stack, cell_mask = _make_image()
    dapi_mip = max_projection(dapi_stack)
    gfp_mip = max_projection(gfp_stack)

    # Build a background patch far from the cell and the speck
    patch = BackgroundPatch(
        row=150, col=150, rows=20, cols=20,
        mask=np.zeros_like(dapi_mip, dtype=bool),
        per_channel_mean={"DAPI": 100.0, "GFP": 100.0},
        per_channel_median={"DAPI": 100.0, "GFP": 100.0},
    )
    patch.mask[150:170, 150:170] = True

    rows = measure_cells(
        image_name="test.nd2",
        cell_mask=cell_mask,
        dapi_mip=dapi_mip,
        stacks_by_channel={"DAPI": dapi_stack, "GFP": gfp_stack},
        mips_by_channel={"DAPI": dapi_mip, "GFP": gfp_mip},
        background=patch,
        pixel_size_um=0.1,
        measure_channels=["DAPI", "GFP"],
    )
    assert len(rows) == 1
    r = rows[0]

    area_px = 40 * 40  # 1600
    assert r.area_px == area_px

    # DAPI MIP: cell region = 1000, background patch = 100
    assert r.intensities["DAPI_mip_roi_mean"] == pytest.approx(1000.0)
    assert r.intensities["DAPI_mip_roi_integrated"] == pytest.approx(1000.0 * area_px)
    assert r.intensities["DAPI_mip_bg_mean"] == pytest.approx(100.0)
    assert r.intensities["DAPI_mip_corrected_mean"] == pytest.approx(900.0)
    assert r.intensities["DAPI_mip_corrected_integrated"] == pytest.approx(900.0 * area_px)

    # GFP MIP: cell = 1500, background patch = 100 (patch avoided the speck)
    assert r.intensities["GFP_mip_roi_mean"] == pytest.approx(1500.0)
    assert r.intensities["GFP_mip_bg_mean"] == pytest.approx(100.0)
    assert r.intensities["GFP_mip_corrected_mean"] == pytest.approx(1400.0)


def test_zspan_matches_mip_for_uniform_signal():
    """Cell has identical signal across its z-range; MIP mean should equal
    z-span mean (both = cell_val - bg_val after correction)."""
    dapi_stack, gfp_stack, cell_mask = _make_image()
    dapi_mip = max_projection(dapi_stack)
    gfp_mip = max_projection(gfp_stack)
    patch = BackgroundPatch(
        row=150, col=150, rows=20, cols=20,
        mask=np.zeros_like(dapi_mip, dtype=bool),
        per_channel_mean={"DAPI": 100.0, "GFP": 100.0},
        per_channel_median={"DAPI": 100.0, "GFP": 100.0},
    )
    patch.mask[150:170, 150:170] = True

    rows = measure_cells(
        image_name="test.nd2",
        cell_mask=cell_mask,
        dapi_mip=dapi_mip,
        stacks_by_channel={"DAPI": dapi_stack, "GFP": gfp_stack},
        mips_by_channel={"DAPI": dapi_mip, "GFP": gfp_mip},
        background=patch,
        pixel_size_um=0.1,
        measure_channels=["GFP"],
    )
    r = rows[0]
    # Cell z-range should be detected as z=1..4 (mean signal above background there)
    assert 1 <= r.z_start <= 2
    assert 3 <= r.z_end <= 4
    assert r.intensities["GFP_zspan_roi_mean"] == pytest.approx(
        r.intensities["GFP_mip_roi_mean"], rel=1e-6
    )


def test_signal_over_background_ratio():
    dapi_stack, gfp_stack, cell_mask = _make_image()
    dapi_mip = max_projection(dapi_stack)
    gfp_mip = max_projection(gfp_stack)
    patch = BackgroundPatch(
        row=150, col=150, rows=20, cols=20,
        mask=np.zeros_like(dapi_mip, dtype=bool),
        per_channel_mean={"GFP": 100.0}, per_channel_median={"GFP": 100.0},
    )
    patch.mask[150:170, 150:170] = True
    rows = measure_cells(
        image_name="t.nd2", cell_mask=cell_mask, dapi_mip=dapi_mip,
        stacks_by_channel={"DAPI": dapi_stack, "GFP": gfp_stack},
        mips_by_channel={"DAPI": dapi_mip, "GFP": gfp_mip},
        background=patch, pixel_size_um=0.1, measure_channels=["GFP"],
    )
    r = rows[0]
    assert r.intensities["GFP_mip_signal_over_bg"] == pytest.approx(1500.0 / 100.0)


def test_background_pool_excludes_gfp_speck():
    """The GFP speck at 10:14 must NOT end up in the background pool."""
    dapi_stack, gfp_stack, cell_mask = _make_image()
    gfp_mip = max_projection(gfp_stack)
    dapi_mip = max_projection(dapi_stack)

    pool = build_background_pool(
        mip_by_channel={"DAPI": dapi_mip, "GFP": gfp_mip},
        cell_mask=cell_mask,
        pixel_size_um=0.1,
        speck_percentile=95.0,
        cell_buffer_um=1.0,
        reference_gfp_channel="GFP",
    )
    # All speck pixels excluded
    assert not pool[10:14, 10:14].any()
    # All cell pixels excluded
    assert not pool[cell_mask == 1].any()
    # Plenty of far-away background pixels remain
    assert pool[100:120, 150:170].all()


def test_find_shared_patch_prefers_speck_free_region():
    """Given a pool that includes a bright region and a dim region, the picker
    should choose the dim one because it has lower median GFP."""
    H = W = 200
    pool = np.zeros((H, W), dtype=bool)
    pool[10:60, 10:60] = True    # region A: bright GFP
    pool[100:170, 100:170] = True  # region B: dim GFP
    gfp = np.full((H, W), 100.0, dtype=np.float32)
    gfp[10:60, 10:60] = 400.0
    gfp[100:170, 100:170] = 105.0
    dapi = np.full((H, W), 50.0, dtype=np.float32)

    patch = find_shared_patch(
        pool_mask=pool,
        target_area_px=40 * 40,
        mip_by_channel={"GFP": gfp, "DAPI": dapi},
    )
    # Chosen patch should sit inside the dim region (row >= 100, col >= 100)
    assert patch.row >= 100 and patch.col >= 100
    assert patch.per_channel_mean["GFP"] < 200.0

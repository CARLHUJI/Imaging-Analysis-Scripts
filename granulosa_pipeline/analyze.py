#!/usr/bin/env python
"""Batch granulosa-cell image analysis pipeline.

CLI:
  python analyze.py --input DIR --output DIR [--config config.yaml] [--dry-run]
"""
from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import tifffile

from src.granulosa.background import build_background_pool, build_ring_masks, find_shared_patch
from src.granulosa.config import Config, load_config
from src.granulosa.excel import write_workbook
from src.granulosa.io_nd2 import ND2Image, ensure_channels, load_nd2
from src.granulosa.measurement import measure_cells
from src.granulosa.overlays import save_overlay
from src.granulosa.projections import max_projection
from src.granulosa.qc import apply_qc
from src.granulosa.rois_ij import export_imagej_rois
from src.granulosa.segmentation import (
    cellpose_model_info,
    dilate_nuclear_to_whole_cell,
    segment_cyto,
    segment_nuclei,
)


logger = logging.getLogger("granulosa")


def _setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, mode="w"),
        ],
    )


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _mip_by_channel(img: ND2Image) -> Dict[str, np.ndarray]:
    return {name: max_projection(img.data[:, i, :, :]) for i, name in enumerate(img.channels)}


def _stack_by_channel(img: ND2Image) -> Dict[str, np.ndarray]:
    return {name: img.data[:, i, :, :] for i, name in enumerate(img.channels)}


def process_image(nd2_path: Path, out_root: Path, cfg: Config, dry_run: bool = False) -> Dict:
    logger.info("=" * 78)
    logger.info("Processing %s", nd2_path.name)
    img = load_nd2(nd2_path)
    logger.info(
        "  shape=%s  channels=%s  pixel=%.4f um  z_step=%.4f um",
        img.data.shape, img.channels, img.pixel_size_um, img.z_step_um,
    )

    # Validate required channels
    required = list({"DAPI", *cfg.MEASURE_CHANNELS})
    if cfg.SEG_CHANNEL == "dapi+gfp":
        required.append("GFP")
    ensure_channels(img, required)

    stacks = _stack_by_channel(img)
    mips = _mip_by_channel(img)

    # --- MIP TIFFs ---
    mip_dir = out_root / "mips"
    mip_dir.mkdir(parents=True, exist_ok=True)
    for ch, arr in mips.items():
        tifffile.imwrite(mip_dir / f"{nd2_path.stem}_{ch}.tif", arr.astype(np.float32))

    # --- Segmentation ---
    dapi_mip = mips["DAPI"] if "DAPI" in mips else next(iter(mips.values()))
    if cfg.SEG_CHANNEL == "dapi+gfp":
        gfp_mip = mips.get("GFP") or next(iter(v for k, v in mips.items() if "gfp" in k.lower()))
        cyto_mask, nuclear_mask = segment_cyto(
            dapi_yx=dapi_mip, gfp_yx=gfp_mip,
            pixel_size_um=img.pixel_size_um,
            model_name=cfg.CELLPOSE_CYTO_MODEL,
            diameter_um=cfg.CELLPOSE_DIAMETER_UM,
            flow_threshold=cfg.CELLPOSE_FLOW_THRESHOLD,
            cellprob_threshold=cfg.CELLPOSE_CELLPROB_THRESHOLD,
            gpu=cfg.CELLPOSE_GPU, min_size=cfg.CELLPOSE_MIN_SIZE,
        )
        roi_mask = cyto_mask if cfg.ROI_MODE == "whole_cell" else nuclear_mask
    else:
        nuclear_mask = segment_nuclei(
            dapi_yx=dapi_mip,
            pixel_size_um=img.pixel_size_um,
            model_name=cfg.CELLPOSE_NUCLEI_MODEL,
            diameter_um=cfg.CELLPOSE_DIAMETER_UM,
            flow_threshold=cfg.CELLPOSE_FLOW_THRESHOLD,
            cellprob_threshold=cfg.CELLPOSE_CELLPROB_THRESHOLD,
            gpu=cfg.CELLPOSE_GPU, min_size=cfg.CELLPOSE_MIN_SIZE,
        )
        if cfg.ROI_MODE == "whole_cell":
            roi_mask = dilate_nuclear_to_whole_cell(
                nuclear_mask, expand_um=cfg.CYTO_EXPAND_UM,
                pixel_size_um=img.pixel_size_um,
            )
        else:
            roi_mask = nuclear_mask

    n_cells = int(roi_mask.max())
    if n_cells == 0:
        logger.warning("No cells segmented in %s; skipping.", nd2_path.name)
        return {"image_name": nd2_path.name, "cells": [], "pixel_um": img.pixel_size_um,
                "z_step_um": img.z_step_um}

    # --- Background ---
    pool = build_background_pool(
        mip_by_channel=mips,
        cell_mask=roi_mask,
        pixel_size_um=img.pixel_size_um,
        speck_percentile=cfg.SPECK_PERCENTILE,
        cell_buffer_um=cfg.BG_CELL_BUFFER_UM,
        reference_gfp_channel="GFP" if "GFP" in mips else None,
    )

    # median cell area for the shared patch size
    unique_labels = [int(l) for l in np.unique(roi_mask) if l != 0]
    areas = [int((roi_mask == l).sum()) for l in unique_labels]
    med_area = int(np.median(areas)) if areas else 100

    if cfg.BACKGROUND_MODE == "shared_patch":
        background = find_shared_patch(pool, med_area, mips)
        ring_masks = None
    else:
        background = find_shared_patch(pool, med_area, mips)  # still a fallback record
        ring_masks = build_ring_masks(
            cell_mask=roi_mask, pool_mask=pool,
            pixel_size_um=img.pixel_size_um,
            inner_um=cfg.RING_INNER_UM, outer_um=cfg.RING_OUTER_UM,
        )

    # --- Measure ---
    rows = measure_cells(
        image_name=nd2_path.name,
        cell_mask=roi_mask,
        dapi_mip=dapi_mip,
        stacks_by_channel=stacks,
        mips_by_channel=mips,
        background=background,
        pixel_size_um=img.pixel_size_um,
        measure_channels=cfg.MEASURE_CHANNELS,
        bg_ring_masks=ring_masks,
    )
    apply_qc(rows, cfg)
    n_pass = sum(1 for r in rows if r.qc_pass)
    logger.info("  cells: %d segmented, %d passed QC", len(rows), n_pass)

    # --- Overlays + ROI zip ---
    (out_root / "overlays").mkdir(parents=True, exist_ok=True)
    save_overlay(
        out_path=out_root / "overlays" / f"{nd2_path.stem}.png",
        dapi_mip=dapi_mip,
        gfp_mip=mips.get("GFP", dapi_mip),
        cell_mask=roi_mask,
        rows=rows,
        background=background,
        dpi=cfg.OVERLAY_DPI,
    )
    (out_root / "rois").mkdir(parents=True, exist_ok=True)
    export_imagej_rois(
        out_zip=out_root / "rois" / f"{nd2_path.stem}.zip",
        cell_mask=roi_mask,
        rows=rows,
        background=background,
    )

    return {
        "image_name": nd2_path.name,
        "cells": rows,
        "pixel_um": img.pixel_size_um,
        "z_step_um": img.z_step_um,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path,
                    help="Folder of .nd2 files")
    ap.add_argument("--output", required=True, type=Path,
                    help="Output folder (will be created)")
    ap.add_argument("--config", type=Path, default=None,
                    help="YAML config; defaults to built-in Config()")
    ap.add_argument("--dry-run", action="store_true",
                    help="Process only the first .nd2 for parameter tuning")
    args = ap.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    _setup_logging(args.output / "run.log")

    cfg = load_config(args.config)
    _seed_everything(cfg.RANDOM_SEED)
    logger.info("Config: %s", cfg.as_dict())
    logger.info("Cellpose: %s (nuclei) / %s (cyto)",
                cellpose_model_info(cfg.CELLPOSE_NUCLEI_MODEL),
                cellpose_model_info(cfg.CELLPOSE_CYTO_MODEL))

    nd2s = sorted(args.input.glob("*.nd2"))
    if not nd2s:
        logger.error("No .nd2 files found under %s", args.input)
        return 2
    if args.dry_run:
        nd2s = nd2s[:1]
        logger.info("--dry-run: processing 1 file (%s)", nd2s[0].name)

    per_image_rows: Dict[str, List] = {}
    pixel_by_image: Dict[str, float] = {}
    zstep_by_image: Dict[str, float] = {}
    for p in nd2s:
        try:
            result = process_image(p, args.output, cfg, dry_run=args.dry_run)
        except Exception as e:
            logger.exception("FAILED on %s: %s", p.name, e)
            continue
        per_image_rows[result["image_name"]] = result["cells"]
        pixel_by_image[result["image_name"]] = result["pixel_um"]
        zstep_by_image[result["image_name"]] = result["z_step_um"]

    out_xlsx = args.output / ("measurements_dryrun.xlsx" if args.dry_run else "measurements.xlsx")
    write_workbook(out_xlsx, per_image_rows, pixel_by_image, zstep_by_image, cfg)
    logger.info("Workbook written: %s", out_xlsx)
    return 0


if __name__ == "__main__":
    sys.exit(main())

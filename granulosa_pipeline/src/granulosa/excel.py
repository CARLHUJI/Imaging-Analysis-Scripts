from __future__ import annotations

import datetime as _dt
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .config import Config
from .measurement import CellRow


BASE_COLS = [
    "image_name", "cell_id", "pixel_size_um", "z_step_um", "unit",
    "area_um2", "area_px", "centroid_x", "centroid_y",
    "solidity", "eccentricity", "focus_score",
    "z_start", "z_end",
    "qc_pass", "qc_reason",
]


def _cells_dataframe(
    rows_by_image: Dict[str, List[CellRow]],
    pixel_size_by_image: Dict[str, float],
    z_step_by_image: Dict[str, float],
    measure_channels: List[str],
) -> pd.DataFrame:
    # Group intensity columns per-channel per-source, ROI then background then corrected.
    channel_col_order: List[str] = []
    metric_suffixes = [
        "roi_mean", "roi_median", "roi_min", "roi_max",
        "roi_integrated", "roi_stdev",
        "bg_mean", "bg_median",
        "corrected_mean", "corrected_integrated", "signal_over_bg",
    ]
    for ch in measure_channels:
        for src in ("mip", "zspan"):
            for s in metric_suffixes:
                channel_col_order.append(f"{ch}_{src}_{s}")

    records = []
    for image_name, rows in rows_by_image.items():
        for r in rows:
            base = {
                "image_name": image_name,
                "cell_id": r.cell_id,
                "pixel_size_um": pixel_size_by_image.get(image_name, float("nan")),
                "z_step_um": z_step_by_image.get(image_name, float("nan")),
                "unit": "µm",
                "area_um2": r.area_um2,
                "area_px": r.area_px,
                "centroid_x": r.centroid_x,
                "centroid_y": r.centroid_y,
                "solidity": r.solidity,
                "eccentricity": r.eccentricity,
                "focus_score": r.focus_score,
                "z_start": r.z_start,
                "z_end": r.z_end,
                "qc_pass": r.qc_pass,
                "qc_reason": r.qc_reason,
            }
            base.update(r.intensities)
            records.append(base)
    df = pd.DataFrame.from_records(records)
    for col in channel_col_order:
        if col not in df.columns:
            df[col] = pd.NA
    return df[BASE_COLS + channel_col_order]


def _summary_dataframe(cells: pd.DataFrame, measure_channels: List[str]) -> pd.DataFrame:
    if cells.empty:
        return pd.DataFrame()
    per_image = []
    for image_name, sub in cells.groupby("image_name"):
        total = len(sub)
        passed = int(sub["qc_pass"].sum())
        rejected = total - passed
        reasons = sub.loc[~sub["qc_pass"], "qc_reason"].value_counts().to_dict()
        rec = {
            "image_name": image_name,
            "n_total": total,
            "n_passed": passed,
            "n_rejected": rejected,
            "rejection_reasons": ";".join(f"{k}:{v}" for k, v in reasons.items()),
        }
        passed_sub = sub[sub["qc_pass"]]
        for ch in measure_channels:
            for src in ("mip", "zspan"):
                col = f"{ch}_{src}_corrected_mean"
                if col in passed_sub.columns and passed_sub[col].notna().any():
                    rec[f"{col}_mean"] = float(passed_sub[col].mean())
                    rec[f"{col}_sd"] = float(passed_sub[col].std(ddof=0))
                    rec[f"{col}_n"] = int(passed_sub[col].notna().sum())
        per_image.append(rec)
    return pd.DataFrame(per_image)


def _params_dataframe(cfg: Config) -> pd.DataFrame:
    import platform
    rows = [{"key": k, "value": v} for k, v in asdict(cfg).items()]

    def _pkg_ver(mod: str) -> str:
        try:
            m = __import__(mod)
            return getattr(m, "__version__", "?")
        except Exception:
            return "not installed"

    for mod in ("numpy", "scipy", "pandas", "skimage", "matplotlib", "nd2",
                "cellpose", "roifile", "openpyxl", "yaml"):
        rows.append({"key": f"pkg::{mod}", "value": _pkg_ver(mod)})
    rows.append({"key": "python", "value": platform.python_version()})
    rows.append({"key": "platform", "value": platform.platform()})
    rows.append({"key": "run_timestamp", "value": _dt.datetime.now().isoformat(timespec="seconds")})
    return pd.DataFrame(rows)


def write_workbook(
    out_xlsx: Path,
    rows_by_image: Dict[str, List[CellRow]],
    pixel_size_by_image: Dict[str, float],
    z_step_by_image: Dict[str, float],
    cfg: Config,
) -> None:
    cells = _cells_dataframe(rows_by_image, pixel_size_by_image, z_step_by_image, cfg.MEASURE_CHANNELS)
    summary = _summary_dataframe(cells, cfg.MEASURE_CHANNELS)
    params = _params_dataframe(cfg)

    out_xlsx.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
        cells.to_excel(w, sheet_name="cells", index=False)
        summary.to_excel(w, sheet_name="summary", index=False)
        params.to_excel(w, sheet_name="params", index=False)

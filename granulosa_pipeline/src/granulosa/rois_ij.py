from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
from skimage.measure import find_contours

from .background import BackgroundPatch
from .measurement import CellRow


def export_imagej_rois(
    out_zip: Path,
    cell_mask: np.ndarray,
    rows: List[CellRow],
    background: BackgroundPatch,
) -> None:
    """Write a Fiji-compatible ROI .zip. Cells are polygon ROIs numbered
    "cell_XXX" ; the background patch is a rectangle ROI "background".
    """
    from roifile import ImagejRoi, ROI_TYPE, roiwrite

    rois: List[ImagejRoi] = []
    for r in rows:
        this = cell_mask == r.cell_id
        contours = find_contours(this.astype(float), 0.5)
        if not contours:
            continue
        # Use the longest contour for the cell body
        c = max(contours, key=len)
        # roifile takes (x, y) as float coords
        coords = np.column_stack([c[:, 1], c[:, 0]]).astype(np.float32)
        roi = ImagejRoi.frompoints(coords)
        roi.name = f"cell_{r.cell_id:03d}{'' if r.qc_pass else '_REJ'}"
        rois.append(roi)

    # background rectangle
    bg = ImagejRoi(
        roitype=ROI_TYPE.RECT,
        left=int(background.col),
        top=int(background.row),
        right=int(background.col + background.cols),
        bottom=int(background.row + background.rows),
    )
    bg.name = "background"
    rois.append(bg)

    if out_zip.exists():
        out_zip.unlink()
    roiwrite(str(out_zip), rois)

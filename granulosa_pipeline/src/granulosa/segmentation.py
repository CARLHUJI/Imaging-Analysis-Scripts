from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _diameter_px(diameter_um: Optional[float], pixel_size_um: float) -> Optional[float]:
    if diameter_um is None or diameter_um <= 0:
        return None
    return diameter_um / pixel_size_um


def cellpose_model_info(model_name: str) -> str:
    """Version tag: cellpose version + model name; logged in params sheet."""
    try:
        import cellpose
        return f"cellpose {cellpose.version} / {model_name}"
    except Exception:
        return f"cellpose ? / {model_name}"


def segment_nuclei(
    dapi_yx: np.ndarray,
    pixel_size_um: float,
    model_name: str = "nuclei",
    diameter_um: Optional[float] = None,
    flow_threshold: float = 0.4,
    cellprob_threshold: float = 0.0,
    gpu: bool = False,
    min_size: int = 15,
) -> np.ndarray:
    """Return label mask (int, 0 = background)."""
    from cellpose import models

    model = models.CellposeModel(gpu=gpu, model_type=model_name) if hasattr(models, "CellposeModel") \
        else models.Cellpose(gpu=gpu, model_type=model_name)
    diameter = _diameter_px(diameter_um, pixel_size_um)
    result = model.eval(
        dapi_yx,
        diameter=diameter,
        channels=[0, 0],
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        min_size=min_size,
    )
    masks = result[0]
    logger.info("nuclei segmentation -> %d objects", int(masks.max()))
    return masks.astype(np.int32)


def segment_cyto(
    dapi_yx: np.ndarray,
    gfp_yx: np.ndarray,
    pixel_size_um: float,
    model_name: str = "cyto3",
    diameter_um: Optional[float] = None,
    flow_threshold: float = 0.4,
    cellprob_threshold: float = 0.0,
    gpu: bool = False,
    min_size: int = 15,
) -> Tuple[np.ndarray, np.ndarray]:
    """Cellpose cyto model with GFP as cyto channel + DAPI as nucleus channel.

    Returns (cyto_masks, nuclear_masks). Uses model_type='cyto3' by default.
    Stacks channels as required by cellpose (H, W, 2): [gfp, dapi].
    """
    from cellpose import models

    model = models.CellposeModel(gpu=gpu, model_type=model_name) if hasattr(models, "CellposeModel") \
        else models.Cellpose(gpu=gpu, model_type=model_name)
    stacked = np.stack([gfp_yx, dapi_yx], axis=-1)
    diameter = _diameter_px(diameter_um, pixel_size_um)
    result = model.eval(
        stacked,
        diameter=diameter,
        channels=[1, 2],           # cyto=1st channel (GFP), nucleus=2nd (DAPI)
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        min_size=min_size,
    )
    cyto = result[0].astype(np.int32)

    # Nuclear masks: re-run nuclei model on DAPI only to get a clean nuclear labelling.
    nuclei = segment_nuclei(
        dapi_yx,
        pixel_size_um=pixel_size_um,
        model_name="nuclei",
        diameter_um=diameter_um,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        gpu=gpu,
        min_size=min_size,
    )
    logger.info("cyto segmentation -> %d cyto / %d nuclei", int(cyto.max()), int(nuclei.max()))
    return cyto, nuclei


def dilate_nuclear_to_whole_cell(nuclear_mask: np.ndarray, expand_um: float, pixel_size_um: float) -> np.ndarray:
    """Dilate a label mask by expand_um µm, keeping labels and preventing merges."""
    from skimage.segmentation import expand_labels
    radius_px = max(1, int(round(expand_um / pixel_size_um)))
    return expand_labels(nuclear_mask, distance=radius_px)

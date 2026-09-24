#!/usr/bin/env python3
"""
Convert Nikon .nd2 files to JPEG for representational images.

Usage:
    python3 convert_nd2_to_jpeg.py

Behavior:
    - Walks the folder where this script lives, recursively.
    - For each .nd2 file:
        * If the name contains "MaxIP" -> process it directly.
        * If it's a raw file AND a "<stem>-MaxIP.nd2" sibling exists in the same
          folder -> skip (the sibling gets processed instead).
        * If it's a raw file with no MaxIP sibling -> compute MaxIP in memory
          (max along the Z axis) then process. Nothing new is saved to disk
          besides the final JPEGs.
    - Mirrors the folder hierarchy into ./JPEG_export/ (inside the main folder).
    - Per source .nd2, output layout:
        JPEG_export/<same/subfolder/path>/<nd2_stem>/
            DAPI/    <stem>_oocyte01_DAPI.jpg, oocyte02_DAPI.jpg, ...
            GFP/     ...
            RFP/     ...
            PH/      ...  (phase / brightfield stays grayscale)
            merged/  <stem>_oocyte01_merged.jpg, ...  (RGB overlay of fluorescent channels)
    - Multi-position files are split: one JPEG per oocyte per channel.
    - Each fluorescent channel gets a pseudocolor LUT (DAPI=blue, GFP=green,
      RFP=red, Cy5/far-red=magenta). Phase/brightfield stays grayscale.
    - Uses percentile auto-contrast (0.1%-99.9%) so 16-bit data displays nicely.
"""

import gc
import os
import subprocess
import sys
from pathlib import Path

REQUIRED = {"nd2": "nd2", "numpy": "numpy", "PIL": "Pillow", "tqdm": "tqdm"}


def _missing_packages():
    missing = []
    for module, pip_name in REQUIRED.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(pip_name)
    return missing


VENV_DIR = Path.home() / ".nd2_convert_venv"


def ensure_environment():
    """
    Bootstrap a venv in the user's home directory (NOT on external volumes,
    where exFAT/non-Unix filesystems corrupt pip installs). Re-exec under
    the venv's Python. Works around PEP 668 (Homebrew/system Python).
    """
    missing = _missing_packages()
    if not missing:
        return

    venv_dir = VENV_DIR
    venv_python = venv_dir / "bin" / "python"

    # sys.prefix differs from sys.base_prefix only inside a venv.
    # Don't use .resolve() on sys.executable — venv pythons are symlinks
    # to the system python and would compare as equal after resolving.
    running_in_target_venv = (
        Path(sys.prefix) == venv_dir.resolve()
    )

    # Warn about a stale .venv sitting next to the script (likely corrupted on
    # external drives) and ignore it.
    legacy_venv = Path(__file__).resolve().parent / ".venv"
    if legacy_venv.exists():
        print(
            f"NOTE: ignoring stale venv at {legacy_venv}\n"
            f"      You can safely delete it: rm -rf \"{legacy_venv}\"\n"
        )

    if not venv_python.exists():
        print(f"Creating virtual environment at: {venv_dir}")
        subprocess.check_call([sys.executable, "-m", "venv", str(venv_dir)])

    # Always ensure all required packages are installed into the venv.
    # pip is idempotent — already-installed packages are skipped quickly.
    all_packages = list(REQUIRED.values())
    print(f"Ensuring packages in venv: {', '.join(all_packages)}")
    subprocess.check_call(
        [str(venv_python), "-m", "pip", "install", "--upgrade", "pip"]
    )
    subprocess.check_call(
        [str(venv_python), "-m", "pip", "install", *all_packages]
    )

    # Verify install actually landed (pip can silently skip on broken envs).
    check = subprocess.run(
        [str(venv_python), "-c", "import nd2, numpy, PIL, tqdm; print('ok')"],
        capture_output=True,
        text=True,
    )
    if check.returncode != 0 or "ok" not in check.stdout:
        print("\nERROR: packages did not install correctly into the venv.")
        print(f"  stdout: {check.stdout.strip()}")
        print(f"  stderr: {check.stderr.strip()}")
        print(f"\nTry removing the venv and retrying: rm -rf {venv_dir}")
        sys.exit(1)

    print("Install complete.\n")

    if not running_in_target_venv:
        print(f"Re-launching under venv Python: {venv_python}\n")
        os.execv(
            str(venv_python),
            [str(venv_python), str(Path(__file__).resolve()), *sys.argv[1:]],
        )


ensure_environment()

import numpy as np  # noqa: E402
import nd2  # noqa: E402
from PIL import Image  # noqa: E402
from tqdm import tqdm  # noqa: E402


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def to_uint8(arr, low_pct=0.1, high_pct=99.9):
    """Percentile-based auto-contrast, 16-bit -> 8-bit."""
    arr = arr.astype(np.float32)
    lo, hi = np.percentile(arr, [low_pct, high_pct])
    if hi <= lo:
        hi = lo + 1.0
    scaled = np.clip((arr - lo) / (hi - lo) * 255.0, 0, 255)
    return scaled.astype(np.uint8)


def sanitize(name):
    return (
        "".join(c if (c.isalnum() or c in "-_") else "_" for c in name).strip("_")
        or "channel"
    )


# Standard fluorescence LUT assignments. Keyed by lowercased fragments in the
# channel name; first match wins.
LUT_MAP = [
    (("dapi", "hoechst", "405"), "blue"),
    (("gfp", "fitc", "488", "alexa488", "a488", "egfp"), "green"),
    (("rfp", "mcherry", "tritc", "cy3", "561", "594", "a568", "texas"), "red"),
    (("cy5", "a647", "647", "640", "far"), "magenta"),
    (("ph", "phase", "bf", "bright", "dic", "trans", "tl"), "gray"),
]


def channel_to_color(name: str) -> str:
    n = name.lower()
    for keys, color in LUT_MAP:
        if any(k in n for k in keys):
            return color
    return "gray"


def apply_lut(gray_u8: np.ndarray, color: str) -> np.ndarray:
    """Return an RGB uint8 image applying the given pseudocolor LUT."""
    h, w = gray_u8.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    if color == "red":
        rgb[..., 0] = gray_u8
    elif color == "green":
        rgb[..., 1] = gray_u8
    elif color == "blue":
        rgb[..., 2] = gray_u8
    elif color == "magenta":
        rgb[..., 0] = gray_u8
        rgb[..., 2] = gray_u8
    elif color == "yellow":
        rgb[..., 0] = gray_u8
        rgb[..., 1] = gray_u8
    elif color == "cyan":
        rgb[..., 1] = gray_u8
        rgb[..., 2] = gray_u8
    else:  # gray
        rgb[..., 0] = gray_u8
        rgb[..., 1] = gray_u8
        rgb[..., 2] = gray_u8
    return rgb


def merge_channels(channel_rgbs: list[np.ndarray]) -> np.ndarray:
    """Additively merge a list of RGB uint8 images (clip at 255)."""
    if not channel_rgbs:
        return None
    acc = np.zeros_like(channel_rgbs[0], dtype=np.uint16)
    for rgb in channel_rgbs:
        acc += rgb.astype(np.uint16)
    return np.clip(acc, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# ND2 axis handling
# ---------------------------------------------------------------------------

def get_channel_names(f, n_channels):
    names = []
    try:
        for ch in f.metadata.channels:
            nm = None
            try:
                nm = ch.channel.name
            except Exception:
                pass
            if not nm:
                try:
                    lam = ch.channel.emissionLambdaNm
                    nm = f"C{lam:.0f}" if lam else None
                except Exception:
                    pass
            names.append(nm or f"C{len(names)}")
    except Exception:
        pass
    if len(names) < n_channels:
        names += [f"C{i}" for i in range(len(names), n_channels)]
    return names[:n_channels]


def iter_positions(f):
    """
    Generator that yields (position_index_1_based, channels_list, channel_names)
    ONE POSITION AT A TIME, holding at most one position's Z-stack in memory.

    Uses f.to_dask() for lazy loading — slicing the dask array to a single
    position triggers I/O only for that position (~hundreds of MB rather
    than tens of GB for a full raw file).

    channels_list is a list of 2D uint16 arrays (max-projected across T/Z),
    one per channel.
    """
    sizes = dict(f.sizes)
    axes = list(sizes.keys())
    n_channels = sizes.get("C", 1)
    channel_names = get_channel_names(f, n_channels)

    darr = f.to_dask()  # lazy view of the whole array

    pos_key = "P" if "P" in axes else ("M" if "M" in axes else None)
    n_positions = sizes.get(pos_key, 1) if pos_key else 1
    p_axis = axes.index(pos_key) if pos_key else None

    for p in range(n_positions):
        # Slice out just this position (drops the P axis).
        if p_axis is not None:
            slicer = [slice(None)] * darr.ndim
            slicer[p_axis] = p
            pos_lazy = darr[tuple(slicer)]
            remaining_axes = [ax for i, ax in enumerate(axes) if i != p_axis]
        else:
            pos_lazy = darr
            remaining_axes = axes[:]

        # Materialize this position only. Memory ~= T*Z*C*Y*X*2 bytes.
        arr = np.asarray(pos_lazy)
        del pos_lazy

        # Max-project across T and Z (this IS the MaxIP for raw files;
        # for pre-computed MaxIP files these axes are size 1 so it's a no-op).
        for extra in ("T", "Z"):
            if extra in remaining_axes:
                idx = remaining_axes.index(extra)
                if arr.shape[idx] > 1:
                    arr = arr.max(axis=idx)
                else:
                    arr = np.squeeze(arr, axis=idx)
                remaining_axes.pop(idx)

        # Should now be (C, Y, X) or (Y, X).
        if "C" not in remaining_axes:
            arr = arr[np.newaxis, ...]
            remaining_axes = ["C"] + remaining_axes
        else:
            c_idx = remaining_axes.index("C")
            if c_idx != 0:
                order = [c_idx] + [i for i in range(arr.ndim) if i != c_idx]
                arr = np.transpose(arr, order)
                remaining_axes = ["C"] + [
                    ax for i, ax in enumerate(remaining_axes) if i != c_idx
                ]

        n_ch = arr.shape[0]
        # .copy() so we can free `arr` and only hold the small per-channel views.
        channels = [arr[c].copy() for c in range(n_ch)]
        del arr

        yield p + 1, channels, channel_names


def count_positions(f):
    sizes = dict(f.sizes)
    return sizes.get("P", sizes.get("M", 1))


# ---------------------------------------------------------------------------
# Per-file processing
# ---------------------------------------------------------------------------

def process_file(nd2_path: Path, out_root: Path, source_root: Path, pbar=None, mode="maxip"):
    """
    Stream one position at a time so raw files (which can be 30-60 GB)
    don't blow past RAM. Returns:
        > 0  : number of JPEGs written
        0    : file failed to read (see error above bar)
        -1   : skipped because output already exists (resume)
    """
    rel_parent = nd2_path.parent.relative_to(source_root)
    stem = nd2_path.stem
    file_out_dir = out_root / rel_parent / stem

    def log(msg):
        if pbar is not None:
            pbar.write(msg)
        else:
            print(msg)

    # Resume: if the output folder already has JPEGs, assume it was finished
    # in a prior run. Delete the folder manually to force re-processing.
    if file_out_dir.exists():
        existing = any(file_out_dir.rglob("*.jpg"))
        if existing:
            return -1

    saved = 0
    channel_colors = None
    safe_channels = None
    channel_names = None

    try:
        with nd2.ND2File(str(nd2_path)) as f:
            n_pos = count_positions(f)
            pad = max(2, len(str(n_pos)))

            for p_idx, channels, ch_names in iter_positions(f):
                if channel_colors is None:
                    channel_names = ch_names
                    channel_colors = [channel_to_color(n) for n in channel_names]
                    safe_channels = [sanitize(n) for n in channel_names]

                oocyte_tag = f"oocyte{p_idx:0{pad}d}"

                pseudocolor_rgbs = []
                for img, safe_ch, ch_color in zip(channels, safe_channels, channel_colors):
                    gray8 = to_uint8(img)
                    rgb = apply_lut(gray8, ch_color)

                    ch_dir = file_out_dir / safe_ch
                    ch_dir.mkdir(parents=True, exist_ok=True)
                    jpg_path = ch_dir / f"{stem}_{oocyte_tag}_{safe_ch}.jpg"
                    Image.fromarray(rgb).save(jpg_path, quality=95)
                    saved += 1

                    if ch_color != "gray":
                        pseudocolor_rgbs.append(rgb)
                    del gray8

                if len(pseudocolor_rgbs) >= 2:
                    merged = merge_channels(pseudocolor_rgbs)
                    merged_dir = file_out_dir / "merged"
                    merged_dir.mkdir(parents=True, exist_ok=True)
                    merged_path = merged_dir / f"{stem}_{oocyte_tag}_merged.jpg"
                    Image.fromarray(merged).save(merged_path, quality=95)
                    saved += 1
                    del merged

                # Aggressively free the position's data before moving on.
                del channels, pseudocolor_rgbs
                gc.collect()

    except Exception as e:
        log(f"ERROR reading {nd2_path.relative_to(source_root)}: {e}")
        return 0

    return saved


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def discover_source_files(source_root: Path, out_root: Path):
    """
    Return a sorted list of (nd2_path, mode) tuples ready for processing.
        mode == "maxip"    -> file already contains "MaxIP" in its name.
        mode == "computed" -> raw file with no MaxIP sibling; do MaxIP in memory.
    Raw files that have a "<stem>-MaxIP.nd2" sibling in the same folder are
    excluded (the sibling will be processed instead).
    """
    all_nd2 = [
        p for p in source_root.rglob("*.nd2") if out_root not in p.parents
    ]

    maxip_files = []
    raw_files = []
    for p in all_nd2:
        if "maxip" in p.stem.lower():
            maxip_files.append(p)
        else:
            raw_files.append(p)

    # Index of MaxIP stems (lowercased) per folder for O(1) sibling lookup.
    maxip_stems_by_folder = {}
    for p in maxip_files:
        maxip_stems_by_folder.setdefault(p.parent, set()).add(p.stem.lower())

    result = [(p, "maxip") for p in maxip_files]

    for p in raw_files:
        expected_sibling = f"{p.stem}-maxip".lower()
        siblings = maxip_stems_by_folder.get(p.parent, set())
        if expected_sibling in siblings:
            continue  # a MaxIP sibling exists; skip this raw file.
        result.append((p, "computed"))

    result.sort(key=lambda x: (str(x[0].parent).lower(), x[0].name.lower()))
    return result


def main():
    source_root = Path(__file__).resolve().parent
    out_root = source_root / "JPEG_export"

    print(f"Source folder: {source_root}")
    print(f"Output folder: {out_root}\n")

    work_items = discover_source_files(source_root, out_root)
    if not work_items:
        print("No .nd2 files found to process.")
        return

    n_maxip = sum(1 for _, m in work_items if m == "maxip")
    n_computed = sum(1 for _, m in work_items if m == "computed")
    print(
        f"Found {len(work_items)} .nd2 file(s) to process: "
        f"{n_maxip} existing MaxIP, {n_computed} raw (MaxIP will be computed in memory).\n"
    )

    total_jpegs = 0
    failures = 0
    skipped = 0
    with tqdm(
        total=len(work_items),
        unit="file",
        desc="Converting",
        ncols=110,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}",
    ) as pbar:
        for nd2_path, mode in work_items:
            rel = nd2_path.relative_to(source_root)
            tag = "[MaxIP]" if mode == "maxip" else "[computing]"
            short = str(rel)
            if len(short) > 50:
                short = "..." + short[-47:]
            pbar.set_postfix_str(f"{tag} {short}", refresh=True)
            n = process_file(nd2_path, out_root, source_root, pbar=pbar, mode=mode)
            if n == -1:
                skipped += 1
            elif n == 0:
                failures += 1
            else:
                total_jpegs += n
            pbar.update(1)
            gc.collect()

    print(f"\nDone. {total_jpegs} JPEG(s) written from {len(work_items)} file(s).")
    if skipped:
        print(f"  {skipped} file(s) skipped (output already existed — delete the folder to redo).")
    if failures:
        print(f"  {failures} file(s) failed to read — see errors above.")


if __name__ == "__main__":
    main()

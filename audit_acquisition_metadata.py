#!/usr/bin/env python3
"""
Audit .nd2 acquisition metadata for consistency across samples of the same
antibody.

Usage:
    python3 audit_acquisition_metadata.py

Reads every RAW .nd2 file (skips *MaxIP*.nd2) under this folder recursively,
extracts per-channel acquisition metadata (exposure, binning, gain, objective,
excitation, pixel size, bit depth, z-step), infers the antibody from the
filename, and reports whether every sample of the same antibody was acquired
with identical settings.

Output:
    - Full report printed to terminal.
    - Same report saved as acquisition_audit.txt next to this script.
"""

import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

REQUIRED = {"nd2": "nd2", "numpy": "numpy", "tqdm": "tqdm"}

VENV_DIR = Path.home() / ".nd2_convert_venv"


def _missing_packages():
    missing = []
    for module, pip_name in REQUIRED.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(pip_name)
    return missing


def ensure_environment():
    missing = _missing_packages()
    if not missing:
        return

    venv_dir = VENV_DIR
    venv_python = venv_dir / "bin" / "python"
    running_in_target_venv = Path(sys.prefix) == venv_dir.resolve()

    if not venv_python.exists():
        print(f"Creating virtual environment at: {venv_dir}")
        subprocess.check_call([sys.executable, "-m", "venv", str(venv_dir)])

    all_packages = list(REQUIRED.values())
    print(f"Ensuring packages in venv: {', '.join(all_packages)}")
    subprocess.check_call(
        [str(venv_python), "-m", "pip", "install", "--upgrade", "pip"]
    )
    subprocess.check_call(
        [str(venv_python), "-m", "pip", "install", *all_packages]
    )

    if not running_in_target_venv:
        print(f"Re-launching under venv Python: {venv_python}\n")
        os.execv(
            str(venv_python),
            [str(venv_python), str(Path(__file__).resolve()), *sys.argv[1:]],
        )


ensure_environment()

import nd2  # noqa: E402
from tqdm import tqdm  # noqa: E402


# ---------------------------------------------------------------------------
# Antibody identification
# ---------------------------------------------------------------------------

# Substring patterns (no word boundaries — "_" is a word char, so \bFTO\b
# would not match "_FTO_"). Order matters: check longer/more specific first.
ANTIBODY_PATTERNS = [
    ("gH2A.X",    [r"y\.?h2a\.?x", r"gamma.*h2ax", r"gh2a\.?x", r"yh2ax"]),
    ("H3K9Me2",   [r"h3k9me2"]),
    ("L1-ORF",    [r"l1[\-_ ]?orf", r"l1orf"]),
    ("FTO",       [r"fto"]),  # last, since "fto" is short & could appear
]


def infer_antibody(filename_stem: str) -> str:
    s = filename_stem.lower()
    for name, patterns in ANTIBODY_PATTERNS:
        for pat in patterns:
            if re.search(pat, s):
                return name
    return "UNKNOWN"


# ---------------------------------------------------------------------------
# Metadata extraction
# ---------------------------------------------------------------------------

def _dig(d, *keys, default=None):
    """Safely walk a nested dict of unknown structure."""
    cur = d
    for k in keys:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur


def _search_key(obj, target_keys, seen=None):
    """
    Recursively search a nested dict/list structure for any of `target_keys`
    (case-insensitive substring match). Returns the first value found or None.
    """
    if seen is None:
        seen = set()
    if id(obj) in seen:
        return None
    seen.add(id(obj))

    if isinstance(obj, dict):
        for k, v in obj.items():
            lk = str(k).lower()
            for target in target_keys:
                if target in lk and not isinstance(v, (dict, list)):
                    return v
        for v in obj.values():
            r = _search_key(v, target_keys, seen)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = _search_key(v, target_keys, seen)
            if r is not None:
                return r
    return None


def _round_float(v, digits=3):
    try:
        return round(float(v), digits)
    except Exception:
        return v


def _get_sample_settings(blob):
    """
    Return the per-channel SampleSetting dict from Nikon .nd2 unstructured
    metadata. Path: ImageMetadataSeqLV|0.SLxPictureMetadata.PicturePlanes.SampleSetting
    Structure is {"0": {...}, "1": {...}, ...} — one entry per channel.
    Returns {} if not found.
    """
    seq = blob.get("ImageMetadataSeqLV|0", {})
    if isinstance(seq, dict):
        pic = seq.get("SLxPictureMetadata", {})
        if isinstance(pic, dict):
            planes = pic.get("PicturePlanes", {})
            if isinstance(planes, dict):
                ss = planes.get("SampleSetting", {})
                if isinstance(ss, dict):
                    return ss
    return {}


def _matching_led_power(device_setting, excitation_nm, tol_nm=15):
    """
    Given a per-channel DeviceSetting dict and a channel's excitation
    wavelength, find the LED wavelength within `tol_nm` and return its
    power (%). Returns None if not found.
    """
    if not isinstance(device_setting, dict) or excitation_nm is None:
        return None
    best = None
    best_dist = tol_nm + 1
    for k, v in device_setting.items():
        if not k.startswith("MultiLaser_PowerLineWavelength"):
            continue
        try:
            wl = float(v)
        except (TypeError, ValueError):
            continue
        dist = abs(wl - excitation_nm)
        if dist < best_dist:
            best_dist = dist
            idx = k.rsplit("-", 1)[-1]
            pw_key = f"MultiLaser_PowerLinePower0-{idx}"
            pw = device_setting.get(pw_key)
            if pw is not None:
                try:
                    best = float(pw)
                except (TypeError, ValueError):
                    pass
    return best


def extract_metadata(nd2_path: Path):
    """Read structured + unstructured metadata for one .nd2 file."""
    result = {"file": nd2_path, "error": None, "channels": []}
    try:
        with nd2.ND2File(str(nd2_path)) as f:
            sizes = dict(f.sizes)
            result["n_channels"] = sizes.get("C", 1)
            result["n_positions"] = sizes.get("P", sizes.get("M", 1))
            result["z_count"] = sizes.get("Z", 1)

            try:
                vs = f.voxel_size()
                result["pixel_size_um"] = _round_float(vs.x, 4)
                result["z_step_um"] = (
                    _round_float(vs.z, 4) if sizes.get("Z", 1) > 1 else None
                )
            except Exception:
                result["pixel_size_um"] = None
                result["z_step_um"] = None

            objective = None
            try:
                m = f.metadata.channels[0].microscope
                mag = getattr(m, "objectiveMagnification", None)
                na = getattr(m, "objectiveNumericalAperture", None)
                name = getattr(m, "objectiveName", None)
                parts = []
                if name:
                    parts.append(str(name))
                elif mag:
                    parts.append(f"{mag}x")
                if na:
                    parts.append(f"NA{na}")
                objective = " ".join(parts) if parts else None
            except Exception:
                pass
            result["objective"] = objective

            try:
                blob = f.unstructured_metadata()
            except Exception:
                blob = {}
            sample_settings = _get_sample_settings(blob)

            for i, ch in enumerate(f.metadata.channels):
                ex = getattr(ch.channel, "excitationLambdaNm", None)
                em = getattr(ch.channel, "emissionLambdaNm", None)
                cdata = {
                    "index": i,
                    "name": getattr(ch.channel, "name", f"C{i}") or f"C{i}",
                    "excitation_nm": _round_float(ex, 1) if ex else None,
                    "emission_nm": _round_float(em, 1) if em else None,
                    "bit_depth": getattr(
                        ch.volume, "bitsPerComponentSignificant", None
                    ),
                    "exposure_ms": None,
                    "binning": None,
                    "readout_speed": None,
                    "camera_offset": None,
                    "led_power_pct": None,
                    "camera_name": None,
                }

                ch_setting = sample_settings.get(str(i), {}) if sample_settings else {}
                cam = ch_setting.get("CameraSetting", {}) if isinstance(ch_setting, dict) else {}
                propq = cam.get("PropertiesQuality", {}) if isinstance(cam, dict) else {}
                fmt_q_desc = (
                    cam.get("FormatQuality", {}).get("Desc", {})
                    if isinstance(cam, dict)
                    else {}
                )

                exposure = propq.get("Exposure")
                if exposure is not None:
                    cdata["exposure_ms"] = _round_float(exposure, 2)

                bin_x = fmt_q_desc.get("BinningX")
                bin_y = fmt_q_desc.get("BinningY")
                if bin_x is not None and bin_y is not None and bin_x > 0 and bin_y > 0:
                    cdata["binning"] = f"{int(bin_x)}x{int(bin_y)}"

                cdata["readout_speed"] = propq.get("ReadoutSpeed")
                cdata["camera_offset"] = propq.get("CameraOffset")
                cam_name = propq.get("CameraName") or cam.get("CameraUserName")
                if cam_name:
                    cdata["camera_name"] = str(cam_name)

                device = ch_setting.get("DeviceSetting", {}) if isinstance(ch_setting, dict) else {}
                cdata["led_power_pct"] = _round_float(
                    _matching_led_power(device, ex), 2
                )

                result["channels"].append(cdata)

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"

    return result


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def discover_raw_files(source_root: Path, out_root: Path):
    """Return list of raw .nd2 files (no 'maxip' in name), excluding output dir."""
    all_nd2 = [
        p for p in source_root.rglob("*.nd2")
        if out_root not in p.parents
    ]
    raw = [p for p in all_nd2 if "maxip" not in p.stem.lower()]
    return sorted(raw, key=lambda p: (str(p.parent).lower(), p.name.lower()))


def find_maxip_only_samples(source_root: Path, out_root: Path):
    """Return MaxIP files that have no raw sibling — flagged as excluded from audit."""
    all_nd2 = [
        p for p in source_root.rglob("*.nd2")
        if out_root not in p.parents
    ]
    raw_stems_by_folder = defaultdict(set)
    for p in all_nd2:
        if "maxip" not in p.stem.lower():
            raw_stems_by_folder[p.parent].add(p.stem.lower())

    orphans = []
    for p in all_nd2:
        if "maxip" not in p.stem.lower():
            continue
        stem_lower = p.stem.lower()
        raw_expected = re.sub(r"-?maxip$", "", stem_lower).rstrip("-_ ")
        if raw_expected not in raw_stems_by_folder.get(p.parent, set()):
            orphans.append(p)
    return orphans


# ---------------------------------------------------------------------------
# Consistency check + reporting
# ---------------------------------------------------------------------------

def genotype_from_path(path: Path, source_root: Path) -> str:
    """Infer WT/Het/Homo from folder path (case-insensitive)."""
    rel = str(path.relative_to(source_root)).lower()
    if "wild type" in rel or "/wt/" in rel or "wildtype" in rel:
        return "WT"
    if "hetero" in rel:
        return "Het"
    if "homo" in rel:
        return "Homo"
    return "?"


def build_channel_key(cdata):
    """
    Key that identifies "which channel is this" across files so we can
    compare like with like. We prefer emission wavelength, then name.
    """
    if cdata.get("emission_nm"):
        return f"em{int(round(cdata['emission_nm']))}nm"
    name = (cdata.get("name") or "").strip()
    return name.upper() if name else f"C{cdata['index']}"


def summarize_group(antibody: str, files_data: list, source_root: Path):
    """
    Return a list of report lines for a single antibody group.
    """
    lines = []
    lines.append("=" * 76)
    lines.append(f"ANTIBODY: {antibody}   ({len(files_data)} file(s))")
    lines.append("=" * 76)

    # File-level parameters (objective, pixel size)
    objectives = defaultdict(list)
    pixel_sizes = defaultdict(list)
    z_steps = defaultdict(list)
    for fd in files_data:
        objectives[fd.get("objective")].append(fd["file"])
        pixel_sizes[fd.get("pixel_size_um")].append(fd["file"])
        if fd.get("z_step_um") is not None:
            z_steps[fd.get("z_step_um")].append(fd["file"])

    lines.extend(_summarize_field("Objective", objectives, source_root))
    lines.extend(_summarize_field("Pixel size (µm)", pixel_sizes, source_root))
    if z_steps:
        lines.extend(_summarize_field("Z step (µm)", z_steps, source_root))

    # Per-channel parameters
    channel_bucket = defaultdict(list)
    for fd in files_data:
        for ch in fd["channels"]:
            key = build_channel_key(ch)
            channel_bucket[key].append((fd["file"], ch))

    for ch_key in sorted(channel_bucket.keys()):
        entries = channel_bucket[ch_key]
        names_seen = sorted({ch["name"] for _, ch in entries if ch.get("name")})
        header = f"\n-- Channel: {ch_key}"
        if names_seen:
            header += f"  (names in files: {', '.join(names_seen)})"
        header += f"  [{len(entries)} file(s)]"
        lines.append(header)

        for field, label in [
            ("exposure_ms", "Exposure (ms)"),
            ("binning", "Binning"),
            ("led_power_pct", "LED power (%)"),
            ("readout_speed", "Readout speed"),
            ("camera_offset", "Camera offset"),
            ("camera_name", "Camera"),
            ("bit_depth", "Bit depth"),
        ]:
            groups = defaultdict(list)
            for f, ch in entries:
                groups[ch.get(field)].append(f)
            lines.extend(_summarize_field(f"  {label}", groups, source_root, indent="    "))

    lines.append("")
    return lines


def _summarize_field(label, groups, source_root: Path, indent=""):
    """
    Format a single field's consistency check.
    `groups` is a dict {value: [files that had that value]}.
    """
    lines = []
    # Drop None values from the "consistent" check but still report them.
    real_values = {v: fs for v, fs in groups.items() if v is not None and v != ""}
    none_files = groups.get(None, []) + groups.get("", [])

    if not real_values and not none_files:
        return lines

    if not real_values:
        lines.append(f"{indent}{label}: -- not available in any file")
        return lines

    if len(real_values) == 1 and not none_files:
        (only_val,) = real_values.keys()
        n = len(next(iter(real_values.values())))
        lines.append(f"{indent}{label}: OK  consistent = {only_val}  ({n} files)")
        return lines

    lines.append(f"{indent}{label}: MISMATCH")
    for val, files in sorted(real_values.items(), key=lambda x: -len(x[1])):
        lines.append(f"{indent}  {val!r}  in {len(files)} file(s):")
        for fp in files[:8]:
            lines.append(f"{indent}    - {fp.relative_to(source_root)}")
        if len(files) > 8:
            lines.append(f"{indent}    ...and {len(files) - 8} more")

    if none_files:
        lines.append(f"{indent}  (unavailable in {len(none_files)} file(s))")
        for fp in none_files[:5]:
            lines.append(f"{indent}    - {fp.relative_to(source_root)}")
        if len(none_files) > 5:
            lines.append(f"{indent}    ...and {len(none_files) - 5} more")
    return lines


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    source_root = Path(__file__).resolve().parent
    out_root = source_root / "JPEG_export"
    report_path = source_root / "acquisition_audit.txt"

    print(f"Source folder: {source_root}")
    print(f"Report file:   {report_path}\n")

    raw_files = discover_raw_files(source_root, out_root)
    orphans = find_maxip_only_samples(source_root, out_root)

    if not raw_files:
        print("No raw .nd2 files found.")
        return

    print(f"Found {len(raw_files)} raw .nd2 file(s) to audit.")
    if orphans:
        print(f"Note: {len(orphans)} MaxIP-only sample(s) will be skipped (no raw available).")
    print()

    files_data = []
    with tqdm(
        total=len(raw_files),
        unit="file",
        desc="Reading metadata",
        ncols=110,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}",
    ) as pbar:
        for p in raw_files:
            rel = p.relative_to(source_root)
            short = str(rel)
            if len(short) > 55:
                short = "..." + short[-52:]
            pbar.set_postfix_str(short, refresh=True)
            files_data.append(extract_metadata(p))
            pbar.update(1)

    # Report errors first
    errored = [d for d in files_data if d.get("error")]
    files_data = [d for d in files_data if not d.get("error")]

    # Group by antibody
    groups = defaultdict(list)
    unclassified = []
    for d in files_data:
        antibody = infer_antibody(d["file"].stem)
        if antibody == "UNKNOWN":
            unclassified.append(d)
        groups[antibody].append(d)

    # Build report
    report_lines = []
    report_lines.append("ND2 Acquisition Metadata Audit")
    report_lines.append(f"Source: {source_root}")
    report_lines.append(f"Raw files scanned: {len(files_data)}   Errors: {len(errored)}   MaxIP-only skipped: {len(orphans)}")
    report_lines.append("")

    if orphans:
        report_lines.append("MaxIP-only samples (excluded from audit, no raw file):")
        for p in orphans:
            report_lines.append(f"  - {p.relative_to(source_root)}")
        report_lines.append("")

    if errored:
        report_lines.append("Files that failed to read:")
        for d in errored:
            report_lines.append(f"  - {d['file'].relative_to(source_root)}  ({d['error']})")
        report_lines.append("")

    # Emit each antibody group
    for antibody in sorted(groups.keys(), key=lambda x: (x == "UNKNOWN", x)):
        report_lines.extend(summarize_group(antibody, groups[antibody], source_root))

    report_text = "\n".join(report_lines)
    print("\n" + report_text)
    report_path.write_text(report_text)
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()

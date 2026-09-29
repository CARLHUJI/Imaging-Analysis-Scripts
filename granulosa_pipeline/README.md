# Granulosa cell batch image-analysis pipeline

Processes a folder of Nikon `.nd2` z-stacks of dissociated granulosa cells and produces
per-cell measurements with paired background, MIP TIFFs, annotated overlays, and
Fiji-editable ROI zips.

## Install (one-time)

```bash
cd /Users/govindprakash/CARL_Data/Ob-Ob/granulosa_pipeline
conda env create -f environment.yml
conda activate granulosa
```

## Run

```bash
conda activate granulosa
python analyze.py \
    --input  "/path/to/nd2/folder" \
    --output "/path/to/output" \
    --config config.yaml

# Fast tuning on the first image only:
python analyze.py --input ... --output ... --config config.yaml --dry-run
```

Outputs:

```
output/
├── mips/           # {image}_{channel}.tif  (max-Z projection per channel)
├── overlays/       # {image}.png            (passed=green, rejected=red, BG=yellow)
├── rois/           # {image}.zip            (Fiji-editable: cell_NNN + background)
├── measurements.xlsx
└── run.log
```

### Excel schema

- **cells** — one row per cell. `image_name, cell_id, area_um2, centroid_x/y, solidity,
  eccentricity, focus_score, z_start/z_end, qc_pass, qc_reason`, followed by, for every
  measured channel, the group
  `{ch}_{mip|zspan}_roi_mean/median/min/max/integrated/stdev` +
  `{ch}_{mip|zspan}_bg_mean/median` +
  `{ch}_{mip|zspan}_corrected_mean/integrated/signal_over_bg`.
  MIP and z-span values sit side-by-side so you can inspect MIP bias.
- **summary** — per-image totals, rejection reasons, and mean/SD of corrected
  intensities.
- **params** — every config value, package versions, timestamp, seed.

### Configuration

Edit `config.yaml`. Key knobs:

- `SEG_CHANNEL`: `"dapi"` or `"dapi+gfp"`
- `ROI_MODE`: `"nuclear"` or `"whole_cell"`
- `BACKGROUND_MODE`: `"shared_patch"` (recommended) or `"ring"`
- `SPECK_PERCENTILE`: excludes bright non-cell GFP debris from the background pool
- `MIN_AREA_UM2` … `FOCUS_MIN`: QC filters. Rejected cells stay in the Excel with
  `qc_pass=False` and a `qc_reason` so nothing is silently dropped.

### Tests

```bash
conda activate granulosa
pytest tests -v
```

The synthetic-image tests verify ROI/background/corrected math and shared-patch
selection with analytically known ground truth.

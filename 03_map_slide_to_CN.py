#!/usr/bin/env python3
"""
Generate a JSON mapping of breast‑LCM samples to their copy‑number status (1q gain or no CN event) and
associate each sample with the available cropped histology image(s) plus key metadata.

The script expects three inputs:
  1. The copy‑number events spreadsheet (xlsx) – must contain columns:
        • Sample – unique sample ID, matching the filenames in the crop directory
        • Event  – e.g. "1q gain", "16q loss", … (may contain NaNs for no CN call)
  2. The master section‑details spreadsheet (xlsx) – must contain:
        • sampleID (unique sample ID, same as above)
        • Patient_Category, PD_ID, Slide_Description, Tissue_Type – the metadata to copy
  3. The root directory containing the cropped WEBP images (all sub‑folders will be searched
     recursively).

It produces a JSON file with the following hierarchical structure and saves it to --out:
{
  "1q": {
    "<sampleID>": {
      "paths": [<webp‑path>, …],
      "Patient_Category": …,
      "PD_ID": …,
      "Slide_Description": …,
      "Tissue_Type": …
    },
    …
  },
  "no CN": {
    "<sampleID>": { … },
    …
  }
}

Usage
-----
python 03_map_slide_to_CN.py \
    --cn Data/copy_number_events_GRCh38_211108.xlsx \
    --master Data/Breast_LCM_Section_Details_210323.xlsx \
    --slides Output/ndpi_crops \
    --out Data/sample_event_mapping.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────────────────────────────────────


def extract_sample_id(webp_path: Path) -> str:
    """Return the sample ID parsed from a crop file name.

    Examples
    --------
    PD37763c_lo0016_14.webp            → PD37763c_lo0016
    200818_B5_SECTION-16__25.webp      → 200818_B5_SECTION-16
    PD39911c_lo0006_3.webp             → PD39911c_lo0006
    """
    match = re.match(r'(.+?)(?:_{1,2}\d+)\.webp$', webp_path.name)
    if not match:
        raise ValueError(f"Unexpected crop filename format: {webp_path}")
    return match.group(1)


def build_crop_index(slide_root: Path) -> dict[str, list[str]]:
    """Return mapping {sampleID: [crop‑paths]} for all .webp files under *slide_root*."""
    crop_index: dict[str, list[str]] = {}
    for webp in slide_root.rglob('*.webp'):
        sample_id = extract_sample_id(webp)
        crop_index.setdefault(sample_id, []).append(str(webp))
    return crop_index


# ──────────────────────────────────────────────────────────────────────────────
# Main routine
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Generate JSON mapping of samples to CN status & crops',
    )
    parser.add_argument(
        '--cn', required=True,
        help='Copy‑number events XLSX path',
    )
    parser.add_argument(
        '--master', required=True,
        help='Master section details XLSX path',
    )
    parser.add_argument(
        '--slides', required=True, type=Path,
        help='Root directory of ndpi_crops',
    )
    parser.add_argument('--out', required=True, help='Output JSON path')
    args = parser.parse_args()

    # 1. Read spreadsheets
    cn_df = pd.read_excel(args.cn, engine='openpyxl')
    master_df = pd.read_excel(args.master, engine='openpyxl')

    # Normalise column names (strip whitespace / case)
    cn_df.columns = cn_df.columns.str.strip()
    master_df.columns = master_df.columns.str.strip()

    # 2. Determine sample → event status (keep only the first event per sample if duplicates exist)
    cn_events = cn_df.groupby('Sample', as_index=False).first()

    # Identify 1q‑gain samples
    samples_1q = set(
        cn_events.loc[
            cn_events['Event'].str.contains(
                '1q_gain', case=False, na=False,
            ), 'Sample',
        ],
    )  # type: ignore

    # Identify samples with *any* CN call (non‑null event)
    samples_with_cn_call = set(
        cn_events.loc[cn_events['Event'].notna(), 'Sample'],
    )  # type: ignore

    # 3. Build crop index
    crop_index = build_crop_index(args.slides)

    # 4. Build output structure
    output: dict[str, dict[str, dict]] = {'1q': {}, 'no CN': {}}

    metadata_cols = [
        'Patient_Category',
        'PD_ID',
        'Slide_Description',
        'Tissue_Type',
    ]

    # Loop over all samples present in the master spreadsheet
    for _, row in master_df.iterrows():
        sample_id = row['sampleID']
        # Fetch image paths (may be empty list if no crops yet)
        paths = crop_index.get(sample_id, [])

        md = {col: row.get(col, None) for col in metadata_cols}
        md['paths'] = paths

        # Decide event bucket
        if sample_id in samples_1q:
            bucket = '1q'
        elif sample_id not in samples_with_cn_call:
            bucket = 'no CN'
        else:
            # Has a CN event but not 1q – skip
            continue

        output[bucket][sample_id] = md

    # 5. Save JSON
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', encoding='utf-8') as fp:
        json.dump(output, fp, indent=2)
    print(
        f"✓ Wrote {out_path} with {sum(len(v) for v in output.values())} samples → {out_path.stat().st_size} bytes",
    )


if __name__ == '__main__':
    main()

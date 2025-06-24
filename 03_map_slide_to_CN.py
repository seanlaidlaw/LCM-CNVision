#!/usr/bin/env python3
"""
Generate four JSON mappings of breast‑LCM samples with binary classifications,
associating each sample with available cropped histology image(s) plus key metadata.

The script expects three inputs:
  1. The copy‑number events spreadsheet (xlsx) – must contain columns:
        • Sample – unique sample ID, matching the filenames in the crop directory
        • Event  – e.g. "1q gain", "16q loss", … (may contain NaNs for no CN call)
        • Tumour – "Y" for tumor samples, "N" for normal samples
  2. The master section‑details spreadsheet (xlsx) – must contain:
        • sampleID (unique sample ID, same as above)
        • Patient_Category, PD_ID, Slide_Description, Tissue_Type – the metadata to copy
  3. The root directory containing the cropped WEBP images (all sub‑folders will be searched
     recursively).

It produces four JSON files with binary classifications:
- tumour_vs_normal.json: tumour vs normal classification
- normal_CN_vs_noCN.json: normal samples with any CN event vs normal samples with no CN
- normal_1q_vs_noCN.json: normal samples with 1q gain vs normal samples with no CN
- normal_1q_or_16q_loss_vs_noCN.json: normal samples with 1q gain or 16q loss vs normal samples with no CN

Each JSON has the structure:
{
  "tumour": {
    "<sampleID>": {
      "paths": [<webp‑path>, …],
      "Patient_Category": …,
      "PD_ID": …,
      "Slide_Description": …,
      "Tissue_Type": …
    },
    …
  },
  "normal": {
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
    --out_dir Data/
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


def save_json(data: dict, filepath: Path, description: str):
    """Save data to JSON file and print summary."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with filepath.open('w', encoding='utf-8') as fp:
        json.dump(data, fp, indent=2)

    # Count samples in each category
    counts = {k: len(v) for k, v in data.items()}
    total = sum(counts.values())
    print(
        f"✓ Wrote {description}: {filepath} with {total} total samples → {filepath.stat().st_size} bytes",
    )
    for category, count in counts.items():
        print(f"  - {category}: {count} samples")


# ──────────────────────────────────────────────────────────────────────────────
# Main routine
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Generate four JSON mappings with binary classifications: tumour vs normal, normal CN vs no CN, normal 1q vs no CN, normal 1q or 16q loss vs no CN',
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
    parser.add_argument(
        '--out_dir', required=True, type=Path,
        help='Output directory for JSON files',
    )
    args = parser.parse_args()

    # 1. Read spreadsheets
    cn_df = pd.read_excel(args.cn, engine='openpyxl')
    master_df = pd.read_excel(args.master, engine='openpyxl')

    # Normalise column names (strip whitespace / case)
    cn_df.columns = cn_df.columns.str.strip()
    master_df.columns = master_df.columns.str.strip()

    # 2. Determine sample classifications (keep only the first event per sample if duplicates exist)
    cn_events = cn_df.groupby('Sample', as_index=False).first()

    # Create sample classification mappings
    samples_tumour = set(
        cn_events.loc[cn_events['Tumour'] == 'Y', 'Sample'],
    )

    samples_normal = set(
        cn_events.loc[cn_events['Tumour'] == 'N', 'Sample'],
    )

    samples_with_cn_event = set(
        cn_events.loc[cn_events['Event'].notna(), 'Sample'],
    )

    samples_1q = set(
        cn_events.loc[
            cn_events['Event'].str.contains(
                '1q_gain', case=False, na=False,
            ), 'Sample',
        ],
    )

    samples_16q_loss = set(
        cn_events.loc[
            cn_events['Event'].str.contains(
                '16q_loss', case=False, na=False,
            ), 'Sample',
        ],
    )

    samples_either_1q_or_16q_loss = samples_1q.union(samples_16q_loss)

    # All samples in events spreadsheet
    samples_in_events = set(cn_events['Sample'])

    # 3. Build crop index
    crop_index = build_crop_index(args.slides)

    # 4. Initialize output structures
    tumour_vs_normal = {'tumour': {}, 'normal': {}}
    normal_cn_vs_nocn = {'normal_with_CN': {}, 'normal_no_CN': {}}
    normal_1q_vs_nocn = {'normal_1q': {}, 'normal_no_CN': {}}
    normal_1q_or_16q_loss_vs_nocn = {
        'normal_1q_or_16q_loss': {}, 'normal_no_CN': {},
    }

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

        # Classification 1: tumour vs normal
        if sample_id in samples_tumour:
            tumour_vs_normal['tumour'][sample_id] = md
        elif sample_id in samples_normal or sample_id not in samples_in_events:
            tumour_vs_normal['normal'][sample_id] = md

            # Classification 2: normal CN vs no CN
            if sample_id in samples_with_cn_event:
                normal_cn_vs_nocn['normal_with_CN'][sample_id] = md
            else:
                normal_cn_vs_nocn['normal_no_CN'][sample_id] = md

            # Classification 3: normal 1q vs no CN
            if sample_id in samples_1q:
                normal_1q_vs_nocn['normal_1q'][sample_id] = md
            elif sample_id not in samples_with_cn_event:
                normal_1q_vs_nocn['normal_no_CN'][sample_id] = md

            # Classification 4: normal 1q or 16q loss vs no CN
            if sample_id in samples_either_1q_or_16q_loss:
                normal_1q_or_16q_loss_vs_nocn['normal_1q_or_16q_loss'][sample_id] = md
            elif sample_id not in samples_with_cn_event:
                normal_1q_or_16q_loss_vs_nocn['normal_no_CN'][sample_id] = md

    # 5. Save all JSON files
    args.out_dir.mkdir(parents=True, exist_ok=True)

    save_json(
        tumour_vs_normal, args.out_dir /
        'tumour_vs_normal.json', 'tumour vs normal classification',
    )
    save_json(
        normal_cn_vs_nocn, args.out_dir /
        'normal_CN_vs_noCN.json', 'normal CN vs no CN classification',
    )
    save_json(
        normal_1q_vs_nocn, args.out_dir /
        'normal_1q_vs_noCN.json', 'normal 1q vs no CN classification',
    )
    save_json(
        normal_1q_or_16q_loss_vs_nocn, args.out_dir /
        'normal_1q_or_16q_loss_vs_noCN.json', 'normal 1q or 16q loss vs no CN classification',
    )

    # Print overall summary
    print(f"\nOverall Summary:")
    print(
        f"  Total samples processed: {len(tumour_vs_normal['tumour']) + len(tumour_vs_normal['normal'])}",
    )
    print(f"  Tumour samples: {len(tumour_vs_normal['tumour'])}")
    print(f"  Normal samples: {len(tumour_vs_normal['normal'])}")
    print(f"  Normal with any CN: {len(normal_cn_vs_nocn['normal_with_CN'])}")
    print(f"  Normal with 1q: {len(normal_1q_vs_nocn['normal_1q'])}")
    print(
        f"  Normal with 1q or 16q loss: {len(normal_1q_or_16q_loss_vs_nocn['normal_1q_or_16q_loss'])}",
    )


if __name__ == '__main__':
    main()

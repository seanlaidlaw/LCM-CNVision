from __future__ import annotations

import glob
import json
import os
import re

base_dir = '/Volumes/casm-lcm/tb14/Breast/'
output_json = 'ndpi_ndpa_pairs.json'

ndpi_candidates = []

# Step 1: Find all valid .ndpi paths
for root, _, files in os.walk(base_dir):
    if 'slide' not in root.lower() or 'PD' not in root:
        continue
    for fname in files:
        if fname.endswith('.ndpi') and not fname.startswith('._'):
            ndpi_candidates.append(os.path.join(root, fname))

# Pattern to strip .edited and date
edited_cleanup = re.compile(r'(_\d+)?\.edited(\.\d+)?')

matched_pairs = {}

for ndpi_path in ndpi_candidates:
    ndpi_dir = os.path.dirname(ndpi_path)
    ndpi_fname = os.path.basename(ndpi_path)

    # Remove any edited-related suffix from filename
    ndpi_stub = edited_cleanup.sub('', ndpi_fname).replace('.ndpi', '')
    ndpa_glob = os.path.join(ndpi_dir, f"{ndpi_stub}*.ndpa")
    ndpa_matches = glob.glob(ndpa_glob)

    # Filter for real files with size > 0
    valid_ndpas = [
        f
        for f in ndpa_matches
        if (
            not os.path.basename(f).startswith('._')
            and os.path.isfile(f)
            and os.path.getsize(f) > 0
        )
    ]

    if not valid_ndpas:
        print(f"[WARN] No valid NDPA found for: {ndpi_path}")
        continue

    # Pick latest modified
    latest_ndpa = max(valid_ndpas, key=os.path.getmtime)
    matched_pairs[ndpi_path] = latest_ndpa

# Step 2: Save as JSON
with open(output_json, 'w') as f:
    json.dump(matched_pairs, f, indent=2)

print(f"✅ Saved {len(matched_pairs)} pairs to {output_json}")

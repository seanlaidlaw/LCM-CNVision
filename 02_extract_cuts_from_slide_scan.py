#!/usr/bin/env python3
"""
Batch‑crop Hamamatsu NDPI slides using NDPA polygon annotations → 80 % WebP.

Usage
-----
Single slide (legacy mode):

    python 02_extract_cuts_from_slide_scan.py slide.ndpi annots.ndpa --out out_dir [options]

Batch mode with JSON mapping:

    python 02_extract_cuts_from_slide_scan.py --pairs ndpi_ndpa_pairs.json --out out_root [options]

The JSON should be a dict where **keys are NDPI paths** and **values are NDPA
paths**.

Options
~~~~~~~
--nmpp N        Override nm / px (e.g. 442)
--rect          Keep rectangle instead of square crops
--verbose       Print debug info
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

from openslide import OpenSlide
from openslide import OpenSlideError
from openslide import OpenSlideUnsupportedFormatError
from PIL import Image

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _load_xml(path: str) -> ET.Element:
    with open(path, 'rb') as fh:
        raw = fh.read()
    bom = b'\xef\xbb\xbf'
    if raw.startswith(bom):
        raw = raw[len(bom):]
    lt = raw.find(b'<')
    if lt > 0:
        raw = raw[lt:]
    return ET.fromstring(raw)


def _mpp(slide: OpenSlide) -> tuple[float, float]:
    props = slide.properties
    mpp_x = props.get(
        'openslide.mpp-x',
    ) or props.get('aperio.MPP') or props.get('hamamatsu.MPP')
    mpp_y = props.get(
        'openslide.mpp-y',
    ) or props.get('aperio.MPP') or props.get('hamamatsu.MPP')
    if mpp_x is None or mpp_y is None:
        raise RuntimeError('Pixel size missing; supply --nmpp')
    return float(mpp_x), float(mpp_y)


def _header_offset(slide: OpenSlide) -> tuple[float, float]:
    p = slide.properties
    xoff = float(p.get('hamamatsu.XOffsetFromSlideCentre', 0))
    yoff = float(p.get('hamamatsu.YOffsetFromSlideCentre', 0))
    return xoff, yoff

# Conversion builder -----------------------------------------------------------


def _build_nm_to_px(sw_px: int, sh_px: int, nmpp: float, xoff_nm: float, yoff_nm: float) -> Callable[[float, float], tuple[int, int]]:
    half_w_nm = sw_px * nmpp / 2.0
    half_h_nm = sh_px * nmpp / 2.0
    origin_x_nm = -half_w_nm + xoff_nm
    origin_y_nm = -half_h_nm + yoff_nm

    def nm2px(x_nm: float, y_nm: float) -> tuple[int, int]:
        return int(round((x_nm - origin_x_nm) / nmpp)), int(round((y_nm - origin_y_nm) / nmpp))

    return nm2px

# NDPA parsing -----------------------------------------------------------------


def _parse_ndpa_points(root: ET.Element, nm2px: Callable[[float, float], tuple[int, int]], verbose=False) -> list[dict]:
    anns: list[dict] = []
    for s_idx, state in enumerate(root.findall('.//ndpviewstate')):
        title = state.findtext('title', '')
        m = re.search(r'(PD[^_]+_lo\d+)', title)
        pd_id = m.group(1) if m else title.replace(':', '_') or f'state{s_idx}'
        for ann in state.findall('.//annotation[@type="freehand"]'):
            pts = []
            for pt in ann.findall('.//pointlist/point'):
                xt, yt = pt.find('x'), pt.find('y')
                if xt is None or yt is None:
                    continue
                try:
                    px, py = nm2px(float(xt.text), float(yt.text))
                except (TypeError, ValueError):
                    continue
                pts.append((px, py))
            if pts:
                if verbose and len(anns) == 0:
                    print(
                        '[DEBUG] first 3 px:', ', '.join(
                            f'{p[0]},{p[1]}' for p in pts[:3]
                        ),
                    )
                anns.append({'pd_id': pd_id, 'points': pts})
    return anns

# Cropping ---------------------------------------------------------------------


def _crop_slide(slide_path: str, anns: list[dict], out_dir: str, square: bool, verbose=False):
    slide = OpenSlide(slide_path)
    sw, sh = slide.dimensions
    skipped_count = 0
    processed_count = 0

    for idx, ann in enumerate(anns, 1):
        fname = f"{ann['pd_id']}_{idx}.webp"
        output_path = os.path.join(out_dir, fname)

        # Skip if file already exists
        if os.path.exists(output_path):
            if verbose:
                print('[SKIP]', fname, 'already exists')
            skipped_count += 1
            continue

        xs = [p[0] for p in ann['points']]
        ys = [p[1] for p in ann['points']]
        minx, maxx = min(xs), max(xs)
        miny, maxy = min(ys), max(ys)
        if square:
            side = max(maxx - minx, maxy - miny)
            w = h = side
        else:
            w, h = maxx - minx, maxy - miny
        x0, y0 = minx, miny
        if x0 + w > sw:
            x0 = max(0, sw - w)
        if y0 + h > sh:
            y0 = max(0, sh - h)
        if w <= 0 or h <= 0:
            if verbose:
                print('[SKIP]', ann['pd_id'], 'empty bbox')
            continue
        try:
            region = slide.read_region((x0, y0), 0, (w, h))
        except OpenSlideError as e:
            print('[ERR]', ann['pd_id'], e)
            continue
        if region.mode == 'RGBA':
            region = region.convert('RGB')
        region.save(output_path, format='WEBP', quality=80, method=6)
        processed_count += 1
        if verbose:
            print('[OK]', fname)

    slide.close()
    if verbose and (skipped_count > 0 or processed_count > 0):
        print(
            f'[SUMMARY] {processed_count} processed, {skipped_count} skipped',
        )

# Driver -----------------------------------------------------------------------


def _process_pair(slide_path: str, ndpa_path: str, out_root: str, nmpp_override: float | None, rect: bool, verbose: bool):
    if not (os.path.exists(slide_path) and os.path.exists(ndpa_path)):
        print('[WARN] missing file, skipping:', slide_path)
        return

    out_dir = os.path.join(
        out_root, os.path.splitext(
            os.path.basename(slide_path),
        )[0],
    )

    # Check if output directory already exists and contains files
    if os.path.exists(out_dir) and os.listdir(out_dir):
        if verbose:
            print(
                f'[SKIP] Output directory already exists and contains files: {out_dir}',
            )
        return

    root = _load_xml(ndpa_path)
    slide = OpenSlide(slide_path)
    sw, sh = slide.dimensions
    nmpp = nmpp_override if nmpp_override else _mpp(slide)[0] * 1000
    xoff_nm, yoff_nm = _header_offset(slide)
    nm2px = _build_nm_to_px(sw, sh, nmpp, xoff_nm, yoff_nm)
    anns = _parse_ndpa_points(root, nm2px, verbose=verbose)
    if not anns:
        print('[WARN] no annotations in', ndpa_path)
        slide.close()
        return
    os.makedirs(out_dir, exist_ok=True)
    _crop_slide(slide_path, anns, out_dir, square=not rect, verbose=verbose)
    slide.close()

# CLI -------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser('Batch crop NDPI+NDPA to WebP')
    ap.add_argument('--pairs', help='JSON mapping of NDPI→NDPA')
    ap.add_argument('paths', nargs='*', help='(legacy) slide.ndpi annots.ndpa')
    ap.add_argument('--out', required=True, help='Output root directory')
    ap.add_argument('--nmpp', type=float, help='nm per pixel override')
    ap.add_argument('--rect', action='store_true')
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    if args.pairs:
        with open(args.pairs) as fh:
            mapping = json.load(fh)
        for slide_path, ndpa_path in mapping.items():
            _process_pair(
                slide_path, ndpa_path, args.out,
                args.nmpp, args.rect, args.verbose,
            )
    else:
        if len(args.paths) < 2:
            sys.exit('Need slide and ndpa paths or --pairs JSON')
        slide_path, ndpa_path = args.paths[0], args.paths[1]
        _process_pair(
            slide_path, ndpa_path, args.out,
            args.nmpp, args.rect, args.verbose,
        )


if __name__ == '__main__':
    main()

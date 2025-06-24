#!/usr/bin/env python3
"""
Fine‑tune the **UNI** ViT‑L/16 foundation model for binary classification of
breast‑LCM crops using any of the binary classification JSON files.

This script can train binary classifiers for:
- tumour vs normal classification
- normal samples with CN events vs normal samples with no CN
- normal samples with 1q gain vs normal samples with no CN
- normal samples with 1q or 16q loss vs normal samples with no CN

This patch addresses
────────────────────
1. **torchmetrics ≥1.0 API change** – `accuracy()` and `f1_score()` now require
   a `task` argument. We pass `task="binary"` for our two‑class setting.
2. **Apple Silicon / MPS** – selects `mps` when available and only enables
   `pin_memory=True` when the device is CUDA (it is unsupported on MPS & CPU).
3. **Loader warning silenced** – the pin‑memory logic avoids the earlier
   runtime warning.
4. **Early stopping** – stops training when validation accuracy doesn't improve
   for a specified number of epochs to prevent overfitting.
5. **Dynamic classification** – automatically detects the two categories from the JSON file
   and creates appropriate label mappings.
6. **Stratified split** – ensures each class has at least 3 samples in validation and test sets.

Run examples
───────────
```bash
export HF_TOKEN=hf_********************************
# Train tumour vs normal classifier
python 04_train_UNI_on_CN.py \
  --json Output/tumour_vs_normal.json \
  --root Output/ndpi_crops \
  --out_dir runs/uni_tumour_vs_normal \
  --epochs 20 --batch 24 --patience 3

# Train normal CN vs no CN classifier
python 04_train_UNI_on_CN.py \
  --json Output/normal_CN_vs_noCN.json \
  --root Output/ndpi_crops \
  --out_dir runs/uni_normal_CN_vs_noCN \
  --epochs 20 --batch 24 --patience 3

# Train normal 1q or 16q loss vs no CN classifier
python 04_train_UNI_on_CN.py \
  --json Output/normal_1q_or_16q_loss_vs_noCN.json \
  --root Output/ndpi_crops \
  --out_dir runs/uni_normal_1q_or_16q_loss_vs_noCN \
  --epochs 20 --batch 24 --patience 3

# Train normal 1q vs no CN classifier
python 04_train_UNI_on_CN.py \
  --json Output/normal_1q_vs_noCN.json \
  --root Output/ndpi_crops \
  --out_dir runs/uni_normal_1q_vs_noCN \
  --epochs 20 --batch 24 --patience 3
```
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path
from typing import List
from typing import Tuple

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from huggingface_hub import login
from PIL import Image
from sklearn.model_selection import train_test_split
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import WeightedRandomSampler
from torchmetrics.functional import accuracy as tm_accuracy
from torchmetrics.functional import f1_score as tm_f1
# torchmetrics >=1.0 unified functional interface

# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────


class HistologyCrops(Dataset):
    """Lazy‑load WEBP crops referenced in the JSON mapping."""

    def __init__(self, items: list[tuple[Path, int]], transform):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, label = self.items[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)


# ──────────────────────────────────────────────────────────────────────────────
# UNI backbone
# ──────────────────────────────────────────────────────────────────────────────

def load_uni_backbone(hf_token: str | None = None, freeze: bool = False):
    if hf_token:
        login(token=hf_token, add_to_git_credential=False)
    encoder = timm.create_model(
        'hf-hub:MahmoodLab/UNI',
        pretrained=True,
        init_values=1e-5,
        dynamic_img_size=True,
        num_classes=0,
    )
    if freeze:
        for p in encoder.parameters():
            p.requires_grad = False
    encoder.out_features = encoder.num_features

    cfg = resolve_data_config(encoder.pretrained_cfg, model=encoder)
    train_tf = create_transform(**cfg, is_training=True)
    eval_tf = create_transform(**cfg, is_training=False)
    return encoder, train_tf, eval_tf


class UniClassifier(nn.Module):
    def __init__(self, encoder: nn.Module, n_classes: int = 2):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.Linear(encoder.out_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        feat = self.encoder(x)
        if isinstance(feat, (tuple, list)):
            feat = feat[0]
        return self.head(feat)


# ──────────────────────────────────────────────────────────────────────────────
# Early Stopping
# ──────────────────────────────────────────────────────────────────────────────

class EarlyStopping:
    """Early stopping to prevent overfitting."""

    def __init__(self, patience: int = 3, min_delta: float = 0.0, restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.best_weights = None

    def __call__(self, val_score: float, model: nn.Module) -> bool:
        """
        Returns True if training should stop.

        Args:
            val_score: Current validation score (higher is better)
            model: The model to save weights from if this is the best score

        Returns:
            True if training should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = val_score
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
            return False

        if val_score > self.best_score + self.min_delta:
            self.best_score = val_score
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1

        return self.counter >= self.patience

    def restore_best_model(self, model: nn.Module):
        """Restore the model to its best weights."""
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def stratified_split(items, val_size=0.15, test_size=0.15, min_samples_per_class=3):
    """
    Create stratified train/val/test splits ensuring each class has at least min_samples_per_class
    in validation and test sets.

    Args:
        items: List of (path, label) tuples
        val_size: Fraction of data for validation
        test_size: Fraction of data for test
        min_samples_per_class: Minimum samples per class in val and test sets
    """
    paths, labels = zip(*items)
    labels = np.array(labels)

    # Count samples per class
    unique_labels, counts = np.unique(labels, return_counts=True)
    print(
        f"Class distribution: {dict(zip(unique_labels.astype(int), counts.astype(int)))}",
    )

    # Check if we have enough samples for the minimum requirement
    min_required = min_samples_per_class * 2  # for val + test
    for label, count in zip(unique_labels, counts):
        if count < min_required:
            print(
                f"[warn] Class {int(label)} has only {int(count)} samples, minimum required is {min_required}",
            )
            print(f"[warn] Reducing min_samples_per_class to {count // 2}")
            min_samples_per_class = max(1, count // 2)

    # Adjust split sizes if needed to ensure minimum samples per class
    total_samples = len(labels)
    min_val_test_samples = min_samples_per_class * \
        len(unique_labels) * 2  # val + test
    min_val_test_fraction = min_val_test_samples / total_samples

    if val_size + test_size < min_val_test_fraction:
        print(
            f"[warn] Requested val+test size ({val_size + test_size:.3f}) too small for minimum samples requirement",
        )
        print(f"[warn] Adjusting to {min_val_test_fraction:.3f}")
        val_size = min_val_test_fraction / 2
        test_size = min_val_test_fraction / 2

    max_attempts = 1000  # Increase attempts for imbalanced datasets
    for attempt in range(max_attempts):
        try:
            # First split: train vs (val+test)
            train_paths, tmp_paths, train_labels, tmp_labels = train_test_split(
                paths,
                labels,
                test_size=val_size + test_size,
                stratify=labels,
                random_state=SEED + attempt,
            )

            # Second split: val vs test
            rel_val = val_size / (val_size + test_size)
            val_paths, test_paths, val_labels, test_labels = train_test_split(
                tmp_paths,
                tmp_labels,
                test_size=1 - rel_val,
                stratify=tmp_labels,
                random_state=SEED + attempt,
            )

            # Check if both classes are present in val and test with minimum samples
            val_unique, val_counts = np.unique(val_labels, return_counts=True)
            test_unique, test_counts = np.unique(
                test_labels, return_counts=True,
            )

            val_has_min = all(
                count >= min_samples_per_class for count in val_counts
            )
            test_has_min = all(
                count >= min_samples_per_class for count in test_counts
            )

            if val_has_min and test_has_min:
                def pack(pths, lbls): return list(zip(pths, lbls))
                result = pack(train_paths, train_labels), pack(
                    val_paths, val_labels,
                ), pack(test_paths, test_labels)

                # Print split statistics with clean formatting
                train_labels_array = np.array(train_labels)
                val_labels_array = np.array(val_labels)
                test_labels_array = np.array(test_labels)

                train_dist = dict(
                    zip(*np.unique(train_labels_array, return_counts=True)),
                )
                val_dist = dict(
                    zip(*np.unique(val_labels_array, return_counts=True)),
                )
                test_dist = dict(
                    zip(*np.unique(test_labels_array, return_counts=True)),
                )

                print(f"Split successful after {attempt + 1} attempts:")
                print(
                    f"  Train: {len(train_labels)} samples, classes: {dict(zip([int(k) for k in train_dist.keys()], [int(v) for v in train_dist.values()]))}",
                )
                print(
                    f"  Val:   {len(val_labels)} samples, classes: {dict(zip([int(k) for k in val_dist.keys()], [int(v) for v in val_dist.values()]))}",
                )
                print(
                    f"  Test:  {len(test_labels)} samples, classes: {dict(zip([int(k) for k in test_dist.keys()], [int(v) for v in test_dist.values()]))}",
                )

                # Validate minimum requirements are met
                for label in unique_labels:
                    val_count = val_dist.get(label, 0)
                    test_count = test_dist.get(label, 0)
                    if val_count < min_samples_per_class:
                        raise ValueError(
                            f"Class {int(label)} has only {int(val_count)} samples in validation set, minimum required is {min_samples_per_class}",
                        )
                    if test_count < min_samples_per_class:
                        raise ValueError(
                            f"Class {int(test_count)} has only {int(test_count)} samples in test set, minimum required is {min_samples_per_class}",
                        )

                print(
                    f"✓ Validation passed: All classes have at least {min_samples_per_class} samples in val and test sets",
                )
                return result

        except Exception as e:
            if attempt == max_attempts - 1:
                print(f"Last attempt failed with error: {e}")
                continue

    # If we get here, try a more aggressive approach with smaller splits
    print(
        f"Failed to create balanced splits after {max_attempts} attempts. Trying with smaller splits...",
    )

    # Use very small val/test sizes to ensure we get samples
    small_val_size = max(
        0.05, min_samples_per_class *
        len(unique_labels) / total_samples,
    )
    small_test_size = small_val_size

    for attempt in range(100):
        try:
            train_paths, tmp_paths, train_labels, tmp_labels = train_test_split(
                paths,
                labels,
                test_size=small_val_size + small_test_size,
                stratify=labels,
                random_state=SEED + 1000 + attempt,
            )

            rel_val = small_val_size / (small_val_size + small_test_size)
            val_paths, test_paths, val_labels, test_labels = train_test_split(
                tmp_paths,
                tmp_labels,
                test_size=1 - rel_val,
                stratify=tmp_labels,
                random_state=SEED + 1000 + attempt,
            )

            val_unique, val_counts = np.unique(val_labels, return_counts=True)
            test_unique, test_counts = np.unique(
                test_labels, return_counts=True,
            )

            if len(val_counts) == len(unique_labels) and len(test_counts) == len(unique_labels):
                def pack(pths, lbls): return list(zip(pths, lbls))
                result = pack(train_paths, train_labels), pack(
                    val_paths, val_labels,
                ), pack(test_paths, test_labels)

                # Validate minimum requirements for fallback approach too
                val_dist = dict(zip(val_unique, val_counts))
                test_dist = dict(zip(test_unique, test_counts))

                for label in unique_labels:
                    val_count = val_dist.get(label, 0)
                    test_count = test_dist.get(label, 0)
                    if val_count < min_samples_per_class:
                        raise ValueError(
                            f"Class {int(label)} has only {int(val_count)} samples in validation set, minimum required is {min_samples_per_class}",
                        )
                    if test_count < min_samples_per_class:
                        raise ValueError(
                            f"Class {int(test_count)} has only {int(test_count)} samples in test set, minimum required is {min_samples_per_class}",
                        )

                print(
                    f"Split successful with smaller sizes after {attempt + 1} additional attempts",
                )
                print(
                    f"✓ Validation passed: All classes have at least {min_samples_per_class} samples in val and test sets",
                )
                return result

        except Exception:
            continue

    raise RuntimeError(
        f"Could not create val/test splits with both classes present after {max_attempts + 100} attempts. "
        f"Dataset may be too imbalanced. Try reducing min_samples_per_class or using a different split strategy.",
    )


def resolve_path(root: Path, rel: str | os.PathLike) -> Path | None:
    p = Path(rel)
    if not p.is_absolute() and not str(p).startswith(str(root)):
        p = root / p
    if p.exists():
        return p
    print(f"[warn] missing image: {p}", file=sys.stderr)
    return None


def load_binary_classification_data(json_path: Path, root_path: Path) -> tuple[list[tuple[Path, int]], dict, str, str]:
    """
    Load binary classification data from JSON file.

    Returns:
        items: List of (image_path, label) tuples
        label_map: Dictionary mapping category names to labels
        class0_name: Name of class 0
        class1_name: Name of class 1
    """
    with open(json_path) as fp:
        mapping = json.load(fp)

    # Validate that we have exactly 2 categories
    categories = list(mapping.keys())
    if len(categories) != 2:
        raise ValueError(
            f"Expected exactly 2 categories in JSON, got {len(categories)}: {categories}",
        )

    class0_name, class1_name = categories
    label_map = {class0_name: 0, class1_name: 1}

    items: list[tuple[Path, int]] = []
    for category, samples in mapping.items():
        for sample_id, meta in samples.items():
            for rel in meta.get('paths', []):
                p = resolve_path(root_path, rel)
                if p is not None:
                    items.append((p, label_map[category]))

    return items, label_map, class0_name, class1_name


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        '--json', required=True, type=Path,
        help='Path to binary classification JSON file',
    )
    ap.add_argument(
        '--root', type=Path, required=True,
        help='Root directory containing images',
    )
    ap.add_argument(
        '--out_dir', type=Path, required=True,
        help='Output directory for model and results',
    )
    ap.add_argument('--hf_token', default=os.getenv('HF_TOKEN'))
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--freeze_encoder', action='store_true')
    ap.add_argument(
        '--patience', type=int, default=5,
        help='Early stopping patience',
    )
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # 1 ▸ gather items
    print(f"Loading binary classification data from: {args.json}")
    items, label_map, class0_name, class1_name = load_binary_classification_data(
        args.json, args.root,
    )

    print(f"Loaded {len(items)} image-label pairs")
    if not items:
        raise RuntimeError('No valid images located – verify JSON and --root')

    print(f"Label mapping: {class0_name} → 0, {class1_name} → 1")
    print('Label counts:', Counter(label for _, label in items))

    print('Starting stratified split...')
    train_items, val_items, test_items = stratified_split(items)
    print('Stratified split done.')

    # 2 ▸ model & transforms
    encoder, train_tf, eval_tf = load_uni_backbone(
        args.hf_token, args.freeze_encoder,
    )

    train_ds = HistologyCrops(train_items, train_tf)
    val_ds = HistologyCrops(val_items, eval_tf)
    test_ds = HistologyCrops(test_items, eval_tf)

    # Compute class weights for WeightedRandomSampler
    train_labels = [label for _, label in train_items]
    class_sample_count = np.bincount(train_labels)
    weights = 1. / class_sample_count
    sample_weights = np.array([weights[label] for label in train_labels])
    sampler = WeightedRandomSampler(
        sample_weights, len(sample_weights), replacement=True,
    )

    # Choose device
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    pin_mem = device.type == 'cuda'  # pin_memory only when supported
    # Use sampler for train_loader, shuffle for val/test
    train_loader = DataLoader(
        train_ds, args.batch, sampler=sampler, num_workers=4, pin_memory=pin_mem,
    )
    val_loader = DataLoader(
        val_ds, args.batch, shuffle=False,
        num_workers=4, pin_memory=pin_mem,
    )
    test_loader = DataLoader(
        test_ds, args.batch, shuffle=False, num_workers=4, pin_memory=pin_mem,
    )

    model = UniClassifier(encoder).to(device)
    loss_fn = nn.CrossEntropyLoss()
    opt = optim.AdamW(
        filter(
            lambda p: p.requires_grad,
            model.parameters(),
        ), lr=args.lr, weight_decay=1e-4,
    )
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    # Initialize early stopping
    early_stopping = EarlyStopping(patience=args.patience)

    best = 0.0
    for ep in range(1, args.epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
        sched.step()

        # validation
        model.eval()
        p, g = [], []
        with torch.no_grad():
            for x, y in val_loader:
                logits = model(x.to(device))
                preds = logits.argmax(1).cpu()
                p.append(preds)
                g.append(y)
        val_preds, val_gts = torch.cat(p), torch.cat(g)

        # Debug validation data
        val_pred_unique, val_pred_counts = torch.unique(
            val_preds, return_counts=True,
        )
        val_gt_unique, val_gt_counts = torch.unique(
            val_gts, return_counts=True,
        )
        print(
            f"  Val preds: {dict(zip(val_pred_unique.tolist(), val_pred_counts.tolist()))}",
        )
        print(
            f"  Val gts:   {dict(zip(val_gt_unique.tolist(), val_gt_counts.tolist()))}",
        )

        f1s = tm_f1(val_preds, val_gts, task='binary', average='none')
        print(f"  F1 scores shape: {f1s.shape}, values: {f1s.tolist()}")

        if f1s.numel() >= 2:
            f1_pos = f1s[0].item()  # F1 for minority/target class (class 0)
        else:
            # Fallback: use macro F1 (scalar)
            f1_pos = float(f1s)
        print(f"Epoch {ep}/{args.epochs}  val_f1_{class0_name}={f1_pos:.4f}")

        # Check early stopping (now on F1-positive)
        if early_stopping(f1_pos, model):
            print(f"Early stopping triggered after {ep} epochs")
            break

        # save first or best model so we always have a model to test
        if ep == 1 or f1_pos > best:
            best = f1_pos
            torch.save(model.state_dict(), args.out_dir / 'best.pth')

    # Restore best model weights
    early_stopping.restore_best_model(model)

    # 3 ▸ test
    best_ckpt = args.out_dir / 'best.pth'
    if not best_ckpt.exists():
        print(
            f"[ERROR] No checkpoint was saved during training. No best.pth found at {best_ckpt}.",
        )
        print('This usually means the model never improved on the validation set. Check your data and training logs.')
        return
    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    model.eval()
    p, g = [], []
    with torch.no_grad():
        for x, y in test_loader:
            logits = model(x.to(device))
            preds = logits.argmax(1).cpu()
            p.append(preds)
            g.append(y)
    preds, gts = torch.cat(p), torch.cat(g)
    f1s = tm_f1(preds, gts, task='binary', average='none')
    if f1s.numel() >= 2:
        f1_pos = f1s[0].item()
    else:
        f1_pos = float(f1s)

    # Save classification info
    classification_info = {
        'class0_name': class0_name,
        'class1_name': class1_name,
        'label_map': label_map,
    }
    with open(args.out_dir / 'classification_info.json', 'w') as fp:
        json.dump(classification_info, fp, indent=2)

    metrics = {
        f"test_f1_{class0_name}": f1_pos,
        'test_f1_macro': tm_f1(preds, gts, task='binary').item(),
        f"best_val_f1_{class0_name}": best,
        'epochs_trained': ep,
        'class0_name': class0_name,
        'class1_name': class1_name,
    }
    with open(args.out_dir / 'metrics.json', 'w') as fp:
        json.dump(metrics, fp, indent=2)
    print('✓ finished', metrics)


if __name__ == '__main__':
    main()

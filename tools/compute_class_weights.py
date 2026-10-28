#!/usr/bin/env python3
"""Inverse-frequency class weights for the pT(H) bins, from the training events.

Training events are those used by the launcher: event_mod5 < 3.
Weight of bin k:  w_k = N / (n_classes * N_k)
(N = all training events, N_k = training events in bin k), so the three bins
contribute equally to the loss.

Usage:
  python tools/compute_class_weights.py "/work/abelvede/multilepton-analysis/neuralnetwork/snapshots/STXS/2lss_signal/2026-09-23/*/TTHtoNon2B.root"
"""

import glob
import sys

import numpy as np
import uproot

N_CLASSES = 3

pattern = sys.argv[1]
files = sorted(glob.glob(pattern))
if not files:
    sys.exit(f"No file matches {pattern}")

counts = np.zeros(N_CLASSES, dtype=np.int64)
for path in files:
    arrays = uproot.open(path)["Events"].arrays(["GenHiggs_ptBin", "event_mod5"], library="np")
    train = arrays["event_mod5"] < 3
    bins = arrays["GenHiggs_ptBin"][train].astype(int)
    file_counts = np.bincount(bins, minlength=N_CLASSES)
    counts += file_counts
    print(f"{path}: training events per bin {file_counts.tolist()}")

total = counts.sum()
fractions = counts / total
weights = total / (N_CLASSES * counts)

print(f"\nTraining events per bin: {counts.tolist()}  (total {total})")
print("Fractions:               " + ", ".join(f"{f:.4f}" for f in fractions))
print("Inverse-frequency weights:" + ", ".join(f" {w:.4f}" for w in weights))
print("Relative to bin 0:       " + ", ".join(f"{w / weights[0]:.3f}" for w in weights))
print("\nNetwork option for the launcher:")
print("  -o class_weights \"[" + ",".join(f"{w:.4f}" for w in weights) + "]\"")
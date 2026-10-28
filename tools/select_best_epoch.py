#!/usr/bin/env python3
"""Choose the best epoch of a Weaver training by balanced accuracy on the validation set.

Weaver logs the validation confusion matrix after every epoch (row-normalized:
each row = one true bin, so the diagonal is the fraction correct per bin).
Balanced accuracy = mean of the diagonal; it counts every pT(H) bin equally.

Usage:
  python tools/select_best_epoch.py logs_stxs/<RUN_NAME>_train.log
  python tools/select_best_epoch.py logs_stxs/<RUN_NAME>_train.log --epoch-only   # prints only the epoch number
"""

import argparse
import re

import numpy as np

ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")
EPOCH_RE = re.compile(r"Epoch #(\d+) validating")
EVAL_ACC_RE = re.compile(r"Eval AvgLoss:\s*[0-9.eE+-]+,\s*AvgAcc:\s*([0-9.eE+-]+)")
NUMBER_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def parse(log_file):
    epoch = None
    accuracy = {}
    matrices = {}
    collecting = None
    with open(log_file, encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = ANSI_ESCAPE.sub("", raw)
            if collecting is not None:
                collecting += line
                if "]]" in line:
                    values = [float(v) for v in NUMBER_RE.findall(collecting)]
                    n = int(round(len(values) ** 0.5))
                    if epoch is not None and n * n == len(values):
                        matrices[epoch] = np.array(values).reshape(n, n)
                    collecting = None
                continue
            m = EPOCH_RE.search(line)
            if m:
                epoch = int(m.group(1))
                continue
            m = EVAL_ACC_RE.search(line)
            if m and epoch is not None:
                accuracy[epoch] = float(m.group(1))
                continue
            if "- confusion_matrix:" in line:
                collecting = ""
    return accuracy, matrices


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("log_file")
    parser.add_argument("--epoch-only", action="store_true", help="Print only the selected epoch number.")
    args = parser.parse_args()

    accuracy, matrices = parse(args.log_file)
    if not matrices:
        raise SystemExit(f"No validation confusion matrices found in {args.log_file}")

    epochs = sorted(matrices)
    balanced = {e: float(np.mean(np.diag(matrices[e]))) for e in epochs}
    best_bal = max(epochs, key=lambda e: balanced[e])
    best_acc = max(accuracy, key=lambda e: accuracy[e]) if accuracy else None

    if args.epoch_only:
        print(best_bal)
        return

    n = matrices[epochs[0]].shape[0]
    print("epoch | " + " | ".join(f"diag bin{k}" for k in range(n)) + " | balanced acc | accuracy")
    for e in epochs:
        diag = np.diag(matrices[e])
        mark = ""
        if e == best_bal:
            mark += "  <- best balanced accuracy"
        if e == best_acc:
            mark += "  <- best accuracy (Weaver's best_epoch)"
        acc = f"{accuracy[e]:.5f}" if e in accuracy else "   -   "
        print(f"{e:5d} | " + " | ".join(f"{d:9.4f}" for d in diag) + f" | {balanced[e]:12.5f} | {acc}{mark}")

    print(f"\nBest epoch by balanced accuracy: {best_bal} (balanced accuracy {balanced[best_bal]:.5f})")
    if best_acc is not None:
        print(f"Weaver's best epoch (accuracy):   {best_acc} (balanced accuracy {balanced.get(best_acc, float('nan')):.5f})")
    print(f"\nConfusion matrix at epoch {best_bal} (rows = true bin, row-normalized):")
    print(np.array2string(matrices[best_bal], precision=4))
    print(f"\nModel file: <model folder>/fullmodel_epoch-{best_bal}_state.pt")


if __name__ == "__main__":
    main()
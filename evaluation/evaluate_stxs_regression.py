#!/usr/bin/env python3

import argparse
import glob
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", f"/tmp/matplotlib-{os.environ.get('USER', 'user')}")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import uproot
    from uproot.source.file import MemmapSource
except ImportError as exc:
    raise SystemExit(
        "Could not import uproot. Run this inside the Weaver environment, e.g.\n"
        "  source /work/inadalli/miniconda3/etc/profile.d/conda.sh\n"
        "  conda activate weaver\n"
    ) from exc


STXS_EDGES = [120.0, 200.0]  # GeV, boundaries of the pT(H) bins
BIN_NAMES = ["0-120 GeV", "120-200 GeV", ">200 GeV"]

# Branch that defines the data split (from the observers in data/stxs_regression.yaml) and the values each dataset
# must have: held-out/test events have event_mod5 == 4; the "Train" dataset is every event that is not held out
# (event_mod5 != 4, the selection in the data config): training events (0, 1, 2) plus validation events (3).
SPLIT_BRANCH = "event_mod5"
EXPECTED_SPLIT = {"Held-out/test": {4}, "Train": {0, 1, 2, 3}}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Make training-curve and pT(H)-distribution plots for the STXS pT(H) regressor "
            "from Weaver prediction outputs and training logs."
        )
    )
    parser.add_argument(
        "--input-dir",
        help=(
            "Directory containing held-out/test Weaver prediction files. If set, the script looks for "
            "<input-stem>.root, e.g. result/stxs/<RUN_NAME>/predictions_sample.root (Weaver adds the sample "
            "name 'sample' from --data-test \"sample:...\" to the PREDICT_OUTPUT file name)."
        ),
    )
    parser.add_argument(
        "--input-files",
        nargs="+",
        default=[],
        help="Explicit held-out/test prediction ROOT files or glob patterns. Can be used instead of --input-dir.",
    )
    parser.add_argument(
        "--input-stem",
        default="predictions_sample",
        help="Prediction-file stem used with --input-dir.",
    )
    parser.add_argument(
        "--train-input-dir",
        help=(
            "Optional directory containing train-split Weaver prediction files. If set, the script looks for "
            "<train-input-stem>.root."
        ),
    )
    parser.add_argument(
        "--train-input-files",
        nargs="+",
        default=[],
        help="Explicit train-split prediction ROOT files or glob patterns.",
    )
    parser.add_argument(
        "--train-input-stem",
        default="predictions_train_sample",
        help="Prediction-file stem used with --train-input-dir.",
    )
    parser.add_argument(
        "--log-file",
        nargs="+",
        default=[],
        help=(
            "Training log(s) to parse for the loss, MSE and MAE curves (all in log(pT) units). "
            "With several logs, the runs are also compared in one plot."
        ),
    )
    parser.add_argument(
        "--log-labels",
        nargs="+",
        default=None,
        help="Name of each run, one per --log-file (default: the log file name).",
    )
    parser.add_argument(
        "--output",
        default="evaluation_output/stxs_regression_plots",
        help="Output directory for plots and metric summaries.",
    )
    parser.add_argument("--tree", default="Events", help="Tree name in the prediction ROOT files.")
    parser.add_argument(
        "--truth-branch",
        default="GenHiggs_pt",
        help="True pT(H) in GeV (an observer in the prediction file).",
    )
    parser.add_argument(
        "--prediction-branch",
        default="output",
        help="Model output branch. The model predicts log(pT) - center, so the predicted pT is exp(output + center).",
    )
    parser.add_argument(
        "--label-branch",
        default="target_logpt",
        help=(
            "Label branch written by Weaver (= log(max(pT, 1)) - center). Used to find the center the model was "
            "trained with (TARGET_CENTER in the launcher)."
        ),
    )
    parser.add_argument(
        "--target-center",
        type=float,
        default=None,
        help=(
            "Center subtracted from log(pT) in the label. Default: found from the label branch "
            "(center = log(max(true pT, 1)) - label); 0 if the label branch is not in the file."
        ),
    )
    parser.add_argument(
        "--corr-bins",
        type=int,
        default=200,
        help="Number of bins per axis in the predicted vs true pT plot (linear axes from 0 to --corr-pt-max).",
    )
    parser.add_argument(
        "--corr-pt-max",
        type=float,
        default=None,
        help=(
            "Overflow edge of both axes of the predicted vs true pT plot in GeV: true or predicted pT above it is "
            "drawn in the last row/column. Default: the same as --pt-max. The correlation coefficients always use "
            "the unchanged pT of every event."
        ),
    )
    parser.add_argument(
        "--pt-max",
        type=float,
        default=800.0,
        help=(
            "Overflow edge of the pT axis in GeV (default 800): every event with true or predicted pT at or above it "
            "is counted in one overflow bin [pt-max, pt-max + bin width). The overflow fractions are given in the "
            "summary."
        ),
    )
    parser.add_argument("--bin-width", type=float, default=1.0, help="Width of the pT bins in GeV.")
    parser.add_argument(
        "--title",
        default="STXS ParticleTransformer regression",
        help="Title prefix used in the plots.",
    )
    args = parser.parse_args()
    if args.log_labels is not None and len(args.log_labels) != len(args.log_file):
        parser.error("--log-labels needs one entry per --log-file")
    return args


def expand_input_files(input_dir, input_files, stem):
    paths = []

    if input_dir:
        paths.append(Path(input_dir) / f"{stem}.root")

    for pattern in input_files:
        expanded = os.path.expandvars(os.path.expanduser(pattern))
        matches = sorted(glob.glob(expanded))
        if matches:
            paths.extend(Path(match) for match in matches)
        else:
            paths.append(Path(expanded))

    unique_paths = []
    seen = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_paths.append(path)

    missing = [str(path) for path in unique_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing input file(s):\n  " + "\n  ".join(missing))
    if not unique_paths:
        raise ValueError("No input files were provided. Use --input-dir or --input-files.")

    return unique_paths


def load_predictions(paths, args):
    needed_branches = [args.truth_branch, args.prediction_branch]
    arrays = {branch: [] for branch in needed_branches}

    for path in paths:
        with uproot.open(path, handler=MemmapSource) as root_file:
            if args.tree not in root_file:
                raise KeyError(f"Tree '{args.tree}' not found in {path}")

            tree = root_file[args.tree]
            available = set(tree.keys())
            missing = [branch for branch in needed_branches if branch not in available]
            if missing:
                raise KeyError(
                    f"Missing branch(es) in {path}: {missing}\n"
                    f"Available branches are: {sorted(available)}"
                )

            optional = [b for b in (SPLIT_BRANCH, args.label_branch) if b in available]
            read_branches = needed_branches + optional
            data = tree.arrays(read_branches, library="np")
            for branch in read_branches:
                arrays.setdefault(branch, []).append(np.asarray(data[branch]))

    for branch in (SPLIT_BRANCH, args.label_branch):
        if branch in arrays and len(arrays[branch]) != len(paths):
            del arrays[branch]
    return {branch: np.concatenate(values) for branch, values in arrays.items()}


def find_center(data, args):
    """Center c of the label log(max(pT, 1)) - c, from --target-center or from the label branch."""
    if args.target_center is not None:
        return args.target_center
    if args.label_branch not in data:
        print(f"Note: no {args.label_branch} branch in the prediction file; assuming center 0 (pT = exp(output)).")
        return 0.0
    log_pt = np.log(np.maximum(np.asarray(data[args.truth_branch]).astype(np.float64), 1.0))
    difference = log_pt - np.asarray(data[args.label_branch]).astype(np.float64)
    difference = difference[np.isfinite(difference)]
    center = float(np.median(difference))
    # the label is stored as float32, so the difference varies by ~1e-6; a larger spread means a different label
    if np.max(np.abs(difference - center)) > 1e-3:
        raise ValueError(
            f"{args.label_branch} is not log(max({args.truth_branch}, 1)) minus a constant "
            f"(spread {np.max(np.abs(difference - center)):.3g}); give the center with --target-center."
        )
    return round(center, 4) + 0.0  # + 0.0 turns -0.0 into 0.0


def sanitize_inputs(data, args):
    center = find_center(data, args)
    pt_true = np.asarray(data[args.truth_branch]).astype(np.float64)
    pt_pred = np.exp(np.asarray(data[args.prediction_branch]).astype(np.float64) + center)

    valid = np.isfinite(pt_true) & np.isfinite(pt_pred) & (pt_true >= 0)

    n_removed = len(valid) - int(np.sum(valid))
    if n_removed:
        print(f"Removed {n_removed} event(s) with invalid true or predicted pT.")

    split = np.asarray(data[SPLIT_BRANCH]).astype(np.int64)[valid] if SPLIT_BRANCH in data else None

    return pt_true[valid], pt_pred[valid], split, center


def load_dataset(label, input_dir, input_files, stem, args):
    input_paths = expand_input_files(input_dir, input_files, stem)
    data = load_predictions(input_paths, args)
    pt_true, pt_pred, split, center = sanitize_inputs(data, args)
    return {
        "label": label,
        "input_paths": input_paths,
        "pt_true": pt_true,
        "pt_pred": pt_pred,
        "split": split,
        "center": center,
    }


def stxs_fractions(pt):
    bins = np.digitize(pt, STXS_EDGES)
    return np.bincount(bins, minlength=len(BIN_NAMES)) / len(pt)


def overflow_edge(datasets, pt_max):
    """Overflow edge in GeV: --pt-max (default 800 GeV)."""
    return float(pt_max)


def to_overflow(pt, edge, bin_width):
    """pT with every value at or above the overflow edge moved into the overflow bin [edge, edge + bin_width)."""
    return np.where(pt >= edge, edge + 0.5 * bin_width, pt)


def save_pt_distribution(output_dir, title, datasets, pt_max, bin_width):
    pt_max = overflow_edge(datasets, pt_max)
    n_bins = int(round(pt_max / bin_width)) + 1  # regular bins up to pt_max, then the overflow bin
    edges = np.arange(n_bins + 1) * bin_width
    centers = 0.5 * (edges[1:] + edges[:-1])
    payload = {"bin_edges": edges, "stxs_edges": np.asarray(STXS_EDGES), "overflow_edge": np.asarray(pt_max)}

    fig, (ax, ax_ratio) = plt.subplots(
        2, 1, figsize=(9, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05}
    )
    # colours per dataset (true, predicted). With 1 GeV bins a dashed line is not visible (every bin edge is a
    # vertical step), so the datasets are told apart by colour. The test dataset is drawn last, on top.
    styles = {0: ("black", "tab:blue"), 1: ("tab:green", "tab:orange")}
    for index in reversed(range(len(datasets))):
        dataset = datasets[index]
        true_color, pred_color = styles[index]
        key = dataset_key(dataset)
        n_events = len(dataset["pt_true"])
        # the last bin is the overflow bin: every event at or above pt_max is counted there
        true_counts, _ = np.histogram(to_overflow(dataset["pt_true"], pt_max, bin_width), bins=edges)
        pred_counts, _ = np.histogram(to_overflow(dataset["pt_pred"], pt_max, bin_width), bins=edges)
        true_frac = true_counts / n_events
        pred_frac = pred_counts / n_events

        ax.stairs(true_frac, edges, color=true_color, linewidth=1.0, label=f"True pT(H), {dataset['label']}")
        ax.stairs(pred_frac, edges, color=pred_color, linewidth=1.0, label=f"Predicted pT(H), {dataset['label']}")

        # ratio predicted / true per bin, with Poisson uncertainty from both counts
        filled = (true_counts > 0) & (pred_counts > 0)
        ratio = np.full(len(centers), np.nan)
        ratio_err = np.full(len(centers), np.nan)
        ratio[filled] = pred_counts[filled] / true_counts[filled]
        ratio_err[filled] = ratio[filled] * np.sqrt(1.0 / pred_counts[filled] + 1.0 / true_counts[filled])
        ax_ratio.errorbar(centers, ratio, yerr=ratio_err, fmt="o", markersize=1.5, elinewidth=0.5,
                          color=pred_color, label=dataset["label"])

        payload[f"{key}_true_counts"] = true_counts
        payload[f"{key}_predicted_counts"] = pred_counts
        payload[f"{key}_n_events"] = np.asarray(n_events)

    for edge in STXS_EDGES:
        ax.axvline(edge, color="grey", linestyle=":", linewidth=1.2)
        ax_ratio.axvline(edge, color="grey", linestyle=":", linewidth=1.2)
    for axis in (ax, ax_ratio):
        axis.axvline(pt_max, color="tab:red", linestyle="--", linewidth=1.0)
    ax.text(pt_max, 0.97, f"overflow\n>= {pt_max:g} GeV ", color="tab:red", fontsize=8, ha="right", va="top",
            transform=ax.get_xaxis_transform())
    ax_ratio.axhline(1.0, color="black", linewidth=0.8)

    ax.set_ylabel(f"Fraction of events per {bin_width:g} GeV")
    ax.set_title(f"{title}: pT(H) distribution (dotted lines: STXS bin edges)")
    handles, labels = ax.get_legend_handles_labels()
    order = sorted(range(len(labels)), key=lambda i: ("Train" in labels[i], labels[i].startswith("Predicted")))
    ax.legend([handles[i] for i in order], [labels[i] for i in order], frameon=False, fontsize=8)
    ax.grid(alpha=0.25)
    ax_ratio.set_xlabel("pT(H) [GeV]")
    ax_ratio.set_ylabel("Predicted / true")
    ax_ratio.set_ylim(0.0, 2.0)
    ax_ratio.set_xlim(edges[0], edges[-1])
    ax_ratio.grid(alpha=0.25)

    fig.savefig(output_dir / "pt_distribution.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "pt_distribution.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    np.savez(output_dir / "pt_distribution.npz", **payload)


def dataset_key(dataset):
    return dataset["label"].lower().replace(" ", "_").replace("/", "_")


def save_migration_matrix(output_dir, title, dataset):
    """Migration matrix between the true and the predicted STXS bin.

    Each event is put in a predicted bin by cutting the predicted pT at the STXS edges (120, 200 GeV), the same
    edges as for the true pT. Two normalizations:
      row-normalized:    fraction of the events of a true bin that land in each predicted bin
                         (same convention as the classifier confusion matrix, so the diagonals can be compared)
      column-normalized: fraction of the events of a predicted bin that come from each true bin (purity)
    """
    n_bins = len(BIN_NAMES)
    true_bin = np.digitize(dataset["pt_true"], STXS_EDGES)
    pred_bin = np.digitize(dataset["pt_pred"], STXS_EDGES)
    counts = np.zeros((n_bins, n_bins), dtype=np.int64)
    np.add.at(counts, (true_bin, pred_bin), 1)

    row_sums = counts.sum(axis=1, keepdims=True)
    col_sums = counts.sum(axis=0, keepdims=True)
    row_norm = np.divide(counts, row_sums, out=np.zeros_like(counts, dtype=np.float64), where=row_sums != 0)
    col_norm = np.divide(counts, col_sums, out=np.zeros_like(counts, dtype=np.float64), where=col_sums != 0)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    for ax, matrix, name in ((axes[0], row_norm, "Row-normalized (fraction of true bin)"),
                             (axes[1], col_norm, "Column-normalized (purity of predicted bin)")):
        image = ax.imshow(matrix, cmap="Blues", vmin=0.0, vmax=1.0)
        fig.colorbar(image, ax=ax, label=name)
        ax.set_xticks(np.arange(n_bins), BIN_NAMES)
        ax.set_yticks(np.arange(n_bins), BIN_NAMES)
        for i in range(n_bins):
            for j in range(n_bins):
                ax.text(j, i, f"{matrix[i, j]:.3f}\n({counts[i, j]})", ha="center", va="center", fontsize=9,
                        color="white" if matrix[i, j] > 0.6 else "black")
        ax.set_xlabel("Predicted pT(H) bin (cut on predicted pT)")
        ax.set_ylabel("True pT(H) bin")
    fig.suptitle(f"{title}: migration matrix, {dataset['label']} (number of events in brackets)")
    fig.tight_layout()

    stem = f"migration_matrix_{dataset_key(dataset)}"
    fig.savefig(output_dir / f"{stem}.pdf")
    fig.savefig(output_dir / f"{stem}.png", dpi=200)
    plt.close(fig)

    metrics = {
        "counts": counts,
        "row_norm": row_norm,
        "col_norm": col_norm,
        "accuracy": float(np.trace(counts) / counts.sum()),
        "balanced_accuracy": float(np.mean(np.diag(row_norm))),
    }
    np.savez(output_dir / f"{stem}.npz", stxs_edges=np.asarray(STXS_EDGES), **metrics)
    return metrics


def rank(values):
    """Rank of each value (0 = smallest). pT is continuous, so ties do not occur in practice."""
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[np.argsort(values, kind="stable")] = np.arange(len(values))
    return ranks


def correlation_metrics(pt_true, pt_pred):
    return {
        # linear correlation of the quantity the network was trained on
        "pearson_log_pt": float(np.corrcoef(np.log(np.maximum(pt_true, 1.0)), np.log(pt_pred))[0, 1]),
        # linear correlation in GeV
        "pearson_pt": float(np.corrcoef(pt_true, pt_pred)[0, 1]),
        # correlation of the ranks: measures only the ordering, which is all that matters for cuts on predicted pT
        "spearman": float(np.corrcoef(rank(pt_true), rank(pt_pred))[0, 1]),
    }


def save_correlation(output_dir, title, dataset, n_bins_2d=200, pt_max=None):
    """2D histogram of predicted vs true pT on linear axes, with the line predicted = true.

    pT at or above pt_max (true or predicted) is drawn in the last bin of that axis (overflow).
    """
    pt_true = dataset["pt_true"]
    pt_pred = dataset["pt_pred"]
    metrics = correlation_metrics(pt_true, pt_pred)  # unchanged pT of every event

    low = 0.0
    bin_width = pt_max / (n_bins_2d - 1)  # n_bins_2d - 1 regular bins up to pt_max, then the overflow bin
    high = pt_max + bin_width
    edges = np.linspace(low, high, n_bins_2d + 1)
    metrics["n_overflow"] = int(np.sum((pt_true >= pt_max) | (pt_pred >= pt_max)))
    counts, _, _ = np.histogram2d(to_overflow(pt_true, pt_max, bin_width), to_overflow(pt_pred, pt_max, bin_width),
                                  bins=[edges, edges])

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    image = ax.pcolormesh(edges, edges, np.where(counts > 0, counts, np.nan).T,
                          norm=matplotlib.colors.LogNorm(), cmap="viridis")
    fig.colorbar(image, ax=ax, label="Events", fraction=0.046, pad=0.04)
    ax.plot([low, high], [low, high], color="red", linewidth=1.0, label="Predicted = true")
    for edge in STXS_EDGES:
        ax.axvline(edge, color="grey", linestyle=":", linewidth=1.2)
        ax.axhline(edge, color="grey", linestyle=":", linewidth=1.2)
    ax.axvline(pt_max, color="tab:red", linestyle="--", linewidth=0.8)
    ax.axhline(pt_max, color="tab:red", linestyle="--", linewidth=0.8)
    ax.text(pt_max, 0.5 * pt_max, f"overflow >= {pt_max:g} GeV ", color="tab:red", fontsize=7, rotation=90,
            ha="right", va="center")
    ax.set_xlim(low, edges[-1])
    ax.set_ylim(low, edges[-1])
    ax.set_aspect("equal")
    ax.set_xlabel("True pT(H) [GeV]")
    ax.set_ylabel("Predicted pT(H) [GeV]")
    ax.set_title(f"{title}: predicted vs true pT(H), {dataset['label']}", fontsize=10)
    ax.text(0.03, 0.97,
            f"Pearson (log pT): {metrics['pearson_log_pt']:.3f}\n"
            f"Pearson (pT): {metrics['pearson_pt']:.3f}\n"
            f"Spearman (rank): {metrics['spearman']:.3f}",
            transform=ax.transAxes, va="top", fontsize=9, bbox={"facecolor": "white", "alpha": 0.8})
    ax.legend(frameon=False, loc="lower right", fontsize=8)

    stem = f"pt_correlation_{dataset_key(dataset)}"
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    np.savez(output_dir / f"{stem}.npz", bin_edges=edges, counts=counts,
             **{k: np.asarray(v) for k, v in metrics.items()})
    return metrics


def save_bias_resolution(output_dir, title, datasets, n_bins=40):
    """Bias and resolution of predicted pT / true pT as a function of true pT.

    The true-pT bins are quantiles of the true pT of each dataset, so every bin has the same number of events
    and every event is in a bin (no lower or upper cut). In each bin of true pT, for r = predicted pT / true pT:
      bias:       median of r (1 = unbiased); the band is the central 68% interval (16% to 84% quantile of r)
      resolution: (84% quantile - 16% quantile) / (2 * median), the relative width of r
    """
    fig, (ax, ax_res) = plt.subplots(
        2, 1, figsize=(9, 7.5), sharex=True, gridspec_kw={"height_ratios": [2, 1], "hspace": 0.05}
    )
    colors = {0: "tab:blue", 1: "tab:orange"}
    payload = {"stxs_edges": np.asarray(STXS_EDGES)}

    for index in reversed(range(len(datasets))):
        dataset = datasets[index]
        pt_true = dataset["pt_true"]
        ratio = dataset["pt_pred"] / np.where(pt_true > 0, pt_true, np.nan)
        edges = np.quantile(pt_true, np.linspace(0.0, 1.0, n_bins + 1))
        which = np.clip(np.digitize(pt_true, edges[1:-1]), 0, n_bins - 1)

        x = np.empty(n_bins)
        q16 = np.empty(n_bins)
        q50 = np.empty(n_bins)
        q84 = np.empty(n_bins)
        for b in range(n_bins):
            in_bin = which == b
            x[b] = np.median(pt_true[in_bin])
            q16[b], q50[b], q84[b] = np.nanquantile(ratio[in_bin], [0.16, 0.50, 0.84])
        resolution = (q84 - q16) / (2.0 * q50)

        color = colors[index]
        ax.fill_between(x, q16, q84, color=color, alpha=0.2, linewidth=0)
        ax.plot(x, q50, "o-", color=color, markersize=3, linewidth=1.2,
                label=f"{dataset['label']}: median, band = 16%-84%")
        ax_res.plot(x, resolution, "o-", color=color, markersize=3, linewidth=1.2, label=dataset["label"])

        key = dataset_key(dataset)
        payload[f"{key}_true_pt_bin_edges"] = edges
        payload[f"{key}_true_pt_median"] = x
        payload[f"{key}_ratio_q16"] = q16
        payload[f"{key}_ratio_median"] = q50
        payload[f"{key}_ratio_q84"] = q84
        payload[f"{key}_resolution"] = resolution

    ax.axhline(1.0, color="black", linewidth=0.8)
    for axis in (ax, ax_res):
        for edge in STXS_EDGES:
            axis.axvline(edge, color="grey", linestyle=":", linewidth=1.2)
        axis.grid(alpha=0.25)
    ax.set_ylabel("Predicted pT / true pT")
    ax.set_title(f"{title}: bias vs true pT(H) (each point: 1/{n_bins} of the events)")
    ax.legend(frameon=False, fontsize=8)
    ax_res.set_xscale("log")
    ax_res.set_xlabel("True pT(H) [GeV] (median of the bin)")
    ax_res.set_ylabel("Relative resolution\n(q84 - q16) / (2 median)")
    ax_res.legend(frameon=False, fontsize=8)

    fig.savefig(output_dir / "bias_resolution.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "bias_resolution.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    np.savez(output_dir / "bias_resolution.npz", **payload)


ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")
EPOCH_RE = re.compile(r"Epoch #(\d+) (training|validating)")
TRAIN_RE = re.compile(r"Train AvgLoss:\s*([0-9.eE+-]+),\s*AvgMSE:\s*([0-9.eE+-]+),\s*AvgMAE:\s*([0-9.eE+-]+)")
VAL_LOSS_RE = re.compile(r"Epoch #(\d+): Current validation metric:\s*([0-9.eE+-]+)")
METRIC_NAME_RE = re.compile(r"-\s*(mean_squared_error|mean_absolute_error):")
NUMBER_RE = re.compile(r"^\s*([-+]?[0-9.]+(?:[eE][-+]?\d+)?)\s*$")


def parse_loss_log(log_file):
    """Return {quantity: (train_values_by_epoch, validation_values_by_epoch)} for loss, mse, mae.

    In regression mode Weaver logs:
      training:   'Train AvgLoss: x, AvgMSE: y, AvgMAE: z'
      validation: 'Epoch #N: Current validation metric: x'   (= validation loss)
                  '- mean_squared_error:' / '- mean_absolute_error:' followed by the value on the next line
    """
    current_epoch = None
    pending_metric = None
    curves = {q: ({}, {}) for q in ("loss", "mse", "mae")}
    validation_metrics = {"mean_squared_error": "mse", "mean_absolute_error": "mae"}

    with open(log_file, "r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = ANSI_ESCAPE.sub("", raw_line)

            if pending_metric is not None:
                number_match = NUMBER_RE.match(line)
                if number_match and current_epoch is not None:
                    curves[pending_metric][1][current_epoch] = float(number_match.group(1))
                pending_metric = None
                continue

            epoch_match = EPOCH_RE.search(line)
            if epoch_match:
                current_epoch = int(epoch_match.group(1))
                continue

            train_match = TRAIN_RE.search(line)
            if train_match:
                epoch = current_epoch if current_epoch is not None else len(curves["loss"][0])
                for quantity, value in zip(("loss", "mse", "mae"), train_match.groups()):
                    curves[quantity][0][epoch] = float(value)
                continue

            val_match = VAL_LOSS_RE.search(line)
            if val_match:
                curves["loss"][1][int(val_match.group(1))] = float(val_match.group(2))
                continue

            metric_match = METRIC_NAME_RE.search(line)
            if metric_match:
                pending_metric = validation_metrics[metric_match.group(1)]

    if not curves["loss"][0] and not curves["loss"][1]:
        raise ValueError(f"Could not find regression loss entries in {log_file}")

    return curves


def _sorted_curve(values_by_epoch):
    epochs = np.asarray(sorted(values_by_epoch), dtype=np.int64)
    values = np.asarray([values_by_epoch[epoch] for epoch in epochs], dtype=np.float64)
    return epochs, values


QUANTITIES = {
    # name: y-axis label (all in log(pT) units, because the model predicts log(pT))
    "loss": "Loss",
    "mse": "Mean squared error of log(pT)",
    "mae": "Mean absolute error of log(pT)",
}


def save_loss_curve(output_dir, title, log_file):
    curves = parse_loss_log(log_file)
    summary = {}

    for quantity, ylabel in QUANTITIES.items():
        train_values, val_values = curves[quantity]

        plt.figure(figsize=(7, 5.5))
        payload = {}

        if train_values:
            epochs, values = _sorted_curve(train_values)
            plt.plot(epochs, values, marker="o", markersize=3, linewidth=1.8, label="Train")
            payload["train_epoch"] = epochs
            payload[f"train_{quantity}"] = values
            summary[f"train_{quantity}_last"] = values[-1]

        if val_values:
            epochs, values = _sorted_curve(val_values)
            plt.plot(epochs, values, marker="o", markersize=3, linewidth=1.8, label="Validation")
            payload["validation_epoch"] = epochs
            payload[f"validation_{quantity}"] = values
            summary[f"validation_{quantity}_last"] = values[-1]
            summary[f"validation_{quantity}_min"] = values[np.argmin(values)]
            summary[f"validation_{quantity}_min_epoch"] = epochs[np.argmin(values)]

        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(f"{title}: {quantity} curve")
        plt.grid(alpha=0.25)
        plt.legend(frameon=False)
        plt.tight_layout()

        plt.savefig(output_dir / f"{quantity}_curve.pdf")
        plt.savefig(output_dir / f"{quantity}_curve.png", dpi=200)
        plt.close()

        payload["log_file"] = np.asarray(str(log_file))
        np.savez(output_dir / f"{quantity}_curve.npz", **payload)

    return summary


def save_loss_comparison(output_dir, title, log_files, run_labels):
    all_curves = [parse_loss_log(log_file) for log_file in log_files]
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    payload = {}

    for quantity, ylabel in QUANTITIES.items():
        plt.figure(figsize=(8, 5.5))
        for run, (run_label, curves) in enumerate(zip(run_labels, all_curves)):
            color = colors[run % len(colors)]
            key = run_slug(run_label)
            for values_by_epoch, linestyle, name in ((curves[quantity][0], "--", "train"),
                                                     (curves[quantity][1], "-", "validation")):
                if values_by_epoch:
                    epochs, values = _sorted_curve(values_by_epoch)
                    plt.plot(epochs, values, linestyle, color=color, linewidth=1.8, label=f"{run_label}, {name}")
                    payload[f"{key}_{name}_epoch"] = epochs
                    payload[f"{key}_{name}_{quantity}"] = values

        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(f"{title}: {quantity}, all runs")
        plt.grid(alpha=0.25)
        plt.legend(frameon=False, fontsize=8)
        plt.tight_layout()

        plt.savefig(output_dir / f"{quantity}_curve_comparison.pdf")
        plt.savefig(output_dir / f"{quantity}_curve_comparison.png", dpi=200)
        plt.close()

    np.savez(output_dir / "curve_comparison.npz", run_labels=np.asarray(run_labels), **payload)


def run_slug(run_label):
    return re.sub(r"[^A-Za-z0-9.+-]+", "_", run_label).strip("_")


def save_summary(output_dir, datasets, args, loss_summaries, quality, pt_max):
    lines = []
    for dataset in datasets:
        lines.append(f"{dataset['label']} input files:")
        lines.extend(f"  {path}" for path in dataset["input_paths"])
        lines.append("")

    if datasets:
        lines.append(f"True pT branch: {args.truth_branch}")
        lines.append(f"Prediction branch: {args.prediction_branch} (predicted pT = exp(output + center))")
        lines.append("Event weights: not used (every event counts 1)")
    for dataset in datasets:
        pt_true = dataset["pt_true"]
        pt_pred = dataset["pt_pred"]
        frac_true = stxs_fractions(pt_true)
        frac_pred = stxs_fractions(pt_pred)

        lines.append("")
        lines.append(f"{dataset['label']} events after filtering: {len(pt_true)}")
        lines.append(f"  target center: {dataset['center']:g} (label = log(max(pT, 1)) - center)")
        if dataset["split"] is not None:
            lines.append(f"  {SPLIT_BRANCH} values: {sorted(set(np.unique(dataset['split']).tolist()))}")
        lines.append(f"  mean pT:   true {np.mean(pt_true):8.2f} GeV, predicted {np.mean(pt_pred):8.2f} GeV")
        lines.append(f"  median pT: true {np.median(pt_true):8.2f} GeV, predicted {np.median(pt_pred):8.2f} GeV")
        lines.append(f"  highest pT: true {np.max(pt_true):8.1f} GeV, predicted {np.max(pt_pred):8.1f} GeV")
        lines.append(f"  events in the overflow bin (pT >= {pt_max:g} GeV): true {np.sum(pt_true >= pt_max)} "
                     f"({np.mean(pt_true >= pt_max):.2e}), predicted {np.sum(pt_pred >= pt_max)} "
                     f"({np.mean(pt_pred >= pt_max):.2e})")
        lines.append("  fraction of events per STXS bin (true -> predicted):")
        for name, f_true, f_pred in zip(BIN_NAMES, frac_true, frac_pred):
            lines.append(f"    {name:12s} {f_true:.4f} -> {f_pred:.4f}")

        migration = quality[dataset["label"]]["migration"]
        correlation = quality[dataset["label"]]["correlation"]
        lines.append("  migration matrix, cuts at the STXS edges on predicted pT (rows = true bin, row-normalized):")
        for name, row, count_row in zip(BIN_NAMES, migration["row_norm"], migration["counts"]):
            lines.append(f"    {name:12s} " + "  ".join(f"{v:.4f}" for v in row) + "   events: " + str(count_row.tolist()))
        lines.append("  purity (column-normalized diagonal): " + ", ".join(f"{v:.4f}" for v in np.diag(migration["col_norm"])))
        lines.append(f"  balanced accuracy: {migration['balanced_accuracy']:.4f}, accuracy: {migration['accuracy']:.4f}")
        lines.append(f"  correlation: Pearson (log pT) {correlation['pearson_log_pt']:.4f}, "
                     f"Pearson (pT) {correlation['pearson_pt']:.4f}, Spearman (rank) {correlation['spearman']:.4f}")
        if correlation["n_overflow"]:
            lines.append(f"  events in the overflow row/column of the correlation plot: {correlation['n_overflow']}")

    for run_label, (log_file, loss_summary) in loss_summaries.items():
        lines.append("")
        lines.append(f"Loss log ({run_label}): {log_file}")
        for key, value in loss_summary.items():
            lines.append(f"{key}: {value:.8g}")

    summary = "\n".join(lines)
    (output_dir / "summary.txt").write_text(summary + "\n")
    print(summary)


def check_split(datasets):
    for dataset in datasets:
        split = dataset["split"]
        if split is None:
            print(f"Note: no {SPLIT_BRANCH} branch in the {dataset['label']} files; cannot check which events they contain.")
            continue
        found = set(np.unique(split).tolist())
        expected = EXPECTED_SPLIT[dataset["label"]]
        if not found <= expected:
            raise ValueError(
                f"The {dataset['label']} predictions contain {SPLIT_BRANCH} values {sorted(found)}, "
                f"but should only contain {sorted(expected)}.\n"
                f"Files: {[str(path) for path in dataset['input_paths']]}\n"
                "The prediction job used the wrong test_time_selection in its data config "
                "(train dataset: event_mod5 != 4, held-out/test events: event_mod5 == 4)."
            )

    if len(datasets) == 2:
        first, second = datasets
        if len(first["pt_pred"]) == len(second["pt_pred"]) and np.array_equal(first["pt_pred"], second["pt_pred"]):
            raise ValueError("The Train and Held-out/test predictions are identical; they were made on the same events.")


def main():
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = []
    if args.input_dir or args.input_files:
        datasets.append(load_dataset("Held-out/test", args.input_dir, args.input_files, args.input_stem, args))
    if args.train_input_dir or args.train_input_files:
        datasets.append(
            load_dataset("Train", args.train_input_dir, args.train_input_files, args.train_input_stem, args)
        )

    check_split(datasets)

    if not datasets and not args.log_file:
        raise ValueError("No inputs were provided. Use prediction inputs and/or --log-file.")

    quality = {}
    if datasets:
        pt_max = overflow_edge(datasets, args.pt_max)
        corr_pt_max = args.corr_pt_max if args.corr_pt_max is not None else pt_max
        save_pt_distribution(output_dir, args.title, datasets, pt_max, args.bin_width)
        save_bias_resolution(output_dir, args.title, datasets)
        for dataset in datasets:
            quality[dataset["label"]] = {
                "migration": save_migration_matrix(output_dir, args.title, dataset),
                "correlation": save_correlation(output_dir, args.title, dataset, args.corr_bins, corr_pt_max),
            }

    loss_summaries = {}
    if args.log_file:
        run_labels = args.log_labels or [Path(log_file).stem for log_file in args.log_file]
        if len(set(run_slug(label) for label in run_labels)) != len(run_labels):
            raise ValueError("Run labels must be distinct; they name the per-run output folders.")
        for run_label, log_file in zip(run_labels, args.log_file):
            run_dir = output_dir if len(args.log_file) == 1 else output_dir / run_slug(run_label)
            run_dir.mkdir(parents=True, exist_ok=True)
            loss_summaries[run_label] = (log_file, save_loss_curve(run_dir, f"{args.title}, {run_label}", Path(log_file)))
        if len(args.log_file) > 1:
            save_loss_comparison(output_dir, args.title, args.log_file, run_labels)

    save_summary(output_dir, datasets, args, loss_summaries, quality,
                 overflow_edge(datasets, args.pt_max) if datasets else None)


if __name__ == "__main__":
    main()
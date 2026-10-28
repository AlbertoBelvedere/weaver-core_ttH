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
from sklearn.metrics import auc, confusion_matrix, roc_curve

try:
    import seaborn as sns
except ImportError:
    sns = None

try:
    import uproot
    from uproot.source.file import MemmapSource
except ImportError as exc:
    raise SystemExit(
        "Could not import uproot. Run this inside the Weaver environment, e.g.\n"
        "  source /work/inadalli/miniconda3/etc/profile.d/conda.sh\n"
        "  conda activate weaver\n"
    ) from exc


CLASS_NAMES = ["0-120 GeV", "120-200 GeV", ">200 GeV"]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Make STXS pT(H)-bin ROC, confusion-matrix, and loss-curve plots "
            "from Weaver prediction outputs and training logs."
        )
    )
    parser.add_argument(
        "--input-dir",
        help=(
            "Directory containing held-out/test Weaver prediction files. If set, the script looks for "
            "<input-stem>.root, e.g. result/stxs/<RUN_NAME>/predictions.root."
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
            "Training log(s) to parse for the loss and accuracy curves. This uses Train AvgLoss/AvgAcc and "
            "Eval AvgLoss/AvgAcc, so the second curve is the validation curve from training. "
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
        default="evaluation_output/stxs_plots",
        help="Output directory for plots and metric summaries.",
    )
    parser.add_argument("--tree", default="Events", help="Tree name in the prediction ROOT files.")
    parser.add_argument(
        "--label-branch",
        default="_label_",
        help="Truth branch. Expected convention: 0, 1, 2 = pT(H) bin, in the order of --score-branches.",
    )
    parser.add_argument(
        "--score-branches",
        nargs="+",
        default=["score_is_ptbin0", "score_is_ptbin1", "score_is_ptbin2"],
        help="Model score branches, one per pT(H) bin, in label order.",
    )
    parser.add_argument(
        "--title",
        default="STXS ParticleTransformer",
        help="Title prefix used in the plots.",
    )
    args = parser.parse_args()
    if len(args.score_branches) != len(CLASS_NAMES):
        parser.error(f"--score-branches needs {len(CLASS_NAMES)} entries, one per entry of CLASS_NAMES")
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
    needed_branches = [args.label_branch, *args.score_branches]
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

            data = tree.arrays(needed_branches, library="np")
            for branch in needed_branches:
                arrays[branch].append(np.asarray(data[branch]))

    return {branch: np.concatenate(values) for branch, values in arrays.items()}


def sanitize_inputs(data, args):
    n_classes = len(CLASS_NAMES)
    y_true = np.asarray(data[args.label_branch]).astype(np.int64)
    scores = np.stack([np.asarray(data[branch]).astype(np.float64) for branch in args.score_branches], axis=1)
    denominator = scores.sum(axis=1)
    valid = (
        np.isfinite(y_true)
        & np.all(np.isfinite(scores), axis=1)
        & np.isfinite(denominator)
        & (denominator > 0)
        & (y_true >= 0)
        & (y_true < n_classes)
    )

    n_removed = len(valid) - int(np.sum(valid))
    if n_removed:
        print(f"Removed {n_removed} event(s) with invalid labels or scores.")

    y_true = y_true[valid]
    scores = scores[valid]
    denominator = denominator[valid]

    scores = scores / denominator[:, None]

    labels_present = sorted(set(y_true.tolist()))
    if labels_present != list(range(n_classes)):
        raise ValueError(
            "Need events from every pT(H) bin to compute the ROC curves. "
            f"Labels present after filtering: {labels_present}"
        )

    return y_true, scores


def load_dataset(label, input_dir, input_files, stem, args):
    input_paths = expand_input_files(input_dir, input_files, stem)
    data = load_predictions(input_paths, args)
    y_true, scores = sanitize_inputs(data, args)
    return {
        "label": label,
        "input_paths": input_paths,
        "y_true": y_true,
        "scores": scores,
    }


def roc_one_vs_rest(dataset, k):
    # positives: events of bin k; negatives: events of all other bins; score: probability of bin k
    return roc_curve(dataset["y_true"] == k, dataset["scores"][:, k])


def roc_pairwise(dataset, i, j):
    # same definition as Weaver's roc_auc_score_matrix: only events of bins i and j, score p_j / (p_i + p_j)
    selected = (dataset["y_true"] == i) | (dataset["y_true"] == j)
    scores = dataset["scores"][selected]
    score_j = scores[:, j] / np.maximum(scores[:, i] + scores[:, j], 1e-6)
    return roc_curve(dataset["y_true"][selected] == j, score_j)


def save_roc(output_dir, title, datasets):
    n_classes = len(CLASS_NAMES)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    roc_results = {}

    curves = {
        "roc_curve": [
            (f"bin{k}_vs_rest", f"{CLASS_NAMES[k]} vs rest", k, lambda d, k=k: roc_one_vs_rest(d, k))
            for k in range(n_classes)
        ],
        "roc_curve_pairwise": [
            (f"bin{j}_vs_bin{i}", f"{CLASS_NAMES[j]} vs {CLASS_NAMES[i]}", p, lambda d, i=i, j=j: roc_pairwise(d, i, j))
            for p, (i, j) in enumerate((i, j) for i in range(n_classes) for j in range(i + 1, n_classes))
        ],
    }
    subtitles = {
        "roc_curve": "ROC, each bin vs the other bins",
        "roc_curve_pairwise": "ROC, pairs of bins",
    }

    for file_stem, curve_list in curves.items():
        roc_payload = {}
        plt.figure(figsize=(7, 6))
        for dataset in datasets:
            linestyle = "-" if dataset is datasets[0] else "--"
            key = dataset["label"].lower().replace(" ", "_").replace("/", "_")
            for curve_key, curve_name, color_index, compute in curve_list:
                fpr, tpr, thresholds = compute(dataset)
                roc_auc = auc(fpr, tpr)
                curve_label = f"{curve_name}, {dataset['label']} (AUC = {roc_auc:.4f})"
                plt.plot(fpr, tpr, linestyle, color=colors[color_index], linewidth=2.0, label=curve_label)

                roc_results[(dataset["label"], curve_name)] = roc_auc
                roc_payload[f"{key}_{curve_key}_fpr"] = fpr
                roc_payload[f"{key}_{curve_key}_tpr"] = tpr
                roc_payload[f"{key}_{curve_key}_thresholds"] = thresholds
                roc_payload[f"{key}_{curve_key}_auc"] = np.asarray(roc_auc)

        plt.plot([0, 1], [0, 1], linestyle="--", color="deeppink", alpha=0.7, label="Random")
        plt.xlim(0.0, 1.0)
        plt.ylim(0.0, 1.0)
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        plt.title(f"{title}: {subtitles[file_stem]}")
        plt.legend(loc="lower right", frameon=False, fontsize=8)
        plt.tight_layout()

        plt.savefig(output_dir / f"{file_stem}.pdf")
        plt.savefig(output_dir / f"{file_stem}.png", dpi=200)
        plt.close()

        np.savez(output_dir / f"{file_stem}.npz", class_names=np.asarray(CLASS_NAMES), **roc_payload)

    return roc_results


def save_confusion_matrix(output_dir, title, y_true, scores):
    y_pred = np.argmax(scores, axis=1)
    labels = list(range(len(CLASS_NAMES)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = np.divide(
        cm,
        cm.sum(axis=1, keepdims=True),
        out=np.zeros_like(cm, dtype=np.float64),
        where=cm.sum(axis=1, keepdims=True) != 0,
    )

    annotations = np.empty_like(cm_norm, dtype=object)
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            annotations[i, j] = f"{cm_norm[i, j]:.3f}"

    plt.figure(figsize=(6.5, 5.8))
    if sns is not None:
        sns.heatmap(
            cm_norm,
            annot=annotations,
            fmt="",
            xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES,
            cmap="Blues",
            vmin=0.0,
            vmax=1.0,
            cbar_kws={"label": "Row-normalized fraction"},
        )
    else:
        image = plt.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0)
        plt.colorbar(image, label="Row-normalized fraction")
        plt.xticks(np.arange(len(CLASS_NAMES)), CLASS_NAMES)
        plt.yticks(np.arange(len(CLASS_NAMES)), CLASS_NAMES)
        for i in range(cm_norm.shape[0]):
            for j in range(cm_norm.shape[1]):
                plt.text(j, i, annotations[i, j], ha="center", va="center")

    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(f"{title}: confusion matrix")
    plt.tight_layout()

    plt.savefig(output_dir / "confusion_matrix.pdf")
    plt.savefig(output_dir / "confusion_matrix.png", dpi=200)
    plt.close()

    np.savez(output_dir / "confusion_matrix.npz", cm=cm, cm_norm=cm_norm, rule="argmax")
    return cm, cm_norm


ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")
EPOCH_RE = re.compile(r"Epoch #(\d+) (training|validating)")
TRAIN_LOSS_RE = re.compile(r"Train AvgLoss:\s*([0-9.eE+-]+)")
EVAL_LOSS_RE = re.compile(r"Eval AvgLoss:\s*([0-9.eE+-]+)")
ACC_RE = re.compile(r"AvgAcc:\s*([0-9.eE+-]+)")


def parse_loss_log(log_file):
    current_epoch = None
    train_losses = {}
    val_losses = {}
    train_accs = {}
    val_accs = {}

    with open(log_file, "r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = ANSI_ESCAPE.sub("", raw_line)

            epoch_match = EPOCH_RE.search(line)
            if epoch_match:
                current_epoch = int(epoch_match.group(1))
                continue

            acc_match = ACC_RE.search(line)

            train_match = TRAIN_LOSS_RE.search(line)
            if train_match:
                epoch = current_epoch if current_epoch is not None else len(train_losses)
                train_losses[epoch] = float(train_match.group(1))
                if acc_match:
                    train_accs[epoch] = float(acc_match.group(1))
                continue

            val_match = EVAL_LOSS_RE.search(line)
            if val_match:
                epoch = current_epoch if current_epoch is not None else len(val_losses)
                val_losses[epoch] = float(val_match.group(1))
                if acc_match:
                    val_accs[epoch] = float(acc_match.group(1))

    if not train_losses and not val_losses:
        raise ValueError(f"Could not find Train AvgLoss or Eval AvgLoss entries in {log_file}")

    return train_losses, val_losses, train_accs, val_accs


def _sorted_curve(values_by_epoch):
    epochs = np.asarray(sorted(values_by_epoch), dtype=np.int64)
    values = np.asarray([values_by_epoch[epoch] for epoch in epochs], dtype=np.float64)
    return epochs, values


QUANTITIES = {
    # name: (index in parse_loss_log output for train, for validation, y-axis label, best = min or max)
    "loss": (0, 1, "Loss", np.argmin),
    "accuracy": (2, 3, "Accuracy", np.argmax),
}


def save_loss_curve(output_dir, title, log_file):
    curves = parse_loss_log(log_file)
    summary = {}

    for quantity, (train_index, val_index, ylabel, best) in QUANTITIES.items():
        train_values, val_values = curves[train_index], curves[val_index]

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
            summary[f"validation_{quantity}_best"] = values[best(values)]
            summary[f"validation_{quantity}_best_epoch"] = epochs[best(values)]

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

    for quantity, (train_index, val_index, ylabel, _) in QUANTITIES.items():
        plt.figure(figsize=(8, 5.5))
        for run, (run_label, curves) in enumerate(zip(run_labels, all_curves)):
            color = colors[run % len(colors)]
            key = run_slug(run_label)
            for index, linestyle, name in ((train_index, "--", "train"), (val_index, "-", "validation")):
                if curves[index]:
                    epochs, values = _sorted_curve(curves[index])
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


def save_summary(output_dir, datasets, args, roc_results, cm, cm_norm, loss_summaries):
    lines = []
    for dataset in datasets:
        lines.append(f"{dataset['label']} input files:")
        lines.extend(f"  {path}" for path in dataset["input_paths"])
        lines.append("")

    lines.append("")
    lines.append(f"Label branch: {args.label_branch}")
    lines.append("Event weights: not used (every event counts 1)")
    lines.append("Confusion matrix rule: predicted bin = bin with the highest score")
    for dataset in datasets:
        y_true = dataset["y_true"]
        scores = dataset["scores"]

        lines.append("")
        lines.append(f"{dataset['label']} events after filtering: {len(y_true)}")
        for label, name in enumerate(CLASS_NAMES):
            mask = y_true == label
            lines.append(
                f"{name}: events={int(np.sum(mask))}, "
                f"mean_own_bin_score={np.mean(scores[mask, label]):.6f}"
            )
        for (dataset_label, curve_name), roc_auc in roc_results.items():
            if dataset_label == dataset["label"]:
                lines.append(f"{dataset['label']} ROC AUC, {curve_name}: {roc_auc:.8f}")

    lines.append("")
    if cm is not None and cm_norm is not None:
        lines.append("Held-out/test confusion matrix:")
        lines.append(np.array2string(cm, precision=6))
        lines.append("")
        lines.append("Held-out/test row-normalized confusion matrix:")
        lines.append(np.array2string(cm_norm, precision=6))

    for run_label, (log_file, loss_summary) in loss_summaries.items():
        lines.append("")
        lines.append(f"Loss log ({run_label}): {log_file}")
        for key, value in loss_summary.items():
            lines.append(f"{key}: {value:.8g}")

    summary = "\n".join(lines)
    (output_dir / "summary.txt").write_text(summary + "\n")
    print(summary)


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

    if not datasets and not args.log_file:
        raise ValueError("No inputs were provided. Use prediction inputs for ROC/confusion and/or --log-file.")

    roc_results = {}
    cm = None
    cm_norm = None
    if datasets:
        roc_results = save_roc(output_dir, args.title, datasets)
        heldout = datasets[0]
        cm, cm_norm = save_confusion_matrix(
            output_dir,
            args.title,
            heldout["y_true"],
            heldout["scores"],
        )

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

    save_summary(output_dir, datasets, args, roc_results, cm, cm_norm, loss_summaries)


if __name__ == "__main__":
    main()
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
        "  source /work/abelvede/miniconda3/etc/profile.d/conda.sh\n"
        "  conda activate weaver\n"
    ) from exc


CLASS_NAMES = ["CP even", "CP odd"]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Make weighted CP-even/CP-odd ROC, confusion-matrix, and loss-curve plots "
            "from Weaver prediction outputs and training logs."
        )
    )
    parser.add_argument(
        "--input-dir",
        help=(
            "Directory containing held-out/test Weaver prediction files. If set, the script looks for "
            "predict_output_test_cp_even.root and predict_output_test_cp_odd.root."
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
        default="predict_output_test",
        help="Prediction-file stem used with --input-dir.",
    )
    parser.add_argument(
        "--train-input-dir",
        help=(
            "Optional directory containing train-split Weaver prediction files. If set, the script looks for "
            "predict_output_train_cp_even.root and predict_output_train_cp_odd.root, with a fallback to "
            "predict_output_test_cp_even.root and predict_output_test_cp_odd.root."
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
        default="predict_output_train",
        help="Prediction-file stem used with --train-input-dir.",
    )
    parser.add_argument(
        "--log-file",
        help=(
            "Training log to parse for the loss curve. This uses Train AvgLoss and Eval AvgLoss, "
            "so the second curve is the validation loss from training."
        ),
    )
    parser.add_argument(
        "--output",
        default="evaluation_output/cpv_part_2lss_plots",
        help="Output directory for plots and metric summaries.",
    )
    parser.add_argument("--tree", default="Events", help="Tree name in the prediction ROOT files.")
    parser.add_argument(
        "--label-branch",
        default="target_is_cpodd",
        help="Truth branch. Expected convention: 0 = CP even, 1 = CP odd.",
    )
    parser.add_argument(
        "--weight-branch",
        default="weight",
        help="Event-weight branch used for the weighted ROC and confusion matrix.",
    )
    parser.add_argument(
        "--signed-weights",
        action="store_true",
        help="Use signed weights. By default absolute weights are used.",
    )
    parser.add_argument(
        "--score-even-branch",
        default="score_is_cp_even",
        help="Model score branch for the CP-even class.",
    )
    parser.add_argument(
        "--score-odd-branch",
        default="score_is_cp_odd",
        help="Model score branch for the CP-odd class.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="CP-odd score threshold used for the confusion matrix.",
    )
    parser.add_argument(
        "--title",
        default="CPV ParticleTransformer",
        help="Title prefix used in the plots.",
    )
    return parser.parse_args()


def expand_input_files(input_dir, input_files, stem):
    paths = []

    if input_dir:
        input_path = Path(input_dir)
        stems = [stem]
        if stem != "predict_output_test":
            stems.append("predict_output_test")

        selected_paths = None
        for candidate_stem in stems:
            candidate_paths = [
                input_path / f"{candidate_stem}_cp_even.root",
                input_path / f"{candidate_stem}_cp_odd.root",
            ]
            if all(path.exists() for path in candidate_paths):
                selected_paths = candidate_paths
                break

        paths.extend(selected_paths if selected_paths is not None else [
            input_path / f"{stem}_cp_even.root",
            input_path / f"{stem}_cp_odd.root",
        ])

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
    needed_branches = [
        args.label_branch,
        args.weight_branch,
        args.score_even_branch,
        args.score_odd_branch,
    ]
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
    y_true = np.asarray(data[args.label_branch]).astype(np.int64)
    weights = np.asarray(data[args.weight_branch]).astype(np.float64)
    score_even = np.asarray(data[args.score_even_branch]).astype(np.float64)
    score_odd = np.asarray(data[args.score_odd_branch]).astype(np.float64)

    if not args.signed_weights:
        weights = np.abs(weights)

    denominator = score_even + score_odd
    valid = (
        np.isfinite(y_true)
        & np.isfinite(weights)
        & np.isfinite(score_even)
        & np.isfinite(score_odd)
        & np.isfinite(denominator)
        & (denominator > 0)
        & (weights > 0)
        & ((y_true == 0) | (y_true == 1))
    )

    n_removed = len(valid) - int(np.sum(valid))
    if n_removed:
        print(f"Removed {n_removed} event(s) with invalid labels, scores, or weights.")

    y_true = y_true[valid]
    weights = weights[valid]
    score_even = score_even[valid]
    score_odd = score_odd[valid]
    denominator = denominator[valid]

    score_cpodd = score_odd / denominator
    score_cpeven = score_even / denominator

    labels_present = sorted(set(y_true.tolist()))
    if labels_present != [0, 1]:
        raise ValueError(
            "Need both CP-even and CP-odd events to compute the ROC curve. "
            f"Labels present after filtering: {labels_present}"
        )

    return y_true, weights, score_cpeven, score_cpodd


def load_dataset(label, input_dir, input_files, stem, args):
    input_paths = expand_input_files(input_dir, input_files, stem)
    data = load_predictions(input_paths, args)
    y_true, weights, score_cpeven, score_cpodd = sanitize_inputs(data, args)
    return {
        "label": label,
        "input_paths": input_paths,
        "y_true": y_true,
        "weights": weights,
        "score_cpeven": score_cpeven,
        "score_cpodd": score_cpodd,
    }


def save_roc(output_dir, title, datasets):
    roc_results = {}
    roc_payload = {}

    plt.figure(figsize=(7, 6))
    for dataset in datasets:
        fpr, tpr, thresholds = roc_curve(
            dataset["y_true"],
            dataset["score_cpodd"],
            sample_weight=dataset["weights"],
        )
        roc_auc = auc(fpr, tpr)
        curve_label = f"{dataset['label']} (AUC = {roc_auc:.4f})"
        plt.plot(fpr, tpr, linewidth=2.0, label=curve_label)

        key = dataset["label"].lower().replace(" ", "_").replace("/", "_")
        roc_results[dataset["label"]] = roc_auc
        roc_payload[f"{key}_fpr"] = fpr
        roc_payload[f"{key}_tpr"] = tpr
        roc_payload[f"{key}_thresholds"] = thresholds
        roc_payload[f"{key}_auc"] = np.asarray(roc_auc)

    plt.plot([0, 1], [0, 1], linestyle="--", color="deeppink", alpha=0.7, label="Random")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(f"{title}: weighted ROC")
    plt.legend(loc="lower right", frameon=False)
    plt.tight_layout()

    plt.savefig(output_dir / "weighted_roc_curve.pdf")
    plt.savefig(output_dir / "weighted_roc_curve.png", dpi=200)
    plt.close()

    np.savez(output_dir / "weighted_roc_curve.npz", **roc_payload)
    return roc_results


def save_confusion_matrix(output_dir, title, y_true, weights, score_cpodd, threshold):
    y_pred = (score_cpodd >= threshold).astype(np.int64)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1], sample_weight=weights)
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
            cbar_kws={"label": "Row-normalized weighted fraction"},
        )
    else:
        image = plt.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0)
        plt.colorbar(image, label="Row-normalized weighted fraction")
        plt.xticks(np.arange(len(CLASS_NAMES)), CLASS_NAMES)
        plt.yticks(np.arange(len(CLASS_NAMES)), CLASS_NAMES)
        for i in range(cm_norm.shape[0]):
            for j in range(cm_norm.shape[1]):
                plt.text(j, i, annotations[i, j], ha="center", va="center")

    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(f"{title}: weighted confusion matrix")
    plt.tight_layout()

    plt.savefig(output_dir / "weighted_confusion_matrix.pdf")
    plt.savefig(output_dir / "weighted_confusion_matrix.png", dpi=200)
    plt.close()

    np.savez(output_dir / "weighted_confusion_matrix.npz", cm=cm, cm_norm=cm_norm, threshold=threshold)
    return cm, cm_norm


ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")
EPOCH_RE = re.compile(r"Epoch #(\d+) (training|validating)")
TRAIN_LOSS_RE = re.compile(r"Train AvgLoss:\s*([0-9.eE+-]+)")
EVAL_LOSS_RE = re.compile(r"Eval AvgLoss:\s*([0-9.eE+-]+)")


def parse_loss_log(log_file):
    current_epoch = None
    train_losses = {}
    val_losses = {}

    with open(log_file, "r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = ANSI_ESCAPE.sub("", raw_line)

            epoch_match = EPOCH_RE.search(line)
            if epoch_match:
                current_epoch = int(epoch_match.group(1))
                continue

            train_match = TRAIN_LOSS_RE.search(line)
            if train_match:
                epoch = current_epoch if current_epoch is not None else len(train_losses)
                train_losses[epoch] = float(train_match.group(1))
                continue

            val_match = EVAL_LOSS_RE.search(line)
            if val_match:
                epoch = current_epoch if current_epoch is not None else len(val_losses)
                val_losses[epoch] = float(val_match.group(1))

    if not train_losses and not val_losses:
        raise ValueError(f"Could not find Train AvgLoss or Eval AvgLoss entries in {log_file}")

    return train_losses, val_losses


def _sorted_curve(values_by_epoch):
    epochs = np.asarray(sorted(values_by_epoch), dtype=np.int64)
    values = np.asarray([values_by_epoch[epoch] for epoch in epochs], dtype=np.float64)
    return epochs, values


def save_loss_curve(output_dir, title, log_file):
    train_losses, val_losses = parse_loss_log(log_file)

    plt.figure(figsize=(7, 5.5))
    payload = {}
    summary = {}

    if train_losses:
        epochs, values = _sorted_curve(train_losses)
        plt.plot(epochs, values, marker="o", markersize=3, linewidth=1.8, label="Train")
        payload["train_epoch"] = epochs
        payload["train_loss"] = values
        summary["train_last"] = values[-1]
        summary["train_min"] = np.min(values)

    if val_losses:
        epochs, values = _sorted_curve(val_losses)
        plt.plot(epochs, values, marker="o", markersize=3, linewidth=1.8, label="Validation")
        payload["validation_epoch"] = epochs
        payload["validation_loss"] = values
        summary["validation_last"] = values[-1]
        summary["validation_min"] = np.min(values)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{title}: loss curve")
    plt.grid(alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()

    plt.savefig(output_dir / "loss_curve.pdf")
    plt.savefig(output_dir / "loss_curve.png", dpi=200)
    plt.close()

    payload["log_file"] = np.asarray(str(log_file))
    np.savez(output_dir / "loss_curve.npz", **payload)
    return summary


def save_summary(output_dir, datasets, args, roc_results, cm, cm_norm, loss_summary):
    lines = []
    for dataset in datasets:
        lines.append(f"{dataset['label']} input files:")
        lines.extend(f"  {path}" for path in dataset["input_paths"])
        lines.append("")

    lines.append("")
    lines.append(f"Label branch: {args.label_branch}")
    lines.append(f"Weight branch: {args.weight_branch}")
    lines.append(f"Weight mode: {'signed' if args.signed_weights else 'absolute'}")
    lines.append(f"Confusion threshold on CP-odd score: {args.threshold}")
    for dataset in datasets:
        y_true = dataset["y_true"]
        weights = dataset["weights"]
        score_cpodd = dataset["score_cpodd"]

        lines.append("")
        lines.append(f"{dataset['label']} events after filtering: {len(y_true)}")
        for label, name in enumerate(CLASS_NAMES):
            mask = y_true == label
            lines.append(
                f"{name}: events={int(np.sum(mask))}, "
                f"sum_weights={np.sum(weights[mask]):.8g}, "
                f"mean_cpodd_score={np.average(score_cpodd[mask], weights=weights[mask]):.6f}"
            )
        lines.append(f"{dataset['label']} weighted ROC AUC: {roc_results[dataset['label']]:.8f}")

    lines.append("")
    if cm is not None and cm_norm is not None:
        lines.append("Held-out/test weighted confusion matrix:")
        lines.append(np.array2string(cm, precision=6))
        lines.append("")
        lines.append("Held-out/test row-normalized weighted confusion matrix:")
        lines.append(np.array2string(cm_norm, precision=6))

    if loss_summary:
        lines.append("")
        lines.append(f"Loss log: {args.log_file}")
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
            heldout["weights"],
            heldout["score_cpodd"],
            args.threshold,
        )

    loss_summary = None
    if args.log_file:
        loss_summary = save_loss_curve(output_dir, args.title, Path(args.log_file))

    save_summary(output_dir, datasets, args, roc_results, cm, cm_norm, loss_summary)


if __name__ == "__main__":
    main()

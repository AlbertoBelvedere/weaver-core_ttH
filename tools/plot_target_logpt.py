#!/usr/bin/env python3
"""Plot the true log(pT(H)) after the Weaver-style standardization  x = clip((log(pT) - center) * scale, min, max).

Default: center 4, scale 1, clipped to [-5, 5] (the same transformation as [jet_logpt, 4, 1] in the data config).
log(pT) uses the same definition as the regression target: log(max(GenHiggs_pt, 1)).

Usage:
  python tools/plot_target_logpt.py
  python tools/plot_target_logpt.py --selection "event_mod5 < 3" --output plots/target_logpt
"""

import argparse
import glob
import os

os.environ.setdefault("MPLCONFIGDIR", f"/tmp/matplotlib-{os.environ.get('USER', 'user')}")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import uproot

DEFAULT_GLOB = "/work/abelvede/multilepton-analysis/neuralnetwork/snapshots/STXS/2lss_signal/2026-09-23/*/TTHtoNon2B.root"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--files", default=DEFAULT_GLOB, help="Input ROOT files (glob pattern).")
    parser.add_argument("--tree", default="Events")
    parser.add_argument("--branch", default="GenHiggs_pt", help="True pT(H) in GeV.")
    parser.add_argument("--selection", default=None,
                        help="Optional selection on event_mod5, e.g. 'event_mod5 < 3' (training events). Default: all events.")
    parser.add_argument("--center", type=float, default=4.0)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--clip-min", type=float, default=-5.0)
    parser.add_argument("--clip-max", type=float, default=5.0)
    parser.add_argument("--bin-width", type=float, default=0.02, help="Bin width in units of the transformed variable.")
    parser.add_argument("--output", default="evaluation_output/target_logpt", help="Output file name without extension.")
    return parser.parse_args()


def load(args):
    files = sorted(glob.glob(args.files))
    if not files:
        raise SystemExit(f"No file matches {args.files}")
    values = []
    for path in files:
        tree = uproot.open(path)[args.tree]
        branches = [args.branch] + (["event_mod5"] if args.selection else [])
        data = tree.arrays(branches, library="np")
        pt = data[args.branch].astype(np.float64)
        if args.selection:
            event_mod5 = data["event_mod5"]
            pt = pt[eval(args.selection, {"np": np}, {"event_mod5": event_mod5})]
        values.append(pt)
        print(f"{path}: {len(pt)} events")
    return np.concatenate(values)


def main():
    args = parse_args()
    pt = load(args)

    log_pt = np.log(np.maximum(pt, 1.0))  # same as the regression target
    transformed = (log_pt - args.center) * args.scale
    clipped = np.clip(transformed, args.clip_min, args.clip_max)

    n = len(clipped)
    below = np.mean(transformed < args.clip_min)
    above = np.mean(transformed > args.clip_max)

    edges = np.arange(args.clip_min, args.clip_max + args.bin_width, args.bin_width)
    counts, _ = np.histogram(clipped, bins=edges)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.stairs(counts / n, edges, color="black", linewidth=1.0)
    ax.axvline(0.0, color="grey", linestyle=":", linewidth=1.0)
    # STXS edges in the transformed variable
    for edge_gev, height in ((120.0, 0.97), (200.0, 0.90)):
        x = (np.log(edge_gev) - args.center) * args.scale
        ax.axvline(x, color="tab:red", linestyle="--", linewidth=1.0)
        ax.text(x, height, f" {edge_gev:.0f} GeV", color="tab:red", fontsize=8, va="top",
                transform=ax.get_xaxis_transform())
    ax.set_xlabel(f"clip((log(pT(H)/GeV) - {args.center:g}) x {args.scale:g}, {args.clip_min:g}, {args.clip_max:g})")
    ax.set_ylabel(f"Fraction of events per {args.bin_width:g}")
    ax.set_title("True log pT(H), standardized and clipped")
    ax.text(0.02, 0.97,
            f"events: {n}\nmean: {np.mean(clipped):.3f}\nstd: {np.std(clipped):.3f}\n"
            f"min: {clipped.min():.3f}, max: {clipped.max():.3f}\n"
            f"clipped below: {below:.2e}\nclipped above: {above:.2e}",
            transform=ax.transAxes, va="top", fontsize=9, bbox={"facecolor": "white", "alpha": 0.8})
    ax.grid(alpha=0.25)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fig.savefig(args.output + ".pdf", bbox_inches="tight")
    fig.savefig(args.output + ".png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    np.savez(args.output + ".npz", bin_edges=edges, counts=counts, center=args.center, scale=args.scale,
             clip_min=args.clip_min, clip_max=args.clip_max, fraction_below=below, fraction_above=above)

    print(f"\nEvents: {n}")
    print(f"Transformed: mean {np.mean(clipped):.4f}, std {np.std(clipped):.4f}, "
          f"min {clipped.min():.4f}, max {clipped.max():.4f}")
    print(f"Fraction clipped: below {below:.3e}, above {above:.3e}")
    print(f"Saved {args.output}.png/.pdf/.npz")


if __name__ == "__main__":
    main()
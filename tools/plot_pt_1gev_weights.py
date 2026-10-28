#!/usr/bin/env python3
"""True pT(H) in 1 GeV bins and the inverse-frequency weight of every bin, both on linear axes.

Bins: [0, 1), [1, 2), ... up to the highest true pT, so every event is in a bin (no overflow bin).
pT is floored at 1 GeV first, as in the regression label log(max(pT, 1)), so the bins match what the loss sees.
Weight of bin k:  w_k = N / (K * N_k)
    N = events, N_k = events in bin k, K = number of bins that contain events.
    Every non-empty bin then holds the same total weight N / K, and the mean weight per event is 1.

Cap (--cap 4 --cap-from 200): the first bin at or above --cap-from GeV whose weight reaches --cap, and every bin
after it, get the weight --cap. Bins below --cap-from are not capped, so the large weights at very low pT stay.
With the cap, every bin needs a weight (validation events can fall in bins that are empty in the training events):
    - empty bins at or after the cap start: --cap
    - empty bins before it: interpolated from the neighbouring filled bins
    - pT above the last bin: the weight of the last bin (the loss puts them in the last bin)
The overall scale of the weights does not matter for the loss (it divides by the sum of the weights in the batch),
only their ratios.

Outputs (in --output):
    pt_1gev_counts.png/.pdf    events per 1 GeV bin vs pT
    pt_1gev_weights.png/.pdf   weight per bin vs pT (with --cap: before and after the cap);
                               y axis cut at --weights-ymax (default: see below)
    pt_1gev_weights.npz        bin edges ("edges"), counts, weights used ("weights"), weights before the cap
    pt_1gev_weights.txt        one line per bin
With --save-weights <file.npz> the same .npz is also written there, for the loss:  -o weight_file "'<file.npz>'"

Default y range of the weight plot: up to 1.2 x the largest weight among the bins below the pT that contains
99% of the events. Bins above that (the sparse high-pT bins) go off the top of the plot; their number and the
largest weight are written on the plot and printed.

Usage:
    python tools/plot_pt_1gev_weights.py
    python tools/plot_pt_1gev_weights.py --weights-ymax 50 --output evaluation_output/pt_1gev
    python tools/plot_pt_1gev_weights.py --cap 4 --cap-from 200 --save-weights data/pt_weights_1gev_cap4.npz
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
STXS_EDGES = (120.0, 200.0)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--files", default=DEFAULT_GLOB, help="Input ROOT files (glob pattern).")
    parser.add_argument("--tree", default="Events")
    parser.add_argument("--branch", default="GenHiggs_pt", help="True pT(H) in GeV.")
    parser.add_argument("--selection", default="event_mod5 < 3",
                        help="Events used (default: the training events). Use 'all' for every event.")
    parser.add_argument("--bin-width", type=float, default=1.0, help="Bin width in GeV.")
    parser.add_argument("--weights-ymax", type=float, default=None,
                        help="Upper end of the y axis of the weight plot (default: automatic, see above).")
    parser.add_argument("--cap", type=float, default=None,
                        help="Weight at which the weights are frozen (e.g. 4). Default: no cap.")
    parser.add_argument("--cap-from", type=float, default=200.0,
                        help="The cap starts at the first bin at or above this pT (GeV) whose weight reaches --cap.")
    parser.add_argument("--save-weights", default=None,
                        help="Also write the weight .npz to this path (e.g. data/pt_weights_1gev_cap4.npz).")
    parser.add_argument("--output", default="evaluation_output/pt_1gev_weights", help="Output directory.")
    return parser.parse_args()


def load(args):
    files = sorted(glob.glob(args.files))
    if not files:
        raise SystemExit(f"No file matches {args.files}")
    use_selection = args.selection != "all"
    values = []
    for path in files:
        tree = uproot.open(path)[args.tree]
        data = tree.arrays([args.branch] + (["event_mod5"] if use_selection else []), library="np")
        pt = data[args.branch].astype(np.float64)
        if use_selection:
            pt = pt[eval(args.selection, {"np": np}, {"event_mod5": data["event_mod5"]})]
        values.append(pt)
        print(f"{path}: {len(pt)} events")
    return np.concatenate(values)


def draw_stxs_edges(ax):
    for edge in STXS_EDGES:
        ax.axvline(edge, color="grey", linestyle=":", linewidth=1.2)


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)
    pt = np.maximum(load(args), 1.0)  # floored at 1 GeV like the label log(max(pT, 1))
    n_events = len(pt)

    n_bins = int(np.floor(pt.max() / args.bin_width)) + 1  # the last bin contains the highest pT
    edges = np.arange(n_bins + 1) * args.bin_width
    counts, _ = np.histogram(pt, bins=edges)

    filled = counts > 0
    n_filled = int(np.sum(filled))
    weights = np.full(n_bins, np.nan)  # empty bins: no weight (gap in the plot)
    weights[filled] = n_events / (n_filled * counts[filled])
    raw_weights = weights.copy()  # before the cap; nan for empty bins
    centers = 0.5 * (edges[1:] + edges[:-1])

    cap_start = None
    if args.cap is not None:
        candidates = np.nonzero(filled & (edges[:-1] >= args.cap_from) & (np.nan_to_num(weights) >= args.cap))[0]
        if len(candidates) == 0:
            print(f"Warning: no bin at or above {args.cap_from:g} GeV reaches weight {args.cap:g}; no cap applied.")
        else:
            cap_start = int(candidates[0])
        # every bin gets a weight: empty bins before the cap start are interpolated from the filled neighbours
        before = np.arange(n_bins) < (cap_start if cap_start is not None else n_bins)
        fill_from = filled & before
        weights[before & ~filled] = np.interp(centers[before & ~filled], centers[fill_from], weights[fill_from])
        if cap_start is not None:
            weights[cap_start:] = args.cap
    event_weights = weights[np.minimum((pt / args.bin_width).astype(int), n_bins - 1)]

    # ---- 1) events per bin
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.stairs(counts, edges, color="black", linewidth=0.6)
    draw_stxs_edges(ax)
    ax.set_xlim(0.0, edges[-1])
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel("True pT(H) [GeV]")
    ax.set_ylabel(f"Events per {args.bin_width:g} GeV")
    ax.set_title(f"True pT(H), {n_events} events ({args.selection}), {n_bins} bins of {args.bin_width:g} GeV, "
                 f"{n_bins - n_filled} empty (grey dotted: STXS edges)", fontsize=10)
    ax.grid(alpha=0.25)
    fig.savefig(os.path.join(args.output, "pt_1gev_counts.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(args.output, "pt_1gev_counts.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---- 2) weight per bin
    pt_99 = np.quantile(pt, 0.99)
    if args.weights_ymax is not None:
        ymax = args.weights_ymax
    else:
        below = filled & (edges[:-1] < pt_99)
        ymax = 1.2 * np.nanmax(weights[below])
    n_above = int(np.sum(weights[filled] > ymax))
    first_above = edges[:-1][filled & (np.nan_to_num(weights) > ymax)]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    if args.cap is not None:
        ax.stairs(np.nan_to_num(raw_weights), edges, color="tab:orange", linewidth=0.6, alpha=0.4,
                  label="Inverse frequency (before the cap)")
        ax.stairs(weights, edges, color="tab:blue", linewidth=1.0,
                  label=f"Used: capped at {args.cap:g} from {edges[cap_start]:g} GeV" if cap_start is not None
                  else "Used (cap not reached)")
        if cap_start is not None:
            ax.axvline(edges[cap_start], color="tab:blue", linestyle="--", linewidth=0.8)
        ax.legend(frameon=False, fontsize=9, loc="upper left")
    else:
        ax.stairs(np.nan_to_num(weights), edges, color="tab:orange", linewidth=0.6)
    draw_stxs_edges(ax)
    ax.axhline(1.0, color="black", linewidth=0.8)
    ax.set_xlim(0.0, edges[-1])
    ax.set_ylim(0.0, ymax)
    ax.set_xlabel("True pT(H) [GeV]")
    ax.set_ylabel("Weight per event")
    ax.set_title(f"Inverse-frequency weight per {args.bin_width:g} GeV bin (black line: weight 1; "
                 f"empty bins drawn at 0 before the cap)", fontsize=10)
    if args.cap is not None:
        ymax = max(ymax, 1.5 * args.cap)
        n_above = int(np.sum(raw_weights[filled] > ymax))
        first_above = edges[:-1][filled & (np.nan_to_num(raw_weights) > ymax)]
    text = (f"y axis cut at {ymax:.3g}\n"
            f"{n_above} of {n_filled} filled bins above it"
            + (f" (first at {first_above[0]:g} GeV)" if n_above else "")
            + f"\nlargest weight before the cap: {np.nanmax(raw_weights):.4g}\n"
            + f"99% of the events below {pt_99:.0f} GeV")
    ax.text(0.98, 0.97, text, transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.85})
    ax.grid(alpha=0.25)
    fig.savefig(os.path.join(args.output, "pt_1gev_weights.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(args.output, "pt_1gev_weights.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---- numbers
    payload = dict(edges=edges, counts=counts, weights=np.nan_to_num(weights),
                   weights_before_cap=np.nan_to_num(raw_weights), selection=args.selection,
                   cap=np.nan if args.cap is None else args.cap,
                   cap_start_gev=np.nan if cap_start is None else edges[cap_start])
    np.savez(os.path.join(args.output, "pt_1gev_weights.npz"), **payload)
    if args.save_weights:
        os.makedirs(os.path.dirname(args.save_weights) or ".", exist_ok=True)
        np.savez(args.save_weights, **payload)
        print(f"Weights for the loss written to {args.save_weights}")
    with open(os.path.join(args.output, "pt_1gev_weights.txt"), "w") as handle:
        handle.write(f"{'bin [GeV]':>15s} {'events':>8s} {'weight':>12s}\n")
        for k in range(n_bins):
            weight = f"{weights[k]:12.4f}" + ("" if filled[k] else "  (empty bin)")
            handle.write(f"{edges[k]:7g} - {edges[k + 1]:<6g}{counts[k]:8d} {weight}\n")

    print(f"\nEvents: {n_events}, bins: {n_bins}, filled: {n_filled}, empty: {n_bins - n_filled}")
    empty_below_400 = int(np.sum(~filled & (edges[:-1] < 400.0)))
    print(f"Empty bins below 400 GeV: {empty_below_400}")
    print(f"Weights before the cap: smallest {np.nanmin(raw_weights):.4f}, largest {np.nanmax(raw_weights):.4g}")
    if cap_start is not None:
        print(f"Cap: weight {args.cap:g} from {edges[cap_start]:g} GeV on "
              f"({np.mean(pt >= edges[cap_start]):.4f} of the events)")
    print(f"Weights used: smallest {np.nanmin(weights):.4f}, largest {np.nanmax(weights):.4g}, "
          f"mean per event {event_weights.mean():.4f}")
    print(f"Weight plot: y axis cut at {ymax:.4g}, {n_above} filled bins above it")
    stxs_bin = np.digitize(pt, STXS_EDGES)
    for b, name in enumerate(("0-120 GeV", "120-200 GeV", ">200 GeV")):
        in_bin = stxs_bin == b
        print(f"STXS bin {name:12s}: fraction of events {in_bin.mean():.4f}, "
              f"fraction of total weight {event_weights[in_bin].sum() / event_weights.sum():.4f}")
    print(f"Saved in {args.output}")


if __name__ == "__main__":
    main()
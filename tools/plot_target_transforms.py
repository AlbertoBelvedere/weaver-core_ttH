#!/usr/bin/env python3
"""True pT(H) next to its logit transformation for several clipping values epsilon.

    u = clip(pT / p_max, epsilon, 1 - epsilon)
    t = logit(u) = log(u / (1 - u))

The clip keeps u inside (0, 1), so the logit is finite:
    - events with pT <= epsilon * p_max all get the lowest value  logit(epsilon)     = -log((1 - eps) / eps)
    - events with pT >= (1 - epsilon) * p_max all get the highest  logit(1 - epsilon) = +log((1 - eps) / eps)
These events pile up in the first / last bin of the logit distribution; their fractions are printed and written
on the plot.

Plots (in --output), each a separate figure:
    pt_distribution.png/.pdf          pT(H) distribution (linear axes), with p_max and epsilon * p_max marked
    logit_eps_<epsilon>.png/.pdf      logit(clip(pT / p_max, eps, 1 - eps)), one figure per epsilon, all with the
                                      same x range; the clipped events are the spikes at both ends

Usage:
    python tools/plot_target_transforms.py
    python tools/plot_target_transforms.py --p-max 500 --epsilons 1e-4 1e-3 1e-2 0.05
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
    parser.add_argument("--p-max", type=float, default=500.0, help="Scale of the logit: u = pT / p_max (GeV).")
    parser.add_argument("--epsilons", type=float, nargs="+", default=[1e-4, 1e-3, 1e-2, 5e-2],
                        help="Clipping values epsilon to compare.")
    parser.add_argument("--pt-bin-width", type=float, default=1.0, help="Bin width of the pT plot in GeV.")
    parser.add_argument("--pt-plot-max", type=float, default=800.0,
                        help="Upper end of the pT plot (GeV); events above are counted in the last bin.")
    parser.add_argument("--logit-bins", type=int, default=200, help="Number of bins of the logit plot.")
    parser.add_argument("--output", default="evaluation_output/target_transforms", help="Output directory.")
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


def logit_target(pt, p_max, epsilon):
    u = np.clip(pt / p_max, epsilon, 1.0 - epsilon)
    return np.log(u / (1.0 - u))


def save(fig, output, stem):
    fig.savefig(os.path.join(output, f"{stem}.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(output, f"{stem}.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot: {os.path.join(output, stem)}.png")


def plot_pt(pt, epsilons, colors, args):
    """pT(H) distribution, last bin = overflow, with p_max and the clipping points marked."""
    n = len(pt)
    edges = np.arange(0.0, args.pt_plot_max + args.pt_bin_width + 1e-9, args.pt_bin_width)
    counts, _ = np.histogram(np.minimum(pt, args.pt_plot_max + 0.5 * args.pt_bin_width), bins=edges)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.stairs(counts / n, edges, color="black", linewidth=0.8)
    for edge in STXS_EDGES:
        ax.axvline(edge, color="grey", linestyle=":", linewidth=1.2)
    ax.axvline(args.p_max, color="tab:red", linestyle="--", linewidth=1.0, label=f"p_max = {args.p_max:g} GeV")
    for index, eps in enumerate(epsilons):
        ax.axvline(eps * args.p_max, color=colors[index % len(colors)], linestyle="-.", linewidth=0.9,
                   label=f"eps = {eps:g}: clipped below {eps * args.p_max:.3g} GeV, above {(1 - eps) * args.p_max:.4g} GeV")
    ax.set_xlim(0.0, edges[-1])
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel("True pT(H) [GeV]")
    ax.set_ylabel(f"Fraction of events per {args.pt_bin_width:g} GeV")
    ax.set_title(f"True pT(H), {n} events ({args.selection})\n"
                 f"last bin: overflow >= {args.pt_plot_max:g} GeV; grey dotted: STXS edges 120, 200 GeV", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.grid(alpha=0.25)
    save(fig, args.output, "pt_distribution")


def plot_logit(pt, eps, color, logit_edges, args):
    """logit(clip(pT / p_max, eps, 1 - eps)) for one epsilon; returns the summary line."""
    n = len(pt)
    t = logit_target(pt, args.p_max, eps)
    bound = np.log((1.0 - eps) / eps)
    low = np.mean(pt <= eps * args.p_max)
    high = np.mean(pt >= (1.0 - eps) * args.p_max)
    counts, _ = np.histogram(t, bins=logit_edges)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.stairs(counts / n, logit_edges, color=color, linewidth=1.0)
    for edge in STXS_EDGES:
        ax.axvline(np.log(edge / args.p_max / (1.0 - edge / args.p_max)), color="grey", linestyle=":", linewidth=1.2)
    ax.text(0.02, 0.97,
            f"range +-{bound:.2f}\nmean {t.mean():.3f}\nstd {t.std():.3f}\n"
            f"clipped at the bottom (pT <= {eps * args.p_max:.3g} GeV): {low:.2%}\n"
            f"clipped at the top (pT >= {(1 - eps) * args.p_max:.4g} GeV): {high:.2%}",
            transform=ax.transAxes, va="top", fontsize=9, bbox={"facecolor": "white", "alpha": 0.85})
    ax.set_xlim(logit_edges[0], logit_edges[-1])
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel(f"logit(clip(pT / {args.p_max:g} GeV, {eps:g}, {1 - eps:g}))")
    ax.set_ylabel(f"Fraction of events per bin ({args.logit_bins} bins)")
    ax.set_title(f"Logit target, p_max = {args.p_max:g} GeV, epsilon = {eps:g}, {n} events ({args.selection})\n"
                 "grey dotted: STXS edges 120, 200 GeV", fontsize=10)
    ax.grid(alpha=0.25)
    save(fig, args.output, f"logit_eps_{eps:g}")

    return (f"eps = {eps:<8g} range [{-bound:7.3f}, {bound:6.3f}]  mean {t.mean():7.4f}  std {t.std():6.4f}  "
            f"clipped below ({eps * args.p_max:.3g} GeV) {low:.4%}, above ({(1 - eps) * args.p_max:.4g} GeV) {high:.4%}")


def main():
    args = parse_args()
    if any(not 0.0 < eps < 0.5 for eps in args.epsilons):
        raise SystemExit("Every epsilon must be between 0 and 0.5")
    os.makedirs(args.output, exist_ok=True)
    pt = load(args)
    epsilons = sorted(args.epsilons)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    print(f"\nEvents: {len(pt)}, p_max = {args.p_max:g} GeV, highest pT {pt.max():.1f} GeV")

    plot_pt(pt, epsilons, colors, args)

    # same bins for every epsilon, covering the widest range (smallest epsilon), so the figures can be compared
    bound = np.log((1.0 - epsilons[0]) / epsilons[0])
    logit_edges = np.linspace(-bound - 1e-6, bound + 1e-6, args.logit_bins + 1)
    lines = [plot_logit(pt, eps, colors[index % len(colors)], logit_edges, args)
             for index, eps in enumerate(epsilons)]

    print("\n".join(lines))
    print(f"Saved in {args.output}")


if __name__ == "__main__":
    main()
"""Shared chart style and primitives for the diagnosis notebooks.

One module so the four notebooks are consistent by construction rather than by
discipline, and so a fix lands everywhere at once.

Three rules this encodes, learned the hard way from a first pass that produced
unreadable charts:

1. **Colour follows the job, not the variable you happen to have.** Magnitude gets a
   single-hue sequential ramp; only genuinely signed quantities get the diverging
   pair. A diverging scale on a variable that averages near zero paints every mark
   the same pale midtone and looks like a bug.
2. **Aggregate the time axis before plotting it.** 126 monthly columns of a noisy
   series is a candy-stripe. Quarterly means show the structure that is actually
   there.
3. **The title is the claim.** "Attention churns on a nearly flat weight vector" is a
   title; "attention by input over time" is a filename.

Palette is the validated default set (`scripts/validate_palette.js`): categorical
order blue / orange / aqua / yellow / magenta, sequential blue, diverging blue-red
with a grey midpoint.
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

# ── tokens ──────────────────────────────────────────────────────────────────
SURFACE = "#fcfcfb"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
POS, NEG, MID = "#2a78d6", "#e34948", "#f0efec"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
OTHER = "#b9b8b0"
BAND = "#f6f5f1"

SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#e8f0fb", "#9ec5f4", POS, "#184f95"])
DIV = LinearSegmentedColormap.from_list("div_br", [NEG, "#f2c3c2", MID, "#bcd6f6", POS])


def use_style() -> None:
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "savefig.bbox": "tight",
        "font.family": "sans-serif", "font.size": 10.5,
        "text.color": INK, "axes.labelcolor": INK2, "axes.labelsize": 9.5,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "grid.color": GRID, "grid.linewidth": 0.6, "axes.grid": False,
        "figure.dpi": 130, "axes.titlesize": 11.5, "axes.titleweight": "600",
        "axes.titlelocation": "left", "axes.titlepad": 14,
        "legend.frameon": False, "legend.fontsize": 9,
    })


def title(ax, claim: str, detail: str = "") -> None:
    """The claim as the title, the reading instruction beneath it in muted ink.

    The pad is set here rather than in rcParams because it has to make room for the
    detail line; a fixed global pad put the two on top of each other.
    """
    ax.set_title(claim, color=INK, loc="left", pad=34 if detail else 14)
    if detail:
        ax.annotate(detail, xy=(0, 1.0), xycoords="axes fraction",
                    xytext=(0, 11), textcoords="offset points",
                    fontsize=9, color=MUTED, va="bottom", annotation_clip=False)


def paired_rank(labels, left, right, *, left_label, right_label,
                left_ref=None, right_ref=None, figsize=(11.5, 5.4), sort_by="left"):
    """Two ranked panels sharing one set of row labels.

    Replaces a scatter when the point labels are long enough to collide — which, with
    eleven analyst names, they always are. Same two variables, zero overlap, and the
    ranking is readable directly instead of being inferred from position.
    """
    order = np.argsort(left if sort_by == "left" else right)
    labels = [labels[i] for i in order]
    left = np.asarray(left)[order]
    right = np.asarray(right)[order]
    y = np.arange(len(labels))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=figsize, sharey=True,
                                 gridspec_kw={"wspace": 0.06})
    a1.barh(y, left, color=seq_colors(left), height=0.62, zorder=3,
            edgecolor=SURFACE, linewidth=0.9)
    for yi, v in zip(y, left):
        a1.text(v + max(left) * 0.015, yi, f"{v:.2f}", va="center",
                fontsize=8.8, color=INK2)
    a1.set_yticks(y); a1.set_yticklabels(labels)
    a1.set_xlim(0, max(left) * 1.22); a1.set_xlabel(left_label)
    if left_ref is not None:
        a1.axvline(left_ref, color=MUTED, lw=1.1, ls=(0, (5, 3)), zorder=4)
    grid(a1, "x")

    a2.hlines(y, 0, right, color=GRID, lw=1.4, zorder=2)
    a2.scatter(right, y, s=70, color=seq_colors(right), zorder=3,
               edgecolor=SURFACE, linewidth=1.2)
    for yi, v in zip(y, right):
        a2.text(v + max(right) * 0.02, yi, f"{v:.0%}", va="center",
                fontsize=8.8, color=INK2)
    a2.set_xlim(0, max(right) * 1.2); a2.set_xlabel(right_label)
    if right_ref is not None:
        a2.axvline(right_ref, color=MUTED, lw=1.1, ls=(0, (5, 3)), zorder=4)
    a2.spines["left"].set_visible(False); a2.tick_params(left=False)
    grid(a2, "x")
    return fig, (a1, a2)


def grid(ax, axis: str = "both") -> None:
    ax.grid(axis=axis, zorder=0)
    ax.set_axisbelow(True)


def seq_colors(values, lo=None, hi=None, floor=0.22):
    """Sequential fills for a magnitude. Floor keeps the lightest mark visible."""
    v = np.asarray(values, dtype=float)
    lo = np.nanmin(v) if lo is None else lo
    hi = np.nanmax(v) if hi is None else hi
    t = np.zeros_like(v) if hi == lo else (v - lo) / (hi - lo)
    return [SEQ(floor + (1 - floor) * x) for x in np.clip(t, 0, 1)]


def ranked_bar(ax, labels, values, *, second=None, second_label="",
               xlabel="", fmt="{:.2f}", xmax=None, refs=()):
    """Horizontal ranked bars, sequential by magnitude, value labelled in place.

    ``second`` overlays a dot per row on a shared 0-1 axis — used for "share of
    meetings this input led", which is a different quantity from the bar and must not
    be encoded as bar length.
    """
    y = np.arange(len(labels))
    ax.barh(y, values, color=seq_colors(values), height=0.6, zorder=3,
            edgecolor=SURFACE, linewidth=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    hi = xmax if xmax is not None else float(np.nanmax(values)) * 1.32
    for yi, v in zip(y, values):
        ax.text(v + hi * 0.012, yi, fmt.format(v), va="center", fontsize=8.8, color=INK2)
    if second is not None:
        ax.scatter(np.asarray(second) * hi, y, s=30, facecolor=SURFACE,
                   edgecolor=INK, linewidth=1.2, zorder=5)
        ax.annotate(second_label, xy=(1.0, 1.0), xycoords="axes fraction",
                    xytext=(-2, 8), textcoords="offset points", ha="right",
                    fontsize=8.5, color=INK2, annotation_clip=False)
    for x, lab in refs:
        ax.axvline(x, color=MUTED, lw=0.9, ls=(0, (4, 3)), zorder=2)
        ax.annotate(lab, xy=(x, 1.0), xycoords=("data", "axes fraction"),
                    xytext=(3, 3), textcoords="offset points", fontsize=8,
                    color=MUTED, annotation_clip=False)
    ax.set_xlim(0, hi)
    ax.set_xlabel(xlabel)
    grid(ax, "x")


def quarterly(df: pd.DataFrame) -> pd.DataFrame:
    """Mean by calendar quarter. The single biggest legibility win on these panels."""
    if df.empty:
        return df
    out = df.copy()
    out.index = pd.PeriodIndex(pd.to_datetime(out.index), freq="Q")
    return out.groupby(level=0).mean()


def heat(ax, M: pd.DataFrame, *, diverging=True, vmax=None, cbar_label=""):
    """Heatmap of a (row x period) frame, with year ticks and regime markers."""
    vmax = vmax or float(np.nanmax(np.abs(M.values))) or 1.0
    ax.set_facecolor("#e4e2da")            # unranked / missing shows through
    im = ax.imshow(M.values, aspect="auto", interpolation="nearest",
                   cmap=DIV if diverging else SEQ,
                   norm=TwoSlopeNorm(0, -vmax, vmax) if diverging else None,
                   vmin=None if diverging else 0, vmax=None if diverging else vmax)
    ax.set_yticks(range(len(M.index)))
    ax.set_yticklabels(M.index, fontsize=9)
    yrs = [str(p)[:4] for p in M.columns]
    ticks = [i for i in range(1, len(yrs)) if yrs[i] != yrs[i - 1] and int(yrs[i]) % 2 == 0]
    ax.set_xticks(ticks)
    ax.set_xticklabels([yrs[i] for i in ticks])
    ax.set_xticks(np.arange(-0.5, len(yrs), 1), minor=True)
    for sp in ax.spines.values():
        sp.set_visible(False)
    return im


def mark_regimes(ax, periods, breaks=(("2020Q1", "COVID"), ("2022Q1", "hiking")),
                 top=True):
    """Vertical rules at the regime breaks, labelled once above the axes."""
    idx = [str(p) for p in periods]
    for key, lab in breaks:
        if key in idx:
            x = idx.index(key)
            ax.axvline(x, color=INK, lw=1.0, ls=(0, (3, 2)), alpha=0.5, zorder=6)
            if top:
                ax.annotate(lab, xy=(x, 1.0), xycoords=("data", "axes fraction"),
                            xytext=(4, 4), textcoords="offset points",
                            fontsize=8.5, color=INK2, annotation_clip=False)


def label_last(ax, series, color, text=None, dx=6, fontsize=9):
    """Direct-label a line at its right end — replaces a legend for <=5 series."""
    s = series.dropna()
    if s.empty:
        return
    ax.annotate(text or s.name, xy=(len(s) - 1, s.iloc[-1]),
                xytext=(dx, 0), textcoords="offset points",
                fontsize=fontsize, color=color, va="center", annotation_clip=False)


def repel(ax, xs, ys, labels, *, fontsize=8.6, dy=13, color=INK):
    """Cheap vertical de-collision for scatter labels.

    Sorts by x, then pushes each label up or down until it clears the previous one in
    display space. Not a full force simulation — enough to stop 11 analyst names
    stacking on top of each other, which was the actual failure.
    """
    order = np.argsort(xs)
    placed = []
    for i in order:
        x, y = xs[i], ys[i]
        off = dy
        for px, po in placed:
            if abs(x - px) < (max(xs) - min(xs)) * 0.13:
                off = -dy if po > 0 else dy
                if any(abs(x - qx) < (max(xs) - min(xs)) * 0.13 and qo == off
                       for qx, qo in placed):
                    off = off + np.sign(off) * dy
        ax.annotate(labels[i], (x, y), textcoords="offset points",
                    xytext=(0, off), ha="center", fontsize=fontsize, color=color,
                    va="bottom" if off > 0 else "top")
        placed.append((x, off))


def footnote(fig, text):
    fig.text(0.005, -0.02, text, fontsize=8.5, color=MUTED, va="top", ha="left")

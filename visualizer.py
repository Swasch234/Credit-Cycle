"""
visualizer.py — Publication-quality charts for the Credit Cycle project.

Charts produced
───────────────
  01_credit_pressure_index.png   — CPI with NBER shading + signal thresholds
  02_cac_rolling.png             — Credit Absorption Coefficient over time
  03_component_dashboard.png     — All five indicators in one dashboard
  04_regime_detection.png        — 3-panel regime chart (colored CPI, probs, timeline)
  05_impulse_responses.png       — IRFs under each regime (the VAR-analogy chart)
  06_early_warning_signals.png   — EWS traffic light with event outcomes
  07_roc_and_lead_time.png       — ROC curve + lead time distribution
"""

import os
import logging

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

from config import (
    EWS_CRISIS_THRESHOLD,
    EWS_SIGNAL_THRESHOLD,
    FIG_DPI,
    OUTPUT_DIR,
)
from regime_model import REGIME_COLORS

os.makedirs(OUTPUT_DIR, exist_ok=True)
logger = logging.getLogger(__name__)

RECESSION_COLOR = "#d5d5d5"
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def _save(fig: plt.Figure, filename: str) -> None:
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    print(f"  ✓ Saved → {path}")


def _shade_recessions(ax: plt.Axes, recession: pd.Series | None) -> None:
    if recession is None:
        return
    in_rec, start = False, None
    for date, val in recession.items():
        if val == 1 and not in_rec:
            in_rec, start = True, date
        elif val == 0 and in_rec:
            ax.axvspan(start, date, color=RECESSION_COLOR, alpha=0.5, zorder=0)
            in_rec = False
    if in_rec and start is not None:
        ax.axvspan(start, recession.index[-1], color=RECESSION_COLOR, alpha=0.5, zorder=0)


# ── Chart 1: Credit Pressure Index ────────────────────────────────────────────

def plot_cpi(
        cpi: pd.Series,
        recession: pd.Series | None = None,
        save: bool = True,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.suptitle(
        "Composite Credit Pressure Index (CPI)\n"
        "Higher = more credit stress  |  Shaded = NBER recessions",
        fontsize=12, fontweight="bold",
    )
    _shade_recessions(ax, recession)

    ax.plot(cpi.index, cpi.values, color="#2c3e50", linewidth=1.5, zorder=3)
    ax.fill_between(cpi.index, cpi, 0,
                    where=cpi > 0, color="#e74c3c", alpha=0.15, zorder=2)
    ax.fill_between(cpi.index, cpi, 0,
                    where=cpi <= 0, color="#27ae60", alpha=0.15, zorder=2)

    # Reference lines
    ax.axhline(0, color="grey", linewidth=0.7, linestyle="--")
    ax.axhline(1.5, color="#e74c3c", linewidth=0.8, linestyle=":",
               alpha=0.8, label="Crisis threshold (+1.5σ)")
    ax.axhline(-1.0, color="#27ae60", linewidth=0.8, linestyle=":",
               alpha=0.8, label="Low-stress threshold (−1σ)")

    ax.set_ylabel("Standard Deviations", fontsize=9)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    if save:
        _save(fig, "01_credit_pressure_index.png")
    return fig


# ── Chart 2: Credit Absorption Coefficient ────────────────────────────────────

def plot_cac(
        cac_df: pd.DataFrame,
        recession: pd.Series | None = None,
        save: bool = True,
) -> plt.Figure:
    """
    Plot the rolling Credit Absorption Coefficient (β) with ±1 std error bands.
    This is the signature chart bridging Shan's FX autonomy work to consumer credit.
    """
    if cac_df.empty or "cac_beta" not in cac_df.columns:
        logger.warning("CAC data not available — skipping chart.")
        return plt.figure()

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(
        "Credit Absorption Coefficient (CAC) — Rolling 36-Month Estimate\n"
        "Methodology extension of Xue & Willett (2024) offset-coefficient framework\n"
        "β > 0: households absorb income shocks via revolving credit (pressure builds)",
        fontsize=11, fontweight="bold",
    )

    # Top panel: β with confidence band
    ax = axes[0]
    _shade_recessions(ax, recession)
    beta = cac_df["cac_beta"]
    stderr = cac_df.get("cac_stderr", pd.Series(np.nan, index=cac_df.index))

    ax.plot(beta.index, beta.values, color="#2980b9", linewidth=1.6, label="CAC (β)")
    if stderr.notna().any():
        upper = beta + 1.96 * stderr
        lower = beta - 1.96 * stderr
        ax.fill_between(beta.index, lower, upper, color="#2980b9", alpha=0.15,
                        label="95% confidence band")

    ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
    ax.set_ylabel("β coefficient", fontsize=9)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("Credit Absorption Coefficient (β)", fontsize=9, loc="left")
    ax.grid(axis="y", alpha=0.25)

    # Bottom panel: R² (how well income shocks explain credit growth)
    ax2 = axes[1]
    _shade_recessions(ax2, recession)
    r2 = cac_df.get("cac_r2", pd.Series(dtype=float))
    if r2.notna().any():
        ax2.plot(r2.index, r2.values, color="#8e44ad", linewidth=1.3,
                 label="Rolling R²")
        ax2.fill_between(r2.index, 0, r2.values, color="#8e44ad", alpha=0.12)
    ax2.set_ylabel("R²", fontsize=9)
    ax2.set_ylim(0, 1)
    ax2.legend(fontsize=8, loc="upper left")
    ax2.set_title("Model Fit (R²): how much of credit growth is explained by income shocks",
                  fontsize=9, loc="left")
    ax2.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    if save:
        _save(fig, "02_cac_rolling.png")
    return fig


# ── Chart 3: Component Dashboard ──────────────────────────────────────────────

def plot_component_dashboard(
        components: pd.DataFrame,
        recession: pd.Series | None = None,
        save: bool = True,
) -> plt.Figure:
    """Grid of small multiples — one panel per credit indicator."""
    cols = [c for c in components.columns if components[c].notna().any()]
    n = len(cols)
    if n == 0:
        return plt.figure()

    ncols = 2
    nrows = (n + 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.5 * nrows), sharex=True)
    fig.suptitle("Credit Pressure Indicator Dashboard", fontsize=13, fontweight="bold")

    axes_flat = axes.ravel() if n > 1 else [axes]
    colors = plt.cm.tab10(np.linspace(0, 1, n))

    for i, col in enumerate(cols):
        ax = axes_flat[i]
        _shade_recessions(ax, recession)
        s = components[col].dropna()
        ax.plot(s.index, s.values, color=colors[i], linewidth=1.3)
        ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
        ax.fill_between(s.index, s, 0, where=s > 0, alpha=0.12, color=colors[i])
        ax.set_title(col.replace("_", " ").title(), fontsize=9)
        ax.grid(axis="y", alpha=0.2)

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout()
    if save:
        _save(fig, "03_component_dashboard.png")
    return fig


# ── Chart 4: Regime Detection ─────────────────────────────────────────────────

def plot_regime_detection(
        cpi: pd.Series,
        smoothed_probs: pd.DataFrame,
        regime_series: pd.Series,
        recession: pd.Series | None = None,
        save: bool = True,
) -> plt.Figure:
    fig = plt.figure(figsize=(14, 11))
    gs = GridSpec(3, 1, figure=fig, hspace=0.42, height_ratios=[2, 2, 1])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
    fig.suptitle("Markov-Switching Regime Detection — Credit Cycle", fontsize=13, fontweight="bold")

    # Panel 1: CPI coloured by regime
    # Align cpi to regime_series index (MS-AR drops first obs due to AR lag)
    cpi_aligned = cpi.reindex(regime_series.index)
    _shade_recessions(ax1, recession)
    ax1.plot(cpi.index, cpi.values, color="black", linewidth=0.5, alpha=0.3, zorder=1)
    for regime, color in REGIME_COLORS.items():
        mask = regime_series == regime
        ax1.scatter(cpi_aligned[mask].index, cpi_aligned[mask].values, color=color, s=8,
                    zorder=3, label=regime)
    ax1.axhline(0, color="grey", linewidth=0.5, linestyle="--")
    ax1.set_ylabel("CPI (Std Devs)", fontsize=9)
    ax1.set_title("Credit Pressure Index — coloured by most likely regime", fontsize=9, loc="left")
    ax1.legend(fontsize=8)
    ax1.grid(axis="y", alpha=0.25)

    # Panel 2: Smoothed probabilities
    for col in ["Expansion", "Strain", "Crisis"]:
        if col in smoothed_probs.columns:
            ax2.plot(smoothed_probs.index, smoothed_probs[col],
                     color=REGIME_COLORS[col], linewidth=1.3, label=f"P({col})")
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("Probability", fontsize=9)
    ax2.set_title("Smoothed Regime Probabilities  P(regime | all data)", fontsize=9, loc="left")
    ax2.legend(fontsize=8, loc="upper left")
    ax2.grid(axis="y", alpha=0.25)

    # Panel 3: Timeline
    for regime, color in REGIME_COLORS.items():
        mask = regime_series == regime
        ax3.fill_between(regime_series.index, 0, 1,
                         where=mask, color=color, alpha=0.7, label=regime)
    ax3.set_yticks([])
    ax3.set_title("Regime Timeline", fontsize=9, loc="left")
    ax3.legend(fontsize=8, loc="upper left", ncol=3)

    plt.tight_layout()
    if save:
        _save(fig, "04_regime_detection.png")
    return fig


# ── Chart 5: Impulse Response Functions by Regime ─────────────────────────────

def plot_impulse_responses(
        irf_df: pd.DataFrame,
        save: bool = True,
) -> plt.Figure:
    """
    Plot regime-conditional IRFs — the key chart connecting this work to
    Shan's VAR-based impulse response analysis in the FX/monetary context.
    Shows how a credit shock propagates differently in each regime.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle(
        "Regime-Conditional Impulse Response Functions\n"
        "Response of Credit Pressure Index to a 1σ shock, by regime\n"
        "Methodology: Markov-Switching AR extension of Xue & Willett (2024) VAR approach",
        fontsize=11, fontweight="bold",
    )

    styles = {"Expansion": "-", "Strain": "--", "Crisis": ":"}
    for col in irf_df.columns:
        color = REGIME_COLORS.get(col, "grey")
        ls = styles.get(col, "-")
        ax.plot(irf_df.index, irf_df[col], color=color, linewidth=2.0,
                linestyle=ls, label=col)

    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xlabel("Months after shock", fontsize=9)
    ax.set_ylabel("CPI response (std devs)", fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    ax.annotate(
        "Steeper decline = faster mean-reversion\nFlatter/rising = shock amplification",
        xy=(0.55, 0.85), xycoords="axes fraction", fontsize=8, color="grey",
    )
    plt.tight_layout()
    if save:
        _save(fig, "05_impulse_responses.png")
    return fig


# ── Chart 6: Early Warning Signals ────────────────────────────────────────────

def plot_ews_signals(
        signals: pd.DataFrame,
        events: pd.DataFrame,
        recession: pd.Series | None = None,
        save: bool = True,
) -> plt.Figure:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(
        "Early Warning System — Credit Cycle Signals\n"
        "Green/Yellow/Red = EWS alert level  |  Bars = credit event outcomes",
        fontsize=12, fontweight="bold",
    )

    # Top: P(Stress or Crisis) with thresholds
    _shade_recessions(ax1, recession)
    p = signals["p_stress_or_crisis"]
    ax1.plot(p.index, p.values, color="#2c3e50", linewidth=1.4, zorder=3)
    ax1.axhline(EWS_SIGNAL_THRESHOLD, color="#f39c12", linewidth=1.0, linestyle="--",
                label=f"Yellow threshold ({EWS_SIGNAL_THRESHOLD})")
    ax1.axhline(EWS_CRISIS_THRESHOLD, color="#e74c3c", linewidth=1.0, linestyle="--",
                label=f"Red threshold ({EWS_CRISIS_THRESHOLD})")
    ax1.fill_between(p.index, p.values, EWS_CRISIS_THRESHOLD,
                     where=p.values > EWS_CRISIS_THRESHOLD, color="#e74c3c", alpha=0.25)
    ax1.fill_between(p.index, p.values, EWS_SIGNAL_THRESHOLD,
                     where=(p.values > EWS_SIGNAL_THRESHOLD) & (p.values <= EWS_CRISIS_THRESHOLD),
                     color="#f39c12", alpha=0.20)
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("P(Strain or Crisis)", fontsize=9)
    ax1.legend(fontsize=8, loc="upper left")
    ax1.grid(axis="y", alpha=0.25)
    ax1.set_title("EWS Signal Probability", fontsize=9, loc="left")

    # Bottom: signal colour bars
    _shade_recessions(ax2, recession)
    color_map = {"GREEN": "#27ae60", "YELLOW": "#f39c12", "RED": "#e74c3c"}
    for color_label, color_val in color_map.items():
        mask = signals["signal_color"] == color_label
        ax2.bar(signals.index[mask], 1, width=25,
                color=color_val, alpha=0.7, label=color_label)

    # Overlay event markers
    if "event_either" in events.columns:
        event_dates = events.index[events["event_either"] == 1]
        ax2.scatter(event_dates, [0.5] * len(event_dates),
                    marker="x", color="black", s=25, zorder=5, label="Credit event")

    ax2.set_yticks([])
    ax2.set_title("EWS Alert Level (Green / Yellow / Red)", fontsize=9, loc="left")
    ax2.legend(fontsize=8, loc="upper left", ncol=4)

    plt.tight_layout()
    if save:
        _save(fig, "06_early_warning_signals.png")
    return fig


# ── Chart 7: ROC Curve + Lead-Time Distribution ───────────────────────────────

def plot_roc_and_lead_time(
        fpr: np.ndarray,
        tpr: np.ndarray,
        auroc: float,
        lead_time_df: pd.DataFrame,
        save: bool = True,
) -> plt.Figure:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("EWS Evaluation: ROC Curve & Lead-Time Distribution",
                 fontsize=12, fontweight="bold")

    # ROC curve
    ax1.plot(fpr, tpr, color="#2980b9", linewidth=2.0,
             label=f"EWS  (AUROC = {auroc:.3f})")
    ax1.plot([0, 1], [0, 1], color="grey", linewidth=0.8, linestyle="--",
             label="Random classifier (0.500)")
    ax1.set_xlabel("False Positive Rate", fontsize=9)
    ax1.set_ylabel("True Positive Rate", fontsize=9)
    ax1.set_title("ROC Curve (6-month horizon)", fontsize=9)
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.25)

    # Lead-time distribution
    lt = lead_time_df["Lead Time (months)"].dropna()
    if not lt.empty:
        ax2.hist(lt, bins=range(0, int(lt.max()) + 2), color="#27ae60",
                 edgecolor="white", alpha=0.8)
        ax2.axvline(lt.mean(), color="#e74c3c", linewidth=1.2, linestyle="--",
                    label=f"Mean = {lt.mean():.1f} months")
        ax2.set_xlabel("Lead Time (months before event onset)", fontsize=9)
        ax2.set_ylabel("Count", fontsize=9)
        ax2.set_title("EWS Lead-Time Distribution", fontsize=9)
        ax2.legend(fontsize=8)
        ax2.grid(axis="y", alpha=0.25)
    else:
        ax2.text(0.5, 0.5, "Insufficient events\nfor lead-time analysis",
                 ha="center", va="center", transform=ax2.transAxes, fontsize=10, color="grey")

    plt.tight_layout()
    if save:
        _save(fig, "07_roc_and_lead_time.png")
    return fig
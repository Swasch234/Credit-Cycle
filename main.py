"""
main.py — Credit Cycle Regime Detection & Early Warning System
==============================================================
Author : Shan Xue
Contact: shan.xue@cgu.edu

This project extends the offset/sterilization coefficient framework from
Xue & Willett (2024) — originally developed for central-bank monetary
autonomy in small open economies — to the household consumer credit domain.

The Credit Absorption Coefficient (CAC) is the central methodological
contribution: it measures the degree to which U.S. households use revolving
credit to absorb negative income shocks, analogous to the way capital flows
offset interest-rate differentials in the FX trilemma framework.

Pipeline stages
───────────────
  1. Fetch credit & macro series from FRED (12 series, 2000–present)
  2. Engineer five credit pressure indicators (including the CAC)
  3. Build the Composite Credit Pressure Index (CPI)
  4. Fit a Markov-Switching AR model → 3 credit regimes
  5. Construct the Early Warning System + evaluate (AUROC, Brier, lead time)
  6. Generate 7 publication-quality charts

Quick start
───────────
  pip install -r requirements.txt
  export FRED_API_KEY="your_key_here"   # free at https://fred.stlouisfed.org
  python main.py
"""

import logging
import os

import pandas as pd

import visualizer as viz
from config import OUTPUT_DIR
from credit_indicators import build_all_indicators
from data_fetcher import CreditDataFetcher
from early_warning import EarlyWarningSystem
from regime_model import CreditRegimeModel

os.makedirs(OUTPUT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DIV = "─" * 66


def section(title: str) -> None:
    print(f"\n{DIV}\n  {title}\n{DIV}")


def main() -> None:

    print("\n" + "=" * 66)
    print("   Credit Cycle Regime Detection & Early Warning System")
    print("=" * 66)
    print("""
  Methodological bridge:
  Extends the offset-coefficient framework of Xue & Willett (2024)
  from FX/monetary autonomy to household consumer credit dynamics.
  The Credit Absorption Coefficient (CAC) is the new contribution.
""")

    # ── 1. Data Ingestion ──────────────────────────────────────────────────────
    section("1 / 6  Fetching data from FRED")

    fetcher    = CreditDataFetcher()
    credit_df  = fetcher.fetch_credit_series()
    benchmarks = fetcher.fetch_benchmarks()

    credit_df.to_csv(f"{OUTPUT_DIR}/raw_credit_data.csv")
    benchmarks.to_csv(f"{OUTPUT_DIR}/benchmark_data.csv")

    print(f"\n  ✓ {credit_df.shape[1]} credit series | {credit_df.shape[0]} months")
    print(f"  ✓ {benchmarks.shape[1]} benchmark series")

    # ── 2. Credit Indicators ───────────────────────────────────────────────────
    section("2 / 6  Engineering credit pressure indicators")

    cpi, components, cac_detail = build_all_indicators(credit_df)

    cpi.to_csv(f"{OUTPUT_DIR}/credit_pressure_index.csv")
    components.to_csv(f"{OUTPUT_DIR}/indicator_components.csv")
    cac_detail.to_csv(f"{OUTPUT_DIR}/cac_rolling_estimates.csv")

    print(f"\n  ✓ CPI built: {cpi.index[0].date()} → {cpi.index[-1].date()}")
    print(f"  ✓ {components.shape[1]} component indicators")

    if not cac_detail.empty and "cac_beta" in cac_detail.columns:
        beta_mean = cac_detail["cac_beta"].mean()
        beta_std  = cac_detail["cac_beta"].std()
        print(f"\n  Credit Absorption Coefficient (CAC) summary:")
        print(f"    Mean β  = {beta_mean:.3f}  (>0: households absorb shocks via credit)")
        print(f"    Std  β  = {beta_std:.3f}")
        if not cac_detail["cac_r2"].empty:
            print(f"    Mean R² = {cac_detail['cac_r2'].mean():.3f}")

    # ── 3. Composite Index Summary ─────────────────────────────────────────────
    print(f"\n  CPI descriptive statistics:")
    print(f"    Mean:   {cpi.mean():.3f}  (by construction ≈ 0)")
    print(f"    Std:    {cpi.std():.3f}  (by construction ≈ 1)")
    print(f"    Max:    {cpi.max():.3f}  on {cpi.idxmax().date()}")
    print(f"    Min:    {cpi.min():.3f}  on {cpi.idxmin().date()}")

    # ── 4. Regime Model ────────────────────────────────────────────────────────
    section("3 / 6  Fitting Markov-Switching AR regime model")

    regime_model   = CreditRegimeModel(cpi)
    regime_model.fit()

    smoothed_probs = regime_model.get_smoothed_probabilities()
    regime_series  = regime_model.get_most_likely_regime()
    regime_params  = regime_model.regime_parameters()
    trans_matrix   = regime_model.transition_matrix()
    regime_dur     = regime_model.regime_duration()
    regime_stats   = regime_model.regime_statistics(cpi)
    irf_df         = regime_model.impulse_responses(horizon=24)

    smoothed_probs.to_csv(f"{OUTPUT_DIR}/regime_probabilities.csv")
    regime_series.to_csv(f"{OUTPUT_DIR}/regime_classifications.csv")
    regime_params.to_csv(f"{OUTPUT_DIR}/regime_parameters.csv")
    trans_matrix.to_csv(f"{OUTPUT_DIR}/transition_matrix.csv")
    regime_dur.to_csv(f"{OUTPUT_DIR}/regime_duration.csv")
    regime_stats.to_csv(f"{OUTPUT_DIR}/regime_statistics.csv")
    irf_df.to_csv(f"{OUTPUT_DIR}/impulse_responses.csv")

    print("\n  Regime Parameters (MS-AR):")
    print(regime_params.to_string())

    print("\n  Transition Matrix (row = current, col = next period):")
    print(trans_matrix.round(3).to_string())

    print("\n  Expected Regime Duration:")
    print(regime_dur.to_string())

    print("\n  CPI Statistics by Regime:")
    print(regime_stats.to_string())

    # ── 5. Early Warning System ────────────────────────────────────────────────
    section("4 / 6  Building & evaluating Early Warning System")

    ews = EarlyWarningSystem(
        smoothed_probs = smoothed_probs,
        cpi            = cpi,
        benchmarks     = benchmarks,
        raw_credit     = credit_df,
    )
    signals  = ews.build_signals()
    events   = ews.build_events()

    eval_df  = ews.evaluate(event_col="event_either")
    lead_df  = ews.lead_time_analysis(event_col="event_recession")
    fpr, tpr, auroc = ews.roc_data(horizon=6)

    signals.to_csv(f"{OUTPUT_DIR}/ews_signals.csv")
    events.to_csv(f"{OUTPUT_DIR}/ews_events.csv")
    eval_df.to_csv(f"{OUTPUT_DIR}/ews_evaluation.csv")
    lead_df.to_csv(f"{OUTPUT_DIR}/ews_lead_times.csv")

    print("\n  EWS Evaluation (signal = P(Strain or Crisis)):\n")
    print(eval_df.to_string())

    print(f"\n  ROC curve AUROC (6-month horizon): {auroc:.3f}")
    print(f"  (0.5 = random, 1.0 = perfect)")

    if not lead_df.empty and lead_df["Lead Time (months)"].notna().any():
        mean_lead = lead_df["Lead Time (months)"].mean()
        print(f"\n  Lead-time analysis:")
        print(lead_df.to_string(index=False))
        print(f"\n  Average lead time: {mean_lead:.1f} months before event onset")

    # ── 6. Visualizations ─────────────────────────────────────────────────────
    section("5 / 6  Generating charts")

    recession = (
        benchmarks["nber_recession"] if "nber_recession" in benchmarks.columns else None
    )

    viz.plot_cpi(cpi, recession=recession)
    viz.plot_cac(cac_detail, recession=recession)
    viz.plot_component_dashboard(components, recession=recession)
    viz.plot_regime_detection(cpi, smoothed_probs, regime_series, recession=recession)
    viz.plot_impulse_responses(irf_df)
    viz.plot_ews_signals(signals, events, recession=recession)
    viz.plot_roc_and_lead_time(fpr, tpr, auroc, lead_df)

    # ── Summary ────────────────────────────────────────────────────────────────
    section("6 / 6  Pipeline complete")

    best_auroc_row = eval_df["AUROC"].idxmax() if not eval_df.empty else "N/A"
    best_auroc_val = eval_df["AUROC"].max() if not eval_df.empty else float("nan")
    current_regime = regime_series.iloc[-1] if len(regime_series) > 0 else "Unknown"
    current_p_crisis = smoothed_probs["Crisis"].iloc[-1] if "Crisis" in smoothed_probs.columns else float("nan")

    print(f"""
  ✅  All outputs → /output/

  Key Results
  ───────────
  Current regime (latest month):   {current_regime}
  Current P(Crisis):                {current_p_crisis:.3f}
  Best EWS AUROC:                   {best_auroc_val:.3f}  (at {best_auroc_row}-month horizon)

  CSV outputs
  ───────────
  raw_credit_data.csv          FRED credit series (monthly)
  benchmark_data.csv           NBER recession + consumer confidence
  credit_pressure_index.csv    Composite CPI (standardized, monthly)
  indicator_components.csv     All five normalized indicators
  cac_rolling_estimates.csv    Rolling CAC β, R², std error, z-score
  regime_probabilities.csv     P(Expansion|Strain|Crisis) per month
  regime_classifications.csv   Most likely regime per month
  regime_parameters.csv        MS-AR regime-specific parameters
  transition_matrix.csv        Regime persistence/transition probabilities
  regime_duration.csv          Expected duration per regime
  regime_statistics.csv        CPI descriptive stats per regime
  impulse_responses.csv        IRF under each regime (0–24 months)
  ews_signals.csv              EWS traffic-light signals + probabilities
  ews_events.csv               Binary credit event indicators
  ews_evaluation.csv           AUROC, Brier, F1 at each horizon
  ews_lead_times.csv           Lead time per historical credit event

  Charts (output/)
  ────────────────
  01_credit_pressure_index.png
  02_cac_rolling.png
  03_component_dashboard.png
  04_regime_detection.png
  05_impulse_responses.png
  06_early_warning_signals.png
  07_roc_and_lead_time.png
""")


if __name__ == "__main__":
    main()

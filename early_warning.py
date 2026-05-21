"""
early_warning.py — Credit Cycle Early Warning System (EWS).

Architecture
────────────
The EWS converts the regime model's probabilistic output into actionable
signals at three severity levels:

  GREEN  : P(Strain or Crisis) ≤ EWS_SIGNAL_THRESHOLD
  YELLOW : P(Strain or Crisis) > EWS_SIGNAL_THRESHOLD
  RED    : P(Crisis)           > EWS_CRISIS_THRESHOLD

Evaluation metrics
──────────────────
  AUROC         : Area under the ROC curve — pure discrimination ability,
                  threshold-independent.  Benchmark: 0.5 (random), 1.0 (perfect).
  Brier Score   : Mean squared error of probability forecasts.  Lower = better.
                  Calibrated model: Brier ≈ base_rate * (1 - base_rate).
  Precision     : P(event | signal fired) — avoids false alarms
  Recall        : P(signal fired | event) — avoids missed events
  F1            : Harmonic mean of precision and recall
  Lead Time     : Months between signal and event onset (how early does it warn?)

Credit events
─────────────
Two event definitions are used:
  1. NBER recession months (binary, from FRED)
  2. "Credit stress" = (delinquency rate > historical 75th percentile)
     OR (charge-off rate > historical 75th percentile)

Threshold calibration
─────────────────────
The optimal threshold balances precision and recall.  We sweep thresholds
from 0.1 to 0.9 and report the threshold that maximises F1 for each horizon.
This is plotted as a precision-recall curve.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from config import EWS_CRISIS_THRESHOLD, EWS_HORIZON_MONTHS, EWS_SIGNAL_THRESHOLD

logger = logging.getLogger(__name__)


class EarlyWarningSystem:
    """
    Build and evaluate the credit cycle EWS.

    Parameters
    ----------
    smoothed_probs : pd.DataFrame
        Columns ['Expansion', 'Strain', 'Crisis'] — regime probabilities.
    cpi : pd.Series
        Composite Credit Pressure Index (for threshold-based rules).
    benchmarks : pd.DataFrame
        Must contain 'nber_recession' column; optionally 'cc_delinquency',
        'cc_chargeoff' for credit-stress event definition.
    raw_credit : pd.DataFrame
        The full credit dataset (for computing credit-stress thresholds).
    """

    def __init__(
        self,
        smoothed_probs: pd.DataFrame,
        cpi: pd.Series,
        benchmarks: pd.DataFrame,
        raw_credit: pd.DataFrame,
    ):
        self.probs     = smoothed_probs
        self.cpi       = cpi
        self.benchmarks= benchmarks
        self.raw_credit= raw_credit

        self._signals: pd.DataFrame | None      = None
        self._events: pd.DataFrame | None       = None

    # ── Signal construction ────────────────────────────────────────────────────

    def build_signals(self) -> pd.DataFrame:
        """
        Construct the three-level EWS signal.

        Columns:
          p_stress_or_crisis  — P(Strain) + P(Crisis)
          p_crisis            — P(Crisis) only
          signal_color        — 'GREEN' | 'YELLOW' | 'RED'
          cpi_level           — raw CPI value (for additional rule-based alerts)
          cpi_alert           — True if CPI > 1.5 std above mean
        """
        df = pd.DataFrame(index=self.probs.index)

        strain = self.probs.get("Strain", pd.Series(0, index=self.probs.index))
        crisis = self.probs.get("Crisis", pd.Series(0, index=self.probs.index))

        df["p_stress_or_crisis"] = (strain + crisis).clip(0, 1)
        df["p_crisis"]           = crisis.clip(0, 1)

        # Colour signal
        def _colour(row: pd.Series) -> str:
            if row["p_crisis"] > EWS_CRISIS_THRESHOLD:
                return "RED"
            elif row["p_stress_or_crisis"] > EWS_SIGNAL_THRESHOLD:
                return "YELLOW"
            return "GREEN"

        df["signal_color"] = df.apply(_colour, axis=1)

        # CPI-level alert (model-free confirmation)
        cpi_aligned = self.cpi.reindex(df.index)
        df["cpi_level"] = cpi_aligned
        df["cpi_alert"] = cpi_aligned > (cpi_aligned.mean() + 1.5 * cpi_aligned.std())

        self._signals = df
        logger.info(
            f"EWS signals: {(df['signal_color']=='GREEN').sum()} GREEN  "
            f"{(df['signal_color']=='YELLOW').sum()} YELLOW  "
            f"{(df['signal_color']=='RED').sum()} RED"
        )
        return df

    # ── Event construction ────────────────────────────────────────────────────

    def build_events(self) -> pd.DataFrame:
        """
        Construct binary credit-event indicators.

        event_recession     : 1 = NBER recession month
        event_credit_stress : 1 = delinquency OR charge-off above 75th pctile
        event_either        : union of the two definitions
        """
        idx = self.probs.index
        ev  = pd.DataFrame(index=idx)

        # Recession event
        if "nber_recession" in self.benchmarks.columns:
            rec = self.benchmarks["nber_recession"].reindex(idx).fillna(0)
            ev["event_recession"] = (rec == 1).astype(int)
        else:
            ev["event_recession"] = 0

        # Credit stress event
        stress_flags = []
        for col in ["cc_delinquency", "cc_chargeoff", "consumer_loan_delinquency"]:
            if col in self.raw_credit.columns:
                s    = self.raw_credit[col].reindex(idx)
                p75  = s.quantile(0.75)
                stress_flags.append((s > p75).astype(int))

        if stress_flags:
            any_stress = pd.concat(stress_flags, axis=1).max(axis=1)
            ev["event_credit_stress"] = any_stress.fillna(0).astype(int)
        else:
            ev["event_credit_stress"] = ev["event_recession"].copy()

        ev["event_either"] = ((ev["event_recession"] + ev["event_credit_stress"]) > 0).astype(int)

        self._events = ev
        return ev

    # ── Evaluation ────────────────────────────────────────────────────────────

    def evaluate(
        self,
        event_col: str = "event_either",
        horizons: list[int] = EWS_HORIZON_MONTHS,
    ) -> pd.DataFrame:
        """
        Evaluate EWS performance at multiple forecast horizons.

        For horizon h: does the signal at time t predict an event
        in any of the months t+1 … t+h?

        Returns
        -------
        pd.DataFrame  — AUROC, Brier score, precision, recall, F1, optimal threshold
                        for each horizon.
        """
        if self._signals is None:
            self.build_signals()
        if self._events is None:
            self.build_events()

        rows = []
        for h in horizons:
            # Future event: did an event occur within the next h months?
            future_event = self._events[event_col].rolling(
                window=h, min_periods=1
            ).max().shift(-h)   # shift backward so t has the next-h-months max

            prob_score = self._signals["p_stress_or_crisis"]

            combined = pd.concat([prob_score, future_event], axis=1).dropna()
            combined.columns = ["score", "event"]

            if combined["event"].sum() < 5:
                logger.warning(f"  Horizon {h}m: too few events ({combined['event'].sum()}) — skipping")
                continue

            y_true = combined["event"].values.astype(int)
            y_score = combined["score"].values

            # AUROC
            try:
                auroc = roc_auc_score(y_true, y_score)
            except Exception:
                auroc = np.nan

            # Brier score
            try:
                brier = brier_score_loss(y_true, y_score)
            except Exception:
                brier = np.nan

            # Optimal threshold (maximise F1)
            best_f1, best_thresh, best_prec, best_rec = 0.0, 0.5, 0.0, 0.0
            for thresh in np.linspace(0.1, 0.9, 81):
                y_pred = (y_score >= thresh).astype(int)
                tp = ((y_pred == 1) & (y_true == 1)).sum()
                fp = ((y_pred == 1) & (y_true == 0)).sum()
                fn = ((y_pred == 0) & (y_true == 1)).sum()
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
                if f1 > best_f1:
                    best_f1, best_thresh, best_prec, best_rec = f1, thresh, prec, rec

            rows.append({
                "Horizon (months)": h,
                "AUROC":            round(auroc, 3),
                "Brier Score":      round(brier, 4),
                "Precision (@opt)": round(best_prec, 3),
                "Recall (@opt)":    round(best_rec, 3),
                "F1 (@opt)":        round(best_f1, 3),
                "Opt. Threshold":   round(best_thresh, 2),
                "N events":         int(y_true.sum()),
                "N observations":   len(y_true),
            })

        return pd.DataFrame(rows).set_index("Horizon (months)")

    def lead_time_analysis(
        self,
        event_col: str = "event_recession",
        threshold: float = EWS_SIGNAL_THRESHOLD,
    ) -> pd.DataFrame:
        """
        For each historical credit event (recession or stress episode), measure
        how many months before the event onset the EWS first fired a YELLOW or
        RED signal.

        Returns
        -------
        pd.DataFrame  — One row per event episode with lead time in months.
        """
        if self._signals is None:
            self.build_signals()
        if self._events is None:
            self.build_events()

        events = self._events[event_col]
        scores = self._signals["p_stress_or_crisis"]

        # Find event episode start dates (first month of each streak of 1s)
        event_starts = []
        in_event = False
        for date, val in events.items():
            if val == 1 and not in_event:
                event_starts.append(date)
                in_event = True
            elif val == 0:
                in_event = False

        rows = []
        for onset in event_starts:
            # Look back up to 24 months before the event
            lookback = scores.loc[
                scores.index <= onset
            ].iloc[-25:]

            signal_dates = lookback[lookback > threshold].index
            if len(signal_dates) == 0:
                lead = np.nan   # signal never fired before this event
            else:
                first_signal = signal_dates[0]
                lead = int((onset - first_signal).days / 30.44)

            rows.append({
                "Event Onset": onset.date(),
                "First Signal": signal_dates[0].date() if len(signal_dates) > 0 else "Never",
                "Lead Time (months)": lead,
            })

        df = pd.DataFrame(rows)
        if not df.empty and df["Lead Time (months)"].notna().any():
            logger.info(
                f"Lead time analysis: "
                f"mean = {df['Lead Time (months)'].mean():.1f} months  "
                f"min = {df['Lead Time (months)'].min():.0f}  "
                f"max = {df['Lead Time (months)'].max():.0f}"
            )
        return df

    def roc_data(
        self,
        event_col: str = "event_either",
        horizon: int = 6,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Return (fpr, tpr, auroc) for the ROC curve plot at the given horizon.
        """
        if self._signals is None:
            self.build_signals()
        if self._events is None:
            self.build_events()

        future_event = (
            self._events[event_col]
            .rolling(window=horizon, min_periods=1)
            .max()
            .shift(-horizon)
        )
        combined = pd.concat(
            [self._signals["p_stress_or_crisis"], future_event], axis=1
        ).dropna()
        combined.columns = ["score", "event"]

        y_true  = combined["event"].values.astype(int)
        y_score = combined["score"].values

        fpr, tpr, _ = roc_curve(y_true, y_score)
        auroc       = auc(fpr, tpr)
        return fpr, tpr, auroc

    def precision_recall_data(
        self,
        event_col: str = "event_either",
        horizon: int = 6,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (precision, recall) arrays for the precision-recall curve."""
        if self._signals is None:
            self.build_signals()
        if self._events is None:
            self.build_events()

        future_event = (
            self._events[event_col]
            .rolling(window=horizon, min_periods=1)
            .max()
            .shift(-horizon)
        )
        combined = pd.concat(
            [self._signals["p_stress_or_crisis"], future_event], axis=1
        ).dropna()
        combined.columns = ["score", "event"]

        precision, recall, _ = precision_recall_curve(
            combined["event"].astype(int).values,
            combined["score"].values,
        )
        return precision, recall

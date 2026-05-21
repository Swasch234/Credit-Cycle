"""
data_fetcher.py — Fetch all credit cycle series from FRED.

Design decisions:
  • Quarterly series (delinquency, SLOOS, debt service) are forward-filled
    within the quarter after resampling to month-end.  This correctly reflects
    point-in-time availability: the Q3 reading becomes known in late October
    and is carried forward until the Q4 reading arrives.
  • Daily series (credit spreads) are averaged to monthly.
  • All series land on month-end dates (pandas "ME" offset).
"""

import logging

import pandas as pd
from fredapi import Fred

from config import (
    BENCHMARK_SERIES,
    CREDIT_SERIES,
    END_DATE,
    FRED_API_KEY,
    START_DATE,
)

logger = logging.getLogger(__name__)


class CreditDataFetcher:
    """
    Download all credit-cycle and benchmark series from FRED,
    harmonize to monthly end-of-period, and return aligned DataFrames.
    """

    def __init__(self, api_key: str = FRED_API_KEY):
        _PLACEHOLDER_KEYS = {
            "your_fred_api_key_here",
            "",
        }
        if not api_key or api_key in _PLACEHOLDER_KEYS:
            raise ValueError(
                "FRED API key not set.\n"
                "  * Get a free key at: https://fred.stlouisfed.org/docs/api/api_key.html\n"
                "  * Then run:  export FRED_API_KEY='<your_key>'  (Mac/Linux)\n"
                "              $Env:FRED_API_KEY='<your_key>'     (PowerShell)"
            )
        self.fred = Fred(api_key=api_key)

    # -- Internal helpers -------------------------------------------------------

    def _fetch(self, series_id: str) -> pd.Series:
        s = self.fred.get_series(
            series_id,
            observation_start=START_DATE,
            observation_end=END_DATE,
        )
        s.name = series_id
        logger.info(
            f"  {series_id:25s}: {len(s):5d} obs  "
            f"({s.index[0].date()} -> {s.index[-1].date()})"
        )
        return s

    @staticmethod
    def _harmonize(s: pd.Series, freq: str) -> pd.Series:
        """Resample to monthly ME, applying frequency-appropriate aggregation."""
        if freq in ("D", "W"):
            return s.resample("ME").mean()
        elif freq == "Q":
            monthly = s.resample("ME").last()
            return monthly.ffill(limit=2)
        else:  # "M"
            return s.resample("ME").last()

    # -- Public API -------------------------------------------------------------

    def fetch_credit_series(self) -> pd.DataFrame:
        """
        Fetch all credit stress series and return as an aligned monthly DataFrame.
        Rows with > 60% missing values are dropped.
        """
        logger.info("-- Fetching credit stress series ---------------------------------")
        frames: dict[str, pd.Series] = {}

        for name, meta in CREDIT_SERIES.items():
            try:
                raw = self._fetch(meta["id"])
                monthly = self._harmonize(raw, meta["frequency"])
                frames[name] = monthly
            except Exception as exc:
                logger.warning(f"  Could not fetch {name} ({meta['id']}): {exc}")

        df = pd.DataFrame(frames)

        if df.empty:
            raise RuntimeError(
                "No credit series could be fetched from FRED -- the DataFrame is empty.\n"
                "All series requests failed (check the WARNING lines above).\n"
                "Ensure FRED_API_KEY is set to a valid 32-character key:\n"
                "  export FRED_API_KEY='<your_key>'   (Mac/Linux)\n"
                "  $Env:FRED_API_KEY='<your_key>'     (PowerShell)"
            )

        missing = df.isnull().mean(axis=1)
        df = df[missing <= 0.60]

        if df.empty:
            raise RuntimeError(
                "All fetched rows exceeded the 60% missing-value threshold and were dropped.\n"
                "Check that the fetched series cover an overlapping date range."
            )

        logger.info(
            f"\nCredit dataset: {df.shape[0]} months x {df.shape[1]} series  "
            f"({df.index[0].date()} -> {df.index[-1].date()})"
        )
        return df

    def fetch_benchmarks(self) -> pd.DataFrame:
        """Fetch NBER recession indicator and consumer confidence."""
        logger.info("-- Fetching benchmark series -------------------------------------")
        frames: dict[str, pd.Series] = {}
        for name, meta in BENCHMARK_SERIES.items():
            try:
                raw = self._fetch(meta["id"])
                frames[name] = raw.resample("ME").last().ffill()
            except Exception as exc:
                logger.warning(f"  Could not fetch {name}: {exc}")
        return pd.DataFrame(frames)
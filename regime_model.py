"""
regime_model.py — Markov-Switching Autoregressive model for the Credit Pressure Index.

Model specification
───────────────────
  MS-AR(p): CPI_t = μ(s_t) + Σ φ_k(s_t) * CPI_{t-k} + σ(s_t) * ε_t

  s_t ∈ {0, 1, 2}  (hidden regime, Markov chain)

  • Switching mean        : μ(s_t)       — regime-specific intercept
  • Switching AR(1)       : φ_1(s_t)     — regime-specific persistence
  • Switching variance    : σ²(s_t)      — crisis regimes are more volatile

  Estimation: EM algorithm via statsmodels MarkovAutoregression
  Smoothing:  Kim (1994) full-sample smoother → P(s_t | y_1,…,y_T)

Regime labeling
───────────────
  Regimes are ordered by their estimated mean CPI (ascending):
    Regime 0 → Expansion (low CPI, low persistence)
    Regime 1 → Strain    (moderate CPI, moderate persistence)
    Regime 2 → Crisis    (high CPI, high variance)

Impulse response analysis
──────────────────────────
  Within each regime, we compute the impulse response of the CPI to a
  one-standard-deviation shock, using the regime-specific AR coefficients.
  This shows how shocks propagate differently across regimes — in a Crisis
  regime, shocks are amplified and more persistent.  This is analogous to
  the regime-conditional impulse responses in Shan's VAR work.
"""

import logging
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression

from config import EM_ITERATIONS, N_REGIMES, SEARCH_REPS

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# ── Regime metadata ────────────────────────────────────────────────────────────
REGIME_LABELS = {0: "Expansion", 1: "Strain", 2: "Crisis"}
REGIME_COLORS = {
    "Expansion": "#27ae60",  # green
    "Strain": "#f39c12",  # amber
    "Crisis": "#e74c3c",  # red
}


class CreditRegimeModel:
    """
    Markov-Switching Autoregressive model fitted on the Credit Pressure Index.

    Parameters
    ----------
    cpi       : pd.Series  — Composite Credit Pressure Index (standardized, monthly)
    n_regimes : int        — Number of hidden regimes (default 3)
    order     : int        — AR order (default 1; AICC suggests 1-2 for monthly CPI)
    """

    def __init__(
            self,
            cpi: pd.Series,
            n_regimes: int = N_REGIMES,
            order: int = 1,
    ):
        self.cpi = cpi.dropna()
        self.n_regimes = n_regimes
        self.order = order

        self._result = None
        self._label_map: dict[int, str] = {}
        self.smoothed_probs: pd.DataFrame | None = None
        self.regime_series: pd.Series | None = None

    # ── Fitting ────────────────────────────────────────────────────────────────

    def fit(self) -> None:
        """
        Estimate the MS-AR model via the EM algorithm.

        switching_ar=True   : AR coefficient differs by regime
        switching_variance=True: error variance differs by regime
        Multiple random starts (search_reps) guard against local optima.
        """
        logger.info(
            f"Fitting MS-AR({self.order}) with {self.n_regimes} regimes "
            f"on {len(self.cpi)} observations…"
        )

        def _try_fit(switching_ar: bool, switching_variance: bool):
            m = MarkovAutoregression(
                self.cpi,
                k_regimes=self.n_regimes,
                order=self.order,
                trend="c",
                switching_ar=switching_ar,
                switching_variance=switching_variance,
            )
            return m.fit(em_iter=EM_ITERATIONS, search_reps=SEARCH_REPS, disp=False)

        # Try full spec first; fall back to simpler specs if LL is NaN (degenerate fit)
        fallback_specs = [
            (True, True),  # full: switching AR + switching variance
            (False, True),  # switching variance only
            (True, False),  # switching AR only
            (False, False),  # simplest: switching mean only
        ]
        result = None
        for sw_ar, sw_var in fallback_specs:
            result = _try_fit(sw_ar, sw_var)
            if result.llf is not None and not np.isnan(result.llf):
                self._result = result
                logger.info(
                    f"Converged (switching_ar={sw_ar}, switching_variance={sw_var}).  "
                    f"LL={result.llf:.2f}  AIC={result.aic:.2f}  BIC={result.bic:.2f}"
                )
                break
            logger.warning(
                f"Degenerate fit (LL=nan) with switching_ar={sw_ar}, "
                f"switching_variance={sw_var} — trying simpler spec…"
            )
        else:
            self._result = result
            logger.warning("All model specs returned LL=nan; results may be unreliable.")

        self._build_label_map()

    def _build_label_map(self) -> None:
        """Order regimes by estimated intercept (low→Expansion, high→Crisis)."""
        means = {}
        for i in range(self.n_regimes):
            try:
                means[i] = self._result.params[f"const[{i}]"]
            except KeyError:
                means[i] = float("nan")
        ordered = sorted(means, key=lambda x: means[x])
        self._label_map = {ordered[i]: list(REGIME_LABELS.values())[i]
                           for i in range(self.n_regimes)}
        logger.info(f"Regime label map: {self._label_map}")

    # ── Smoothed probabilities ─────────────────────────────────────────────────

    def get_smoothed_probabilities(self) -> pd.DataFrame:
        """
        Kim (1994) full-sample smoothed marginal probabilities.
        Returns DataFrame with columns ['Expansion', 'Strain', 'Crisis'].
        """
        if self._result is None:
            self.fit()
        raw = self._result.smoothed_marginal_probabilities
        cols = [self._label_map[i] for i in range(self.n_regimes)]
        raw.columns = cols
        self.smoothed_probs = raw
        return raw

    def get_most_likely_regime(self) -> pd.Series:
        if self.smoothed_probs is None:
            self.get_smoothed_probabilities()
        # Fill any all-NaN rows with uniform probability before argmax
        # (can occur if EM converged to a degenerate solution)
        filled = self.smoothed_probs.apply(
            lambda row: row.fillna(1.0 / len(row)) if row.isna().all() else row,
            axis=1,
        )
        regime = filled.idxmax(axis=1)
        regime.name = "Regime"
        self.regime_series = regime
        return regime

    # ── Parameter summary ──────────────────────────────────────────────────────

    def regime_parameters(self) -> pd.DataFrame:
        """
        Summary table of regime-specific parameters:
        mean (intercept), AR(1) coefficient, error std, and persistence.
        """
        if self._result is None:
            self.fit()
        rows = []
        for i in range(self.n_regimes):
            label = self._label_map.get(i, f"Regime {i}")
            try:
                intercept = self._result.params.get(f"const[{i}]", np.nan)
                ar1 = self._result.params.get(f"ar.L1[{i}]", np.nan)
                sigma = np.sqrt(self._result.params.get(f"sigma2[{i}]", np.nan))
            except Exception:
                intercept, ar1, sigma = np.nan, np.nan, np.nan
            rows.append({
                "Regime": label,
                "Intercept (μ)": round(intercept, 4),
                "AR(1) (φ)": round(ar1, 4),
                "Std Dev (σ)": round(sigma, 4),
                "Half-life (months)": (
                    round(np.log(0.5) / np.log(abs(ar1)), 1)
                    if not np.isnan(ar1) and abs(ar1) < 1 and ar1 != 0
                    else np.nan
                ),
            })
        return pd.DataFrame(rows).set_index("Regime")

    def transition_matrix(self) -> pd.DataFrame:
        """Estimated regime transition probability matrix (row → next period col)."""
        if self._result is None:
            self.fit()
        n = self.n_regimes
        labels = [self._label_map[i] for i in range(n)]
        mat = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                key = f"p[{i}->{j}]"
                if key in self._result.params:
                    mat[i, j] = self._result.params[key]
        row_sums = mat.sum(axis=1, keepdims=True)
        mat = mat / np.where(row_sums == 0, 1, row_sums)
        return pd.DataFrame(mat, index=labels, columns=labels)

    def regime_duration(self) -> pd.DataFrame:
        """Expected duration of each regime (in months) from the diagonal of P."""
        trans = self.transition_matrix()
        rows = []
        for regime in trans.index:
            p_stay = trans.loc[regime, regime]
            expected_dur = 1 / (1 - p_stay) if p_stay < 1 else np.inf
            rows.append({"Regime": regime, "E[Duration] (months)": round(expected_dur, 1),
                         "P(stay)": round(p_stay, 3)})
        return pd.DataFrame(rows).set_index("Regime")

    # ── Impulse responses ──────────────────────────────────────────────────────

    def impulse_responses(self, horizon: int = 24) -> pd.DataFrame:
        """
        Compute impulse response functions under each regime.

        A one-standard-deviation shock to the CPI at time 0 is propagated
        forward using each regime's AR(1) coefficient.  This shows how
        shocks are amplified and absorbed differently across regimes.

        In the Crisis regime, the higher φ means shocks are more persistent —
        the system takes longer to return to neutral.  This is the consumer-
        credit analogue of the impulse-response analysis in Shan's VAR work,
        where exchange-rate shocks under a currency board dissipate faster
        than under a managed float.

        Parameters
        ----------
        horizon : int — number of months to project the impulse response

        Returns
        -------
        pd.DataFrame — index = horizon (months 0..horizon), columns = regimes
        """
        if self._result is None:
            self.fit()

        irf = {}
        for i in range(self.n_regimes):
            label = self._label_map.get(i, f"Regime {i}")
            ar1 = self._result.params.get(f"ar.L1[{i}]", 0.0)
            sigma = np.sqrt(self._result.params.get(f"sigma2[{i}]", 1.0))

            # IRF of AR(1): response at horizon h = φ^h * σ
            responses = [sigma * (ar1 ** h) for h in range(horizon + 1)]
            irf[label] = responses

        return pd.DataFrame(irf, index=range(horizon + 1))

    # ── Regime statistics ──────────────────────────────────────────────────────

    def regime_statistics(self, cpi: pd.Series) -> pd.DataFrame:
        """Descriptive statistics of the CPI within each identified regime."""
        if self.regime_series is None:
            self.get_most_likely_regime()

        combined = pd.DataFrame({"CPI": cpi, "Regime": self.regime_series}).dropna()
        stats = (
            combined.groupby("Regime")["CPI"]
            .agg(Mean="mean", Std="std", Min="min", Max="max", Months="count")
            .round(3)
        )
        stats["% of Sample"] = (stats["Months"] / stats["Months"].sum() * 100).round(1)
        order = [l for l in ["Expansion", "Strain", "Crisis"] if l in stats.index]
        return stats.loc[order]
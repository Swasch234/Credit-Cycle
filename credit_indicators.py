"""
credit_indicators.py — Engineer credit pressure indicators and the
                        Credit Absorption Coefficient (CAC).

This file is the methodological heart of the project.  It builds five
composite indicators that together measure where the consumer credit
cycle stands, then packages them into a single Credit Pressure Index (CPI).

─────────────────────────────────────────────────────────────────────────────
The Credit Absorption Coefficient — bridge to Xue & Willett (2024)
─────────────────────────────────────────────────────────────────────────────
Xue & Willett (2024) measure monetary autonomy in small open economies via
the "offset coefficient" β in:

    ΔDomesticRate_t = α + β * ΔWorldRate_t + ε_t

    β → 1 : domestic rates fully track world rates (no autonomy)
    β → 0 : domestic rates are insulated (full autonomy)

This paper is extended here to the household credit domain.  Define the
"Credit Absorption Coefficient" (CAC) as β in the rolling regression:

    Δlog(RevolvingCredit_t) = α + β * ΔIncomeShock_t + ε_t

where ΔIncomeShock = max(0, -Δlog(RealIncome)) captures negative income shocks.

Interpretation:
    β > 0  : households actively borrow to offset income shortfalls
             (analogous to capital flowing in to keep the exchange rate
             stable — the system "absorbs" the shock via credit)
    β → 0  : credit is not used to smooth income shocks
    β < 0  : credit contracts when income falls (banks tighten simultaneously)

A persistently positive and rising CAC signals that household balance sheets
are acting as the shock absorber — accumulating debt in lieu of consumption
adjustment.  When the CAC reverses (β collapses), the buffer has been
exhausted and a stress episode typically follows — exactly the regime
transition the Markov-Switching model is trained to detect.

─────────────────────────────────────────────────────────────────────────────
Five composite indicators
─────────────────────────────────────────────────────────────────────────────
1. CreditGap           : deviation of credit-to-income ratio from HP trend
2. DelinquencyMomentum : acceleration in delinquency rate (2nd derivative)
3. LoanTighteningIndex : composite of SLOOS tightening standards
4. ChargeOffAcceleration: rate of change in charge-off rate
5. CreditAbsorptionCoef: rolling CAC (the FX-analogy indicator)
"""

import logging

import numpy as np
import pandas as pd
from scipy.signal import lfilter
from sklearn.linear_model import LinearRegression

from config import (
    CAC_MIN_PERIODS,
    CAC_WINDOW_MONTHS,
    CREDIT_SERIES,
    HP_LAMBDA,
)

logger = logging.getLogger(__name__)


# ── HP Filter ─────────────────────────────────────────────────────────────────

def hp_filter(series: pd.Series, lamb: float = HP_LAMBDA) -> tuple[pd.Series, pd.Series]:
    """
    Hodrick-Prescott filter.  Returns (trend, cycle) as pd.Series.
    Uses the standard matrix formulation (Hamilton 2018 critique
    acknowledged — HP is used here as an established credit-gap benchmark
    consistent with BIS methodology, not as a general forecasting tool).
    """
    T = len(series)
    y = series.values

    # Build second-difference matrix
    from scipy.sparse import eye, diags
    from scipy.sparse.linalg import spsolve
    from scipy.sparse import csr_matrix

    I = eye(T, format="csr")
    D2 = diags([1, -2, 1], [0, 1, 2], shape=(T - 2, T), format="csr")
    A = I + lamb * D2.T @ D2
    trend = spsolve(A, y)

    return (
        pd.Series(trend, index=series.index, name=f"{series.name}_trend"),
        pd.Series(y - trend, index=series.index, name=f"{series.name}_cycle"),
    )


# ── Indicator 1: Credit Gap ────────────────────────────────────────────────────

class CreditGapEstimator:
    """
    Credit-to-income gap: deviation of log(revolving credit / real income)
    from its HP-filtered trend.

    Positive gap → credit growing faster than income → pressure accumulating.
    This is the consumer-credit analogue of the BIS credit-to-GDP gap, which
    the BIS has identified as one of the best early warning indicators of
    banking crises (Drehmann & Tsatsaronis 2014).

    We use revolving (unsecured) credit rather than total credit because
    revolving credit is the most discretionary and most sensitive to
    household financial stress — matching Plaid's transaction-level focus.
    """

    def __init__(self, credit: pd.Series, income: pd.Series):
        self.credit = credit
        self.income = income
        self.gap: pd.Series | None = None

    def estimate(self) -> pd.Series:
        aligned = pd.concat([self.credit, self.income], axis=1).dropna()
        ratio = np.log(aligned.iloc[:, 0] / aligned.iloc[:, 1])
        _, cycle = hp_filter(ratio)
        cycle.name = "credit_gap"
        self.gap = cycle
        logger.info(
            f"Credit gap estimated: "
            f"{cycle.index[0].date()} → {cycle.index[-1].date()}  "
            f"| Mean: {cycle.mean():.4f}  Std: {cycle.std():.4f}"
        )
        return cycle


# ── Indicator 2: Delinquency Momentum ─────────────────────────────────────────

class DelinquencyMomentumEstimator:
    """
    Measures the *acceleration* of credit card delinquency rates —
    the second derivative of delinquency with respect to time.

    Motivation: the level of delinquency is a lagging indicator (it rises
    only after households have already been struggling for months).  The
    *rate of change* (first derivative) is faster.  The *acceleration*
    (second derivative) is faster still and leads turning points.

    Analogous to the "trilemma pressure index" in Xue & Willett (2024),
    which measures not just the level of interest-rate deviation but its
    velocity relative to peg commitments.
    """

    def __init__(
            self,
            delinquency: pd.Series,
            smooth_window: int = 3,
    ):
        self.delinquency = delinquency
        self.smooth_window = smooth_window

    def estimate(self) -> pd.DataFrame:
        s = self.delinquency.dropna()

        # Smooth to remove quarterly-release noise
        smoothed = s.rolling(self.smooth_window, min_periods=1).mean()
        velocity = smoothed.diff(1)  # first derivative
        acceleration = velocity.diff(1)  # second derivative

        # Z-score normalize each
        def zscore(x: pd.Series) -> pd.Series:
            std = x.std()
            return (x - x.mean()) / (std if std != 0 else np.nan)

        out = pd.DataFrame({
            "delinquency_level": zscore(smoothed),
            "delinquency_velocity": zscore(velocity),
            "delinquency_acceleration": zscore(acceleration),
        })
        logger.info(f"Delinquency momentum: {out.shape[0]} months")
        return out


# ── Indicator 3: Loan Tightening Index ────────────────────────────────────────

class LoanTighteningIndexEstimator:
    """
    Composite of SLOOS (Senior Loan Officer Opinion Survey) tightening measures.

    The SLOOS is a forward-looking supply-side indicator: when banks tighten
    standards, credit growth slows 2–4 quarters later, often preceding the
    deterioration captured in delinquency rates.  This is the credit-supply
    analogue of sterilization in the FX context — the system's attempt to
    contain the pressure.
    """

    def __init__(self, tightening_cols: dict[str, pd.Series]):
        self.cols = tightening_cols  # name → pd.Series

    def estimate(self) -> pd.Series:
        if not self.cols:
            logger.warning("No tightening series available — returning zeros.")
            return pd.Series(dtype=float, name="tightening_index")

        df = pd.DataFrame(self.cols).dropna(how="all")
        # Forward-fill quarterly readings
        df = df.ffill(limit=2)

        # Equal-weight composite, normalized
        composite = df.mean(axis=1)
        normalized = (composite - composite.mean()) / composite.std()
        normalized.name = "tightening_index"
        logger.info(f"Loan tightening index: {len(normalized)} months")
        return normalized


# ── Indicator 4: Charge-Off Acceleration ──────────────────────────────────────

class ChargeOffAccelerationEstimator:
    """
    Rate of change of credit card charge-off rates.

    Charge-offs are an actualized-loss measure: they reflect credit stress
    that materialised 6–18 months earlier.  Their *acceleration* is a more
    timely signal than the level and acts as a confirmation of stress episodes
    detected by leading indicators.
    """

    def __init__(self, chargeoff: pd.Series, smooth_window: int = 3):
        self.chargeoff = chargeoff
        self.smooth_window = smooth_window

    def estimate(self) -> pd.Series:
        s = self.chargeoff.dropna().rolling(self.smooth_window, min_periods=1).mean()
        accel = s.diff(1)
        normalized = (accel - accel.mean()) / accel.std()
        normalized.name = "chargeoff_acceleration"
        return normalized


# ── Indicator 5: Credit Absorption Coefficient (CAC) ─────────────────────────

class CreditAbsorptionCoefficientEstimator:
    """
    Rolling estimation of the Credit Absorption Coefficient (CAC).

    Model: Δlog(RevolvingCredit_t) = α + β * NegIncomeShock_t + ε_t

    where NegIncomeShock_t = max(0, -Δlog(RealIncome_t))
    captures only the downside income shocks households must absorb.

    β (the CAC) is estimated via OLS on a rolling window of
    CAC_WINDOW_MONTHS months.

    This is the direct methodological extension of the offset-coefficient
    framework in Xue & Willett (2024) from the central-bank / FX context
    to the household credit context.  A high and rising CAC indicates that
    households are absorbing income shocks via revolving debt — building up
    latent balance-sheet vulnerability.

    Signal interpretation:
      CAC ↑ → households leaning on credit → pressure accumulating
      CAC ↓ (or goes negative) → credit no longer available as buffer
              → stress episode or tightening regime likely
    """

    def __init__(
            self,
            revolving_credit: pd.Series,
            real_income: pd.Series,
            window: int = CAC_WINDOW_MONTHS,
            min_periods: int = CAC_MIN_PERIODS,
    ):
        self.credit = revolving_credit
        self.income = real_income
        self.window = window
        self.min_periods = min_periods

    def estimate(self) -> pd.DataFrame:
        """
        Returns a DataFrame with columns:
          cac_beta       — rolling OLS coefficient (the CAC)
          cac_r2         — rolling R² (how well income shocks explain credit)
          cac_stderr     — standard error of β (uncertainty of the estimate)
          cac_zscore     — standardized CAC (for composite index construction)
        """
        # Align and compute growth rates
        aligned = pd.concat([self.credit, self.income], axis=1).dropna()
        aligned.columns = ["credit", "income"]

        d_credit = np.log(aligned["credit"]).diff()
        d_income = np.log(aligned["income"]).diff()

        # Negative income shock (clamp positive income growth to zero)
        neg_shock = (-d_income).clip(lower=0)

        df = pd.DataFrame({
            "d_credit": d_credit,
            "neg_shock": neg_shock,
        }).dropna()

        # Rolling OLS
        betas, r2s, stderrs = [], [], []

        for i in range(len(df)):
            start = max(0, i - self.window + 1)
            window_df = df.iloc[start: i + 1]

            if len(window_df) < self.min_periods:
                betas.append(np.nan)
                r2s.append(np.nan)
                stderrs.append(np.nan)
                continue

            X = window_df["neg_shock"].values.reshape(-1, 1)
            y = window_df["d_credit"].values

            # Need at least some non-zero shocks to fit meaningfully
            if (X != 0).sum() < 6:
                betas.append(np.nan)
                r2s.append(np.nan)
                stderrs.append(np.nan)
                continue

            lr = LinearRegression().fit(X, y)
            beta = lr.coef_[0]

            # Compute R² and stderr
            y_hat = lr.predict(X)
            ss_res = np.sum((y - y_hat) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r2_val = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

            n = len(y)
            k = 1
            if n > k + 1:
                mse = ss_res / (n - k - 1)
                x_var = np.sum((X.ravel() - X.mean()) ** 2)
                stderr = np.sqrt(mse / x_var) if x_var > 0 else np.nan
            else:
                stderr = np.nan

            betas.append(beta)
            r2s.append(r2_val)
            stderrs.append(stderr)

        result = pd.DataFrame({
            "cac_beta": betas,
            "cac_r2": r2s,
            "cac_stderr": stderrs,
        }, index=df.index)

        # Z-score for composite inclusion
        result["cac_zscore"] = (
                (result["cac_beta"] - result["cac_beta"].mean())
                / result["cac_beta"].std()
        )

        logger.info(
            f"CAC estimated: {result['cac_beta'].notna().sum()} valid windows  "
            f"| Mean β = {result['cac_beta'].mean():.3f}  "
            f"Std β = {result['cac_beta'].std():.3f}"
        )
        return result


# ── Composite Credit Pressure Index ───────────────────────────────────────────

class CreditPressureIndex:
    """
    Combine the five indicators into one Credit Pressure Index (CPI).

    Weighting scheme: equal weights across available indicators,
    with all components oriented so HIGHER = MORE credit pressure.
    Final index is standardized to mean=0, std=1.

    This is the composite that feeds the Markov-Switching regime model —
    analogous to the composite "exchange-rate pressure index" used in the
    FX trilemma literature (Aizenman, Chinn & Ito 2013).
    """

    def __init__(
            self,
            credit_gap: pd.Series,
            delinquency_momentum: pd.DataFrame,
            tightening_index: pd.Series,
            chargeoff_acceleration: pd.Series,
            cac: pd.DataFrame,
    ):
        self.credit_gap = credit_gap
        self.delinquency_momentum = delinquency_momentum
        self.tightening_index = tightening_index
        self.chargeoff_acceleration = chargeoff_acceleration
        self.cac = cac

    def build(self) -> tuple[pd.Series, pd.DataFrame]:
        """
        Build the CPI and return (composite_series, all_components_DataFrame).
        """
        components: dict[str, pd.Series] = {}

        # 1. Credit gap (positive = credit expanding faster than income)
        if self.credit_gap is not None and not self.credit_gap.empty:
            components["credit_gap"] = self.credit_gap

        # 2. Delinquency level + velocity (both stress-positive)
        for col in ["delinquency_level", "delinquency_velocity",
                    "delinquency_acceleration"]:
            if col in self.delinquency_momentum.columns:
                components[col] = self.delinquency_momentum[col]

        # 3. Loan tightening (tightening = stress)
        if (
                self.tightening_index is not None
                and not self.tightening_index.empty
        ):
            components["tightening_index"] = self.tightening_index

        # 4. Charge-off acceleration (rising charge-offs = stress)
        if (
                self.chargeoff_acceleration is not None
                and not self.chargeoff_acceleration.empty
        ):
            components["chargeoff_acceleration"] = self.chargeoff_acceleration

        # 5. CAC z-score
        # Note: CAC is ambiguous in direction —
        #   rising CAC (absorbing shocks)   → rising vulnerability
        #   falling CAC (buffer exhausted)  → crisis imminent
        # We use the absolute deviation from the mean as the stress signal.
        if "cac_zscore" in self.cac.columns:
            cac_abs_dev = self.cac["cac_zscore"].abs()
            cac_abs_dev.name = "cac_abs_deviation"
            components["cac_abs_deviation"] = cac_abs_dev

        # Align all components
        comp_df = pd.DataFrame(components).dropna(how="all")

        # Drop columns with > 40% missing
        missing = comp_df.isnull().mean()
        keep = missing[missing <= 0.40].index.tolist()
        comp_df = comp_df[keep]

        # Equal-weight mean across available indicators
        composite = comp_df.mean(axis=1, skipna=True)

        # Re-standardize
        composite = (composite - composite.mean()) / composite.std()
        composite.name = "credit_pressure_index"

        # Drop rows with no data
        composite = composite.dropna()
        comp_df = comp_df.loc[composite.index]

        logger.info(
            f"Credit Pressure Index built: "
            f"{len(composite)} months  "
            f"({composite.index[0].date()} → {composite.index[-1].date()})  "
            f"| Components: {list(comp_df.columns)}"
        )
        return composite, comp_df


# ── Factory function ───────────────────────────────────────────────────────────

def build_all_indicators(df: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """
    Orchestrate construction of all five indicators and the composite CPI.

    Parameters
    ----------
    df : pd.DataFrame
        Output of CreditDataFetcher.fetch_credit_series().

    Returns
    -------
    cpi         : pd.Series   — Composite Credit Pressure Index
    components  : pd.DataFrame — Individual normalized indicators
    cac_detail  : pd.DataFrame — Full CAC output (beta, R², std error, z-score)
    """
    # ── 1. Credit gap ────────────────────────────────────────────────────────
    credit_gap = pd.Series(dtype=float)
    if "revolving_credit" in df.columns and "real_disposable_income" in df.columns:
        gap_est = CreditGapEstimator(df["revolving_credit"], df["real_disposable_income"])
        credit_gap = gap_est.estimate()

    # ── 2. Delinquency momentum ───────────────────────────────────────────────
    delinq_momentum = pd.DataFrame()
    if "cc_delinquency" in df.columns:
        delinq_momentum = DelinquencyMomentumEstimator(df["cc_delinquency"]).estimate()

    # ── 3. Loan tightening index ──────────────────────────────────────────────
    tightening_cols = {}
    for col in ["ci_loan_tightening_large", "ci_loan_tightening_small"]:
        if col in df.columns:
            tightening_cols[col] = df[col]
    tightening_index = LoanTighteningIndexEstimator(tightening_cols).estimate()

    # ── 4. Charge-off acceleration ────────────────────────────────────────────
    chargeoff_accel = pd.Series(dtype=float)
    if "cc_chargeoff" in df.columns:
        chargeoff_accel = ChargeOffAccelerationEstimator(df["cc_chargeoff"]).estimate()

    # ── 5. Credit Absorption Coefficient ─────────────────────────────────────
    cac_detail = pd.DataFrame()
    if "revolving_credit" in df.columns and "real_disposable_income" in df.columns:
        cac_est = CreditAbsorptionCoefficientEstimator(
            df["revolving_credit"],
            df["real_disposable_income"],
        )
        cac_detail = cac_est.estimate()

    # ── Composite ────────────────────────────────────────────────────────────
    cpi_builder = CreditPressureIndex(
        credit_gap=credit_gap,
        delinquency_momentum=delinq_momentum,
        tightening_index=tightening_index,
        chargeoff_acceleration=chargeoff_accel,
        cac=cac_detail,
    )
    cpi, components = cpi_builder.build()
    return cpi, components, cac_detail


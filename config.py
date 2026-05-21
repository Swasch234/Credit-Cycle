"""
config.py — Configuration for the Credit Cycle Regime Detection &
            Early Warning System pipeline.

Methodological note — the bridge to Shan Xue's published work
──────────────────────────────────────────────────────────────
This project extends the offset/sterilization coefficient framework from
Xue & Willett (2024) — originally developed for central-bank monetary
autonomy under open capital accounts — to the household credit domain.

In the FX setting:
  ΔDomesticRate_t = α + β_offset * ΔWorldRate_t + ε_t
  β_offset → 1  means domestic rates fully follow world rates (no autonomy)
  β_offset → 0  means domestic rates are fully autonomous

The analogous consumer-credit framework (the "Credit Absorption Coefficient"):
  ΔRevolvingCredit_t = α + β_CAC * ΔIncomeShock_t + ε_t
  β_CAC > 0  means households actively use credit to absorb income shocks
             (smooth consumption via debt → pressure building)
  β_CAC → 0  means households adjust consumption directly (no credit buffer)
  β_CAC < 0  means credit and income move together (credit amplifies cycles)

A rising β_CAC signals increasing household reliance on revolving credit to
insulate consumption from income volatility — the consumer-credit analogue of
"losing monetary autonomy" in the FX context.

FRED API key: https://fred.stlouisfed.org/docs/api/api_key.html  (free)
"""

import os

FRED_API_KEY = os.environ.get("FRED_API_KEY", "5a425e0b9a778c93a257c410bb51cf83")

# ── Credit Stress Indicators ───────────────────────────────────────────────────
# All raw series are fetched at source frequency; harmonized to monthly in
# data_fetcher.py.  publication_lag_months drives the EWS's real-time simulation.

CREDIT_SERIES = {
    # ── Delinquency & charge-offs ──────────────────────────────────
    "cc_delinquency": {
        "id": "DRCCLACBS",
        "description": "Delinquency Rate on Credit Card Loans, All Commercial Banks (%)",
        "frequency": "Q",
        "publication_lag_months": 1,
        "stress_direction": 1,   # higher = more stress
    },
    "cc_chargeoff": {
        "id": "CORCCACBS",
        "description": "Charge-Off Rate on Credit Card Loans, All Commercial Banks (%)",
        "frequency": "Q",
        "publication_lag_months": 1,
        "stress_direction": 1,
    },
    "consumer_loan_delinquency": {
        "id": "DRCLACBS",
        "description": "Delinquency Rate on Consumer Loans, All Commercial Banks (%)",
        "frequency": "Q",
        "publication_lag_months": 1,
        "stress_direction": 1,
    },
    # ── Debt burden ────────────────────────────────────────────────
    "debt_service_ratio": {
        "id": "CDSP",
        "description": "Consumer Debt Service Payments as % of Disposable Income",
        "frequency": "Q",
        "publication_lag_months": 1,
        "stress_direction": 1,
    },
    "financial_obligations": {
        "id": "FODSP",
        "description": "Financial Obligations Ratio for Homeowners",
        "frequency": "Q",
        "publication_lag_months": 1,
        "stress_direction": 1,
    },
    # ── Credit supply (SLOOS — Senior Loan Officer Opinion Survey) ─
    "ci_loan_tightening_large": {
        "id": "DRTSCILM",
        "description": "Net % Banks Tightening C&I Loan Standards (Large & Medium Firms)",
        "frequency": "Q",
        "publication_lag_months": 0,
        "stress_direction": 1,   # tightening = more stress
    },
    "ci_loan_tightening_small": {
        "id": "DRTSCISM",
        "description": "Net % Banks Tightening C&I Loan Standards (Small Firms)",
        "frequency": "Q",
        "publication_lag_months": 0,
        "stress_direction": 1,
    },
    # ── Credit volumes (for Credit Absorption Coefficient) ─────────
    "revolving_credit": {
        "id": "REVOLSL",
        "description": "Revolving Consumer Credit Outstanding (Millions)",
        "frequency": "M",
        "publication_lag_months": 1,
        "stress_direction": None,   # used in regression, not directly as indicator
    },
    # ── Income & consumption (for Credit Absorption Coefficient) ───
    "real_disposable_income": {
        "id": "DSPIC96",
        "description": "Real Disposable Personal Income (Billions, Chained 2017$)",
        "frequency": "M",
        "publication_lag_months": 1,
        "stress_direction": None,
    },
    "pce": {
        "id": "PCE",
        "description": "Personal Consumption Expenditures (Billions)",
        "frequency": "M",
        "publication_lag_months": 1,
        "stress_direction": None,
    },
    # ── Labor market ───────────────────────────────────────────────
    "unemployment": {
        "id": "UNRATE",
        "description": "Unemployment Rate (%)",
        "frequency": "M",
        "publication_lag_months": 0,
        "stress_direction": 1,
    },
    # ── Credit spread (market-based stress) ────────────────────────
    "hy_spread": {
        "id": "BAMLH0A0HYM2",
        "description": "ICE BofA US High Yield Index Option-Adjusted Spread (%)",
        "frequency": "D",
        "publication_lag_months": 0,
        "stress_direction": 1,
    },
    "bbb_spread": {
        "id": "BAMLC0A4CBBB",
        "description": "ICE BofA BBB US Corporate Bond Option-Adjusted Spread (%)",
        "frequency": "D",
        "publication_lag_months": 0,
        "stress_direction": 1,
    },
}

# ── Validation / Benchmark Series ─────────────────────────────────────────────
BENCHMARK_SERIES = {
    "nber_recession": {
        "id": "USREC",
        "description": "NBER Recession Indicator (1 = recession)",
    },
    "consumer_confidence": {
        "id": "UMCSENT",
        "description": "University of Michigan Consumer Sentiment",
    },
}

# ── Model Settings ─────────────────────────────────────────────────────────────
START_DATE          = "2000-01-01"
END_DATE            = None

N_REGIMES           = 3            # Expansion | Strain | Crisis
EM_ITERATIONS       = 500
SEARCH_REPS         = 30           # more restarts → more robust global optimum

# Credit Absorption Coefficient (rolling regression window)
CAC_WINDOW_MONTHS   = 36           # 3-year rolling window for coefficient estimation
CAC_MIN_PERIODS     = 24           # minimum observations for a valid estimate

# Credit gap (HP filter lambda for monthly data)
HP_LAMBDA           = 129600       # standard for monthly series (Ravn & Uhlig 2002)

# Early Warning System thresholds
EWS_SIGNAL_THRESHOLD = 0.40       # P(Strain or Crisis) > this → yellow alert
EWS_CRISIS_THRESHOLD = 0.65       # P(Crisis) > this → red alert
EWS_HORIZON_MONTHS   = [3, 6, 12] # forecast horizons for EWS evaluation

# ── Output ─────────────────────────────────────────────────────────────────────
OUTPUT_DIR = "output"
FIG_DPI    = 150

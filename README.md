# Credit Cycle Regime Detection & Early Warning System

A Python pipeline that detects consumer credit cycle regimes and generates
early warning signals for credit stress episodes — built on a direct methodological
extension of the offset-coefficient framework from **Xue & Willett (2024)**.

---

## The Intellectual Bridge

**Xue & Willett (2024)** measure short-run monetary autonomy using the *offset coefficient* β:

```
ΔDomesticRate_t = α + β · ΔWorldRate_t + ε_t
```

- β → 1 : domestic rates fully track world rates (no autonomy, e.g. Hong Kong currency board)
- β → 0 : domestic rates move independently (full autonomy, e.g. floating exchange rate)

This project introduces the **Credit Absorption Coefficient (CAC)** — the household-credit analogue:

```
Δlog(RevolvingCredit_t) = α + β_CAC · NegIncomeShock_t + ε_t
```

where `NegIncomeShock_t = max(0, −Δlog(RealIncome_t))` captures downside income shocks.

| FX Context (Xue & Willett 2024) | Credit Context (this project) |
|---|---|
| Central bank monetary autonomy | Household balance sheet resilience |
| Capital flows offsetting rate differentials | Credit absorbing income shortfalls |
| Offset coefficient β (world rate → domestic rate) | CAC β (income shock → credit drawdown) |
| β → 1 = no autonomy (pegged rate) | β > 0 = credit-dependent (shock absorption via debt) |
| Currency board collapse risk | Credit stress / deleveraging episode |
| Impulse response under exchange rate regimes | Impulse response under credit regimes |

A persistently high and rising CAC signals that households are using revolving credit
as a shock absorber — building latent balance-sheet vulnerability that typically
resolves through a delinquency or deleveraging episode.

---

## Five Credit Pressure Indicators

| Indicator | Source | Analogue in FX Work |
|---|---|---|
| **Credit Gap** | HP-filtered credit-to-income ratio deviation | Exchange rate misalignment |
| **Delinquency Momentum** | 2nd derivative of delinquency rate | Acceleration of rate divergence |
| **Loan Tightening Index** | SLOOS composite | Sterilization coefficient |
| **Charge-Off Acceleration** | Rate of change in charge-off rate | Actualized pressure measure |
| **Credit Absorption Coef.** | Rolling OLS (income shock → credit) | Offset coefficient β |

---

## Regime Model

A **Markov-Switching AR(1)** model with switching mean, AR coefficient, and variance
is fitted on the Composite Credit Pressure Index (CPI).  Three regimes are identified:

| Regime | Description | Typical CPI | AR(1) φ | Volatility |
|---|---|---|---|---|
| **Expansion** | Credit growing, delinquency low | Negative | Low (fast mean-reversion) | Low |
| **Strain** | Delinquency rising, standards tightening | Near zero | Moderate | Moderate |
| **Crisis** | Delinquency surging, CAC collapsing | Strongly positive | High (persistent) | High |

Regime-conditional impulse responses show how credit shocks propagate differently
across regimes — in Crisis, shocks are amplified and more persistent, mirroring the
impulse-response analysis in Shan's VAR-based FX research.

---

## Early Warning System

The EWS converts regime probabilities into three alert levels:

- 🟢 **GREEN**: P(Strain or Crisis) ≤ 0.40
- 🟡 **YELLOW**: P(Strain or Crisis) > 0.40
- 🔴 **RED**: P(Crisis) > 0.65

Evaluated at 3-, 6-, and 12-month horizons using AUROC, Brier score, and F1.

---

## Setup

```bash
pip install -r requirements.txt
export FRED_API_KEY="your_key_here"   # free at https://fred.stlouisfed.org
python main.py
```

---

## Project Structure

```
credit_cycle/
├── main.py               Entry point — full pipeline
├── config.py             Series definitions, thresholds, model settings
├── data_fetcher.py       FRED fetcher with frequency harmonization
├── credit_indicators.py  Five indicators + CAC + Composite CPI
├── regime_model.py       Markov-Switching AR, impulse responses, diagnostics
├── early_warning.py      EWS signal generation, AUROC, Brier, lead-time
├── visualizer.py         7 publication-quality charts
├── requirements.txt
└── output/               Generated on run
```

---

## References

- **Xue, S. & Willett, T.D. (2024).** The monetary trilemma need not hold in the short run: The case of Hong Kong. *Journal of International Commerce, Economics and Policy.*
- Hamilton, J.D. (1989). A new approach to the economic analysis of nonstationary time series. *Econometrica.*
- Kim, C.-J. (1994). Dynamic linear models with Markov-switching. *Journal of Econometrics.*
- Drehmann, M. & Tsatsaronis, K. (2014). The credit-to-GDP gap and countercyclical capital buffers. *BIS Quarterly Review.*
- Aizenman, J., Chinn, M. & Ito, H. (2013). The impossible trinity. *Journal of International Money and Finance.*

---

## Author
**Shan Xue, PhD**  
Applied Economist | Quantitative Research Scientist  
shan.xue@cgu.edu  
*Working paper: Monetary Autonomy Under Managed Floating Exchange Rates: Evidence from Singapore*

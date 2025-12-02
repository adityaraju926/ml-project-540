# Realized Volatility Forecasting with Machine Learning Methods

## Project Overview

This project develops machine learning models to forecast next-day realized variance (volatility) for 30 Dow Jones Industrial Average stocks using high-frequency trading data.

## Initial Data

### Dataset: `RV_March2024.xlsx`

**Source:** High-frequency (1-minute and 5-minute) intraday price data for 30 DJIA stocks  
**Time Period:** January 2003 - March 2024 (5,346 trading days, ~21 years)  
**Companies:** 30 Dow Jones constituents (AAPL, MSFT, JPM, GS, BA, etc.)

### Data Structure

The dataset contains **13 sheets** organized as follows:

- **Dates** (5,346 rows): Trading dates from 2003-2024
- **Companies** (30 rows): Ticker symbols for DJIA stocks
- **10 Volatility Measure Sheets** (5,346 × 30 matrices):

### Key Data Characteristics

- **Total observations:** 160,380 (5,346 days × 30 stocks)
- **Missing data:** ~4.24% (encoded as zeros)
- **Variance decomposition:** RV = Good + Bad (continuous + jump components)
- **Heavy-tailed distribution:** Extreme volatility spikes during market crises
- **Volatility clustering:** High volatility periods persist over time

---

## Exploratory Data Analysis (EDA)

The EDA phase focused on understanding volatility dynamics, data quality, and key patterns across the market data.

### Key Findings

#### 1. **Volatility Distribution**
- **Highly right-skewed:** Most days have low volatility, but rare extreme events create long tails
- **Skewness:** 16.3 (RV), 129.9 (RQ) - indicating extreme outliers
- **Kurtosis:** 524 (RV), 20,148 (RQ) - heavy-tailed behavior
- **Implication:** Log-transformation required for modeling

#### 2. **Temporal Patterns**
- **Volatility clustering:** High volatility periods persist
- **Crisis periods:** Major spikes during:
  - 2008 Financial Crisis
  - 2020 COVID-19 pandemic
  - 2022 inflation/rate-hike cycle

#### 3. **Cross-Sectional Patterns**
- **Tech/finance stocks:** Higher average volatility (CRM: 4.94, DOW: 3.96, AMZN: 3.94)
- **Stable dividend stocks:** Lower volatility (WMT: 1.59, KO: 1.61, PG: 1.70)
- **Cross-stock correlation:** Average pairwise correlation = 0.41 (moderate co-movement)

#### 4. **Variance Decomposition**
- **Good vs. Bad variance:** ~50/50 split on average
- **Jump component:** Spikes during market stress
- **Decomposition validation:** RV = Good + Bad holds exactly

#### 5. **Missing Data Patterns**
- **Companies with gaps:** DOW (76%), V (25%), TRV (20%), CRM (7%)
- **Reason:** Later IPO dates or index additions
- **Clean period:** 2019-2024 has complete data for all 30 stocks

---

## Feature Engineering

### Feature Categories

#### 1. **Temporal Features**
Capture time-series dynamics and volatility persistence:

- **Lags:** RV_lag1, RV_lag5, RV_lag10, RV_lag20
- **Rolling statistics (windows: 5, 20, 60 days):**
  - Mean: `RV_roll_mean_5`, `RV_roll_mean_20`, `RV_roll_mean_60`
  - Std: `RV_roll_std_5`, `RV_roll_std_20`, `RV_roll_std_60`
  - Min/Max: `RV_roll_min_*`, `RV_roll_max_*`
- **Momentum:** `RV_momentum_5`, `RV_momentum_20` (% change vs. lagged values)
- **Volatility of volatility:** `RV_volatility_20` (rolling std of RV)

#### 2. **Variance Decomposition Features**
Leverage Good/Bad variance split for jump detection:

- **Ratios:** `Good_Bad_ratio`, `Bad_pct`, `Good_pct`
- **Jump indicators:**
  - `jump_indicator` (1 if Bad > 20% of RV, else 0)
  - `jump_freq_20` (rolling 20-day jump frequency)
  - `jump_intensity` (magnitude of jump component)
- **Cross-frequency:** `jump_diff_freq` (1-min vs 5-min jump comparison)

#### 3. **Cross-Sectional Features**
Compare stock volatility to market-wide measures:

- **Market statistics:** `market_RV_mean`, `market_RV_median`, `market_RV_std`, `market_RV_min`, `market_RV_max`
- **Relative measures:**
  - `RV_vs_market` (stock RV / market average)
  - `RV_zscore` (standardized deviation from market)
  - `RV_rank` (percentile ranking among 30 stocks)
- **Market dispersion:** `market_dispersion`, `market_CV` (coefficient of variation)

#### 4. **Frequency Relationship Features**
Exploit 1-min vs 5-min measure relationships:

- **Ratios:** `RV_freq_ratio`, `BPV_freq_ratio`, `Good_freq_ratio`, `Bad_freq_ratio`
- **Microstructure noise:** `microstructure_noise` (RV - RV_5)
- **Consistency:** `freq_consistency` (agreement between frequencies)

#### 5. **Calendar Features**
Capture seasonal and day-of-week effects:

- **Time components:** `year`, `month`, `quarter`, `day_of_week`, `day_of_month`, `week_of_year`
- **Special days:** `is_monday`, `is_friday`, `is_month_end`

#### 6. **Data Quality Features**
Track missing data and imputation:

- **Missingness flags:** `RV_is_missing`, `BPV_is_missing`, `Good_is_missing`, `Bad_is_missing`
- **Completeness:** `data_completeness_20` (% non-missing in last 20 days)
- **Consecutive gaps:** `consec_missing`

#### 7. **Original Measures**
Raw volatility measures from the dataset:

- 1-min: `BPV`, `Good`, `Bad`, `RQ`
- 5-min: `RV_5`, `BPV_5`, `Good_5`, `Bad_5`, `RQ_5`

#### 8. **Identifiers**
- `Date`, `Ticker`

### Data Preprocessing

1. **Missing data handling:**
   - Forward fill (limit: 3 days)
   - Linear interpolation (limit: 10 days)
   - Drop remaining gaps (4.24% of data)

2. **Train/Validation/Test splits (time-based):**
   - **Training:** 2003-2018 (74.3%, 114,058)
   - **Validation:** 2019-2021 (14.8%, 22,657)
   - **Test:** 2022-2024 (11.0%, 16,860)

3. **Feature transformations:**
   - Log1p transformation for 33 positive-valued features (RV, BPV, lags, rolling stats)
   - StandardScaler for 57 continuous features

### Target Variable

**RV_t+1:** Next-day Realized Variance (log-transformed)

---

## Model Development

### Models Evaluated

| Model | Architecture | Key Hyperparameters |
|-------|--------------|---------------------|
| **Random Forest** | Ensemble of 100 decision trees | max_depth=15, min_samples_split=10 |
| **XGBoost** | Gradient boosting (tree-based) | max_depth=8, learning_rate=0.05, n_estimators=1000 |
| **LightGBM** | Gradient boosting (leaf-wise) | max_depth=8, learning_rate=0.05, n_estimators=1000 |

### Hyperparameter Tuning

**Method:** RandomizedSearchCV with TimeSeriesSplit  
**Best model:** LightGBM with optimized parameters:
- `n_estimators=200`, `max_depth=10`, `learning_rate=0.05`
- `subsample=0.9`, `colsample_bytree=0.7`
- `min_child_samples=20`, `reg_alpha=0.5`, `reg_lambda=0.1`

### Feature Importance (SHAP Analysis)

Top 4 most important features for predicting next-day RV:

1. **RV_roll_mean_5:** 5-day rolling average (captures short-term persistence)
2. **BPV:** Bipower Variation (robust continuous variance estimate)
3. **BPV_5:** 5-minute Bipower Variation (cross-frequency signal)
4. **RV_roll_min_5:** 5-day rolling minimum (baseline volatility level)

---

## Results

### Model Performance (Log-Space Metrics)

| Model | Train R² | Train RMSE | Val R² | Val RMSE | Test R² | Test RMSE |
|-------|----------|------------|--------|----------|---------|-----------|
| **Random Forest** | 0.896 | 0.193 | 0.732 | 0.314 | - | - |
| **XGBoost** | 0.857 | 0.227 | 0.731 | 0.315 | - | - |
| **LightGBM (Baseline)** | 0.818 | 0.256 | **0.749** | 0.304 | - | - |
| **LightGBM (Tuned)** | 0.878 | 0.210 | **0.733** | 0.313 | **0.627** | **0.274** |

### Final Model: Tuned LightGBM

**Validation Performance:**
- **R² = 0.733:** Model explains 73.3% of next-day volatility variance
- **RMSE = 0.313:** Average prediction error in log-space

**Test Performance (2022-2024):**
- **R² = 0.627:** Strong generalization to recent market conditions
- **RMSE = 0.274:** Slightly better error on test set

### Interpretation

- **Strong predictive power:** R² > 0.70 on validation is excellent for financial volatility forecasting
- **Generalization:** Test R² = 0.63 confirms model robustness on unseen data
- **Practical value:** Accurate volatility forecasts enable better risk management, option pricing, and portfolio optimization

---
# Assessment A — Vehicle Price Prediction: Full Implementation Flowchart

> **Purpose:** Interview-ready reference covering every implementation step, decision, and data transformation in Assessment A.

---

## Full Pipeline Flowchart

```mermaid
flowchart TD
    %% ─────────────────────────────────────────
    %% STAGE 1: DATA INGESTION
    %% ─────────────────────────────────────────
    A([START]) --> B[Load train.rpt\n50,703 rows × 130 cols\nTSV format]
    A --> C[Load test.rpt\n11,488 rows × 130 cols\nTSV format]

    B --> D[Replace literal 'NULL'\nstrings with np.nan]
    C --> D2[Replace literal 'NULL'\nstrings with np.nan]

    %% ─────────────────────────────────────────
    %% STAGE 2: DATA CLEANING — TRAIN
    %% ─────────────────────────────────────────
    D --> E{Sold_Amount < $500?}
    E -- Yes\n828 rows --> F[DROP ROW\ndata entry errors /\ninternal transfers]
    E -- No --> G[Keep row\n49,875 rows remain]
    F --> G

    G --> H[Drop BANNED features\n7 columns\nAvgWholesale, AvgRetail,\nGoodWholesale, GoodRetail,\nTradeMin, TradeMax, PrivateMax]

    H --> I[Drop REDUNDANT columns\n12 columns removed]

    I --> I1[MakeCode → duplicate of Make]
    I --> I2[FamilyCode → duplicate of Model]
    I --> I3[DriveCode → duplicate of DriveDescription]
    I --> I4[EngineDescription → rounded,\nless precise than EngineSize]
    I --> I5[KerbWeight → corr=0.99 with TareMass\nTareMass is AU standard]
    I --> I6[VIN, EngineNum → unique IDs\nmemorisation risk]
    I --> I7[Description → composite of\nMake+Model+Series+Badge]
    I --> I8[CurrentRelease, ImportFlag,\nEngineCycleDescription,\nEngineLocation → zero variance]

    I1 & I2 & I3 & I4 & I5 & I6 & I7 & I8 --> J[Drop HIGH-MISSING columns\n threshold > 70% missing\nincludes EV/alt-fuel cols\nfleet is predominantly ICE]

    %% ─────────────────────────────────────────
    %% STAGE 3: FEATURE ENGINEERING
    %% ─────────────────────────────────────────
    J --> K[FEATURE ENGINEERING]

    K --> K1[Parse Sold_Date\n→ Sold_Year\n→ Sold_Month\n→ Sold_Quarter\n→ Is_End_Of_Quarter\n  month ∈ 3,6,9,12\n  dealer clearance flag]

    K --> K2[Age_At_Sale\n= Sold_Year − YearGroup\nvehicle age in years]

    K --> K3[KM Processing\n1. Cap at 99th pct ≈ 243,000 km\n2. Is_High_KM binary flag\n3. KM_Per_Month = KM/Age_Comp_Months]

    K --> K4[Engine & Price Features\n1. EngineSize → float\n2. NewPrice_log = log1p(NewPrice)\n3. KM_Per_NewPrice_K\n   = KM / NewPrice*1000\n   depreciation pressure per $1k MSRP]

    K --> K5[Brand Tier Classification\nluxury: Mercedes, BMW, Audi,\n  Lexus, Porsche, etc.\nbudget: Daewoo, Chery, MG,\n  LDV, Foton, Haval, etc.\nmainstream: all others]

    K --> K6[Drop raw date cols\nSold_Date, Compliance_Date]

    K1 & K2 & K3 & K4 & K5 & K6 --> L[LOG-TRANSFORM TARGET\ny = log1p(Sold_Amount)\nreduces right-skew\nmakes residuals ~normal]

    %% ─────────────────────────────────────────
    %% STAGE 4: TRAIN / VALIDATION SPLIT
    %% ─────────────────────────────────────────
    L --> M[Train/Validation Split\n80% / 20%\nrandom_state=42]
    M --> M1[X_train: 39,901 rows\ny_train: log Sold_Amount]
    M --> M2[X_val: 9,975 rows\ny_val: log Sold_Amount]

    %% ─────────────────────────────────────────
    %% STAGE 5: COLUMN CLASSIFICATION
    %% ─────────────────────────────────────────
    M1 --> N[Classify columns by type]
    N --> N1[Numeric cols: 53\nAll float/int features]
    N --> N2[Low-cardinality cat: 17\nunique values ≤ 15]
    N --> N3[High-cardinality cat: 16\nunique values > 15\ne.g. Make, Model, Series,\nBadge, Colour, Branch...]

    %% ─────────────────────────────────────────
    %% STAGE 6: PREPROCESSING PIPELINE
    %% ─────────────────────────────────────────
    N1 --> O1[Numeric Pipeline\n1. SimpleImputer median\n2. StandardScaler\n   zero-mean unit-variance]

    N2 --> O2[Categorical Pipeline\n1. SimpleImputer most_frequent\n2. OrdinalEncoder\n   unknown → -1]

    N3 --> O3[Target Encoding Pipeline\n1. SimpleImputer most_frequent\n2. TargetEncoder continuous\n   encodes each category as\n   smoothed mean of log-price\n   fitted PER CV FOLD\n   prevents data leakage]

    O1 & O2 & O3 --> P[ColumnTransformer\ncombines all branches\ninto single feature matrix]

    %% ─────────────────────────────────────────
    %% STAGE 7: BASELINE MODEL COMPARISON
    %% ─────────────────────────────────────────
    P --> Q[5-Fold Cross-Validation\nKFold n_splits=5\nshuffled]

    Q --> R1[Ridge Regression\nL2 regularisation\nlinear baseline\nCV R²=0.8685\nsMAPE=1.63%]

    Q --> R2[Random Forest\n100 trees\nbagging ensemble\nCV R²=0.8957\nsMAPE=1.43%]

    Q --> R3[XGBoost\ngradient boosting\ndepth-wise growth\nCV R²=0.9216\nsMAPE=1.22%]

    Q --> R4[LightGBM\ngradient boosting\nleaf-wise growth\nhistogram-based\nCV R²=0.9223\nsMAPE=1.19%]

    R1 & R2 & R3 & R4 --> S{Best model?}
    S -- LightGBM wins\nlowest MAE + highest R² --> T[Select LightGBM\nfor hyperparameter tuning]

    %% ─────────────────────────────────────────
    %% STAGE 8: HYPERPARAMETER TUNING
    %% ─────────────────────────────────────────
    T --> U[Optuna Bayesian Optimisation\nTree-structured Parzen Estimator\n5 trials\nObjective: minimise CV MAE]

    U --> U1[Search space:\nn_estimators: 200–2000\nlearning_rate: 0.01–0.3\nnum_leaves: 20–300\nmax_depth: 3–12\nmin_child_samples: 10–100\nsubsample: 0.5–1.0\ncolsample_bytree: 0.5–1.0\nreg_alpha L1: 0–10\nreg_lambda L2: 0–10]

    U1 --> V[Best LightGBM config\nselected by Optuna]

    %% ─────────────────────────────────────────
    %% STAGE 9: ENSEMBLE
    %% ─────────────────────────────────────────
    V --> W[Build Ensemble\nTuned LightGBM + Baseline XGBoost]

    W --> W1[Fit both models on X_train]
    W1 --> W2[Evaluate both on X_val\nLGBM val R²=0.9213\nXGB  val R²=0.9199]

    W2 --> W3[Compute weights\nproportional to val R²\nw_lgbm = 0.500\nw_xgb  = 0.500]

    W3 --> W4[Ensemble Prediction\nŷ = 0.5 × LGBM_pred\n  + 0.5 × XGB_pred]

    W4 --> W5[Evaluate Ensemble on X_val\nMAE:   0.1089\nRMSE:  0.1784\nR²:    0.9220\nsMAPE: 1.19%]

    %% ─────────────────────────────────────────
    %% STAGE 10: MODEL DIAGNOSTICS
    %% ─────────────────────────────────────────
    W5 --> X[Model Diagnostics]
    X --> X1[Actual vs Predicted\nscatter plot\n~linear correlation]
    X --> X2[Residual plot\ncentered near 0\nsome heteroscedasticity\nat high price range]
    X --> X3[SHAP Feature Importance\nTop drivers:\n1. Make/Model TargetEncoded\n2. Age_At_Sale\n3. NewPrice_log\n4. KM metrics\n5. Engine size]
    X --> X4[Error by Price Bucket\n$10-40k: ~1.2% error\n<$5k or >$80k: ~2% error]

    %% ─────────────────────────────────────────
    %% STAGE 11: MLFLOW TRACKING
    %% ─────────────────────────────────────────
    X1 & X2 & X3 & X4 --> Y[MLflow Experiment Tracking\nexperiment: vehicle_price_v2]

    Y --> Y1[Log per run:\n• All pipeline params\n• CV metrics mean±std\n• Val MAE, RMSE, R², sMAPE\n• Training time seconds\n• Model artifact joblib]

    %% ─────────────────────────────────────────
    %% STAGE 12: FINAL TRAINING & PREDICTION
    %% ─────────────────────────────────────────
    Y1 --> Z[Final Retrain\non ALL labelled data\ntrain + validation combined\nmaximises signal]

    Z --> Z1[Preprocess test.rpt\nsame pipeline\nTransformer fitted on train only\napplied to test]

    Z1 --> Z2[Generate Test Predictions\nEnsemble → log-space preds\nnp.expm1 to invert log transform\nback to dollar values]

    Z2 --> Z3[Test Set Results\n11,488 predictions\nMean actual:    $18,142\nMean predicted: $17,902\nMAE: 0.1253 log units\nRMSE: 0.1915\nR²: 0.9090\nsMAPE: 1.35%]

    Z3 --> Z4[Save to\nlogs/predictions.csv\nCols: Actual_Sold_Amount,\nPredicted_Sold_Amount,\nError, Error_Pct]

    Z4 --> END([END])

    %% ─────────────────────────────────────────
    %% STYLING
    %% ─────────────────────────────────────────
    style A fill:#2d6a4f,color:#fff,stroke:#1b4332
    style END fill:#2d6a4f,color:#fff,stroke:#1b4332
    style F fill:#d62828,color:#fff
    style H fill:#e07a5f,color:#fff
    style I fill:#e07a5f,color:#fff
    style J fill:#e07a5f,color:#fff
    style L fill:#3a86ff,color:#fff
    style S fill:#f4a261,color:#000
    style T fill:#2a9d8f,color:#fff
    style U fill:#2a9d8f,color:#fff
    style W4 fill:#457b9d,color:#fff
    style W5 fill:#457b9d,color:#fff
    style Z3 fill:#264653,color:#fff
    style Z4 fill:#264653,color:#fff
```

---

## Stage-by-Stage Explanation (Interview Reference)

### Stage 1 — Data Ingestion

| Detail | Value |
|--------|-------|
| File format | Tab-separated (`.rpt`) |
| Train size | 50,703 rows × 130 columns |
| Test size | 11,488 rows × 130 columns |
| First action | Replace literal `'NULL'` strings with `np.nan` |

**Why `.rpt` not `.csv`?** It's a vendor export format that uses tabs as delimiters; `pd.read_csv(..., sep='\t')` handles it identically to a TSV.

---

### Stage 2 — Data Cleaning

#### 2a. Remove invalid sale prices
```python
df = df[df['Sold_Amount'] >= 500]
```
- 828 records removed
- Sub-$500 prices represent internal transfers, write-offs, or data entry errors — not real market transactions

#### 2b. Drop banned features (7 columns)
The assessment brief explicitly prohibits these because they **encode average or benchmark prices** — using them would make the model predict prices using prices (data leakage):
- `AvgWholesale`, `AvgRetail`, `GoodWholesale`, `GoodRetail`, `TradeMin`, `TradeMax`, `PrivateMax`

#### 2c. Drop redundant / constant columns (12 columns)

| Column Dropped | Reason |
|----------------|--------|
| `MakeCode` | 1-to-1 alias of `Make` (code vs label) |
| `FamilyCode` | 1-to-1 alias of `Model` |
| `DriveCode` | 1-to-1 alias of `DriveDescription` |
| `EngineDescription` | Rounded version of `EngineSize`; less precise |
| `KerbWeight` | Pearson r = 0.99 with `TareMass`; `TareMass` is Australian standard |
| `VIN`, `EngineNum` | Unique per vehicle — model would memorise, not generalise |
| `Description` | Redundant composite string of Make+Model+Series+Badge |
| `CurrentRelease`, `ImportFlag`, `EngineCycleDescription`, `EngineLocation` | Single unique value (zero variance) |

#### 2d. Drop high-missing columns (>70% NaN)
- Mostly electric / alternative-fuel columns (zero EV fleet share in dataset era)
- Automatic threshold avoids manual selection

---

### Stage 3 — Feature Engineering

**Why engineer features?** Raw columns like dates and odometer readings need transformation to expose the underlying signal that tree models can split on efficiently.

| New Feature | Formula | Signal captured |
|-------------|---------|----------------|
| `Sold_Year` | from `Sold_Date` | General market price level over time |
| `Sold_Month` | from `Sold_Date` | Seasonal patterns |
| `Sold_Quarter` | from `Sold_Date` | Quarterly reporting cycles |
| `Is_End_Of_Quarter` | month ∈ {3,6,9,12} | Dealer clearance discounting |
| `Age_At_Sale` | `Sold_Year − YearGroup` | Primary depreciation driver |
| `KM_cap` | 99th pct ≈ 243k | Reduces outlier influence |
| `Is_High_KM` | binary above cap | Separate signal for extreme usage |
| `KM_Per_Month` | `KM / Age_Comp_Months` | Usage intensity (hard vs easy life) |
| `NewPrice_log` | `log1p(NewPrice)` | Reduces MSRP right-skew |
| `KM_Per_NewPrice_K` | `KM / (NewPrice/1000)` | Depreciation pressure per $1k MSRP |
| `Brand_Tier` | rule-based | Luxury / mainstream / budget segment |

**Why log-transform the target?**
- `Sold_Amount` is right-skewed (long tail of expensive cars)
- Log-transform makes residuals more normally distributed
- MAE in log-space ≈ proportional error in dollar-space
- Prevents large-price vehicles from dominating the loss function

---

### Stage 4 — Train / Validation Split

```python
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.20, random_state=42
)
```
- 80/20 split → 39,901 train / 9,975 validation
- Fixed `random_state=42` for reproducibility
- Stratification not used (continuous target, not classification)

---

### Stage 5 — Preprocessing Pipeline (ColumnTransformer)

Three parallel branches handle different feature types:

```
ColumnTransformer
├─ Numeric (53 cols)
│   ├─ SimpleImputer(strategy='median')   ← robust to outliers
│   └─ StandardScaler()                  ← zero mean, unit variance
│
├─ Low-cardinality categorical (17 cols, ≤15 unique values)
│   ├─ SimpleImputer(strategy='most_frequent')
│   └─ OrdinalEncoder(unknown_value=-1)  ← maps categories to integers
│
└─ High-cardinality categorical (16 cols, >15 unique values)
    ├─ SimpleImputer(strategy='most_frequent')
    └─ TargetEncoder(target_type='continuous', smooth='auto')
        ← replaces category with smoothed mean of log(Sold_Amount)
        ← fitted PER FOLD during cross-validation (no leakage)
```

**Key design decision — TargetEncoder for high-cardinality:**
- `Make`, `Model`, `Series`, `BadgeDescription`, etc. can have hundreds of unique values
- OrdinalEncoder would assign arbitrary integers (no ordering signal)
- One-hot encoding would explode dimensionality (~500+ columns)
- TargetEncoder maps each brand/model to its average log-price → preserves price signal without inflating features

**Leakage prevention:**
```
Cross-validation fold:
  Fold 1 fitting: TargetEncoder fitted on folds 2-5 only
                  Applied to fold 1 for evaluation
  Fold 2 fitting: TargetEncoder fitted on folds 1,3,4,5 only
  ... etc.
```
This ensures the encoder never sees the validation fold during fitting.

---

### Stage 6 — Model Comparison (5-Fold CV)

All four models wrapped in the same `Pipeline([preprocessor, model])`:

| Model | CV MAE | CV R² | Val sMAPE | Key characteristics |
|-------|--------|-------|-----------|-------------------|
| Ridge | 0.15 | 0.8685 | 1.63% | Linear; fast baseline; can't capture interactions |
| Random Forest | 0.13 | 0.8957 | 1.43% | Bagging; robust but slow; depth-wise growth |
| XGBoost | 0.11 | 0.9216 | 1.22% | Boosting; depth-wise; strong regularisation |
| **LightGBM** | **0.11** | **0.9223** | **1.19%** | Boosting; **leaf-wise**; histogram; fastest |

**Why LightGBM wins:**
- Leaf-wise growth (vs depth-wise) finds better splits faster
- Histogram-based binning reduces memory and computation
- Effectively same accuracy as XGBoost but trains significantly faster
- Handles mixed feature types and high cardinality natively

**Why not CatBoost?**
- CatBoost has built-in categorical handling but longer training time
- With our explicit TargetEncoder pre-processing, LightGBM performs equivalently

---

### Stage 7 — Hyperparameter Tuning (Optuna)

```python
study = optuna.create_study(direction='minimize')  # minimise MAE
study.optimize(objective, n_trials=5)
```

**Objective function per trial:**
1. Optuna samples a configuration from the search space
2. A full `Pipeline(preprocessor, LGBMRegressor(**params))` is evaluated with 5-fold CV
3. Mean CV MAE is returned as the objective value
4. Optuna updates its Parzen estimator model and proposes the next trial

**Tuned parameters and their effect:**

| Parameter | Range | Effect |
|-----------|-------|--------|
| `n_estimators` | 200–2000 | More trees = lower bias, higher compute |
| `learning_rate` | 0.01–0.3 | Lower = more trees needed but smoother fit |
| `num_leaves` | 20–300 | Controls model complexity (leaf-wise depth) |
| `max_depth` | 3–12 | Caps tree depth to prevent memorisation |
| `min_child_samples` | 10–100 | Minimum samples per leaf (regularisation) |
| `subsample` | 0.5–1.0 | Row sub-sampling (stochastic boosting) |
| `colsample_bytree` | 0.5–1.0 | Feature sub-sampling per tree |
| `reg_alpha` | 0–10 | L1 regularisation (sparsity) |
| `reg_lambda` | 0–10 | L2 regularisation (weight shrinkage) |

---

### Stage 8 — Ensemble Model

**Why ensemble?**
- Different algorithms make different errors — combining them cancels out individual model biases
- Even when two models have similar accuracy, their errors are not perfectly correlated

**Ensemble construction:**
```python
# 1. Fit both tuned models
lgbm_pipe.fit(X_train, y_train)
xgb_pipe.fit(X_train, y_train)

# 2. Get validation predictions
lgbm_val_pred = lgbm_pipe.predict(X_val)
xgb_val_pred  = xgb_pipe.predict(X_val)

# 3. Evaluate
lgbm_r2 = r2_score(y_val, lgbm_val_pred)  # 0.9213
xgb_r2  = r2_score(y_val, xgb_val_pred)   # 0.9199

# 4. Weighted average (weights proportional to R²)
total = lgbm_r2 + xgb_r2
w_lgbm = lgbm_r2 / total  # ≈ 0.500
w_xgb  = xgb_r2  / total  # ≈ 0.500

# 5. Final prediction
ensemble_pred = w_lgbm * lgbm_val_pred + w_xgb * xgb_val_pred
```

**Ensemble validation metrics:**

| Metric | LightGBM alone | XGBoost alone | Ensemble |
|--------|---------------|---------------|---------|
| MAE | 0.1095 | 0.1112 | **0.1089** |
| RMSE | 0.1793 | 0.1824 | **0.1784** |
| R² | 0.9213 | 0.9199 | **0.9220** |
| sMAPE | 1.22% | 1.24% | **1.19%** |

---

### Stage 9 — Metrics Explained

**Why sMAPE instead of MAPE?**

Standard MAPE: `|y_true - y_pred| / y_true * 100`
- When `y_true → 0`, MAPE → ∞ (blow-up problem)

sMAPE: `2 * |y_true - y_pred| / (|y_true| + |y_pred|) * 100`
- Bounded [0%, 200%]
- Symmetric — penalises over- and under-prediction equally

**Metrics reported:**
| Metric | Meaning |
|--------|---------|
| MAE | Mean absolute error in log-space (≈ proportional dollar error) |
| RMSE | Root mean square error; penalises large errors more |
| R² | Variance explained (0=baseline, 1=perfect) |
| sMAPE | Symmetric percentage error; business-interpretable |

---

### Stage 10 — VehiclePriceTrainer Class

The `src/trainer.py` wraps the entire workflow into a reusable class:

```
VehiclePriceTrainer
├─ __init__(pipeline, experiment_name, cv_folds, models_dir)
│   └─ initialises MLflow experiment + creates model directory
│
├─ fit(X_train, y_train, X_val, y_val, run_name, extra_tags)
│   ├─ Step 1: Validate no banned features in X_train
│   ├─ Step 2: Run k-fold cross-validation → log CV metrics
│   ├─ Step 3: Fit pipeline on full training set
│   ├─ Step 4: Evaluate on X_val → log val metrics
│   └─ Step 5: Save model artifact with joblib → log to MLflow
│
├─ predict(X) → np.ndarray
│   └─ Requires prior fit(); returns log-space predictions
│
├─ evaluate(X, y) → dict{MAE, RMSE, R2, sMAPE}
│
├─ save(path) → Path
│   └─ Persists fitted pipeline with joblib
│
└─ load(path) → VehiclePriceTrainer  [staticmethod]
    └─ Restores previously saved pipeline
```

---

### Stage 11 — Final Training & Test Prediction

```python
# 1. Combine all labelled data
X_full = pd.concat([X_train, X_val])
y_full = pd.concat([y_train, y_val])

# 2. Retrain ensemble on full dataset
# (same architecture, same hyperparams, more data)
lgbm_final.fit(X_full, y_full)
xgb_final.fit(X_full, y_full)

# 3. Preprocess test set
# Note: ColumnTransformer was fitted on train data only
X_test_proc = preprocessor.transform(X_test)

# 4. Predict in log-space
log_preds = ensemble_predict(X_test_proc)

# 5. Invert log transform → dollar values
dollar_preds = np.expm1(log_preds)

# 6. Save results
results_df.to_csv('logs/predictions.csv', index=False)
```

**Test set results:**
| Metric | Value |
|--------|-------|
| Mean actual price | $18,142 |
| Mean predicted price | $17,902 |
| Mean absolute dollar error | ~$1,250 |
| R² | 0.9090 (91% variance explained) |
| sMAPE | 1.35% |

---

## Data Volume Summary (Interview Quick Reference)

```
Raw train data:    50,703 rows × 130 cols
After cleaning:    49,875 rows × 87 cols  (removed 828 rows, 43 cols)
Train split:       39,901 rows
Val split:          9,975 rows
Test data:         11,488 rows × 87 cols

Feature breakdown after preprocessing:
  Numeric features:          53 cols
  Low-cardinality cat:       17 cols
  High-cardinality cat:      16 cols
  ─────────────────────────────────
  Total input features:      86 cols
```

---

## Key Design Decisions Summary

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Target transformation | `log1p(Sold_Amount)` | Normalises right-skewed distribution |
| High-card encoding | TargetEncoder | Preserves price signal without column explosion |
| Leakage prevention | TargetEncoder fitted per CV fold | Prevents validation data bleeding into training |
| Model selection | LightGBM | Best CV metrics; fastest training; leaf-wise growth |
| Final model | Ensemble (LGBM + XGB) | Reduces variance; marginal but consistent improvement |
| Metric reporting | sMAPE alongside MAE/R² | Business interpretable; avoids near-zero blow-up |
| Experiment tracking | MLflow | Reproducible; full parameter + metric history |
| Outlier handling | Cap KM at 99th pct + binary flag | Preserves outlier information without distortion |

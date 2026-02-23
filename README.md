# Datium Data Science Assessment

## Structure

```
datium-assessment/
├── data/
│   ├── train.rpt          # 50,703 labelled vehicle sales records
│   └── test.rpt           # 11,487 records (Sold_Amount present for evaluation)
├── notebooks/
│   ├── 01_eda_and_modelling.ipynb    # Section A – price prediction
│   └── 02_section_b_visual_analysis.ipynb  # Section B – image analysis design
├── src/
│   └── trainer.py         # VehiclePriceTrainer class with MLflow tracking
├── models/                # Saved model artefacts (gitignored)
├── logs/                  # MLflow runs, diagnostic plots, predictions
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running the Notebooks

```bash
jupyter notebook notebooks/01_eda_and_modelling.ipynb
```

Run cells top-to-bottom. MLflow UI can be launched in a separate terminal:

```bash
mlflow ui --backend-store-uri logs/mlruns
```

Then open http://localhost:5000 → click **Training runs** (not Traces) → select the `vehicle_price_v2` experiment.

---

## Section A – Vehicle Price Prediction

**Target:** `Sold_Amount`

**Banned features (per brief):** `AvgWholesale`, `AvgRetail`, `GoodWholesale`, `GoodRetail`, `TradeMin`, `TradeMax`, `PrivateMax`

### Workflow

1. EDA — missing rates, target distribution, outlier analysis, duplicate/NULL-string checks
2. Data cleaning — redundant column removal, zero-price filtering, KM outlier capping
3. Feature engineering — date decomposition, age at sale, KM/month, KM per $1k MSRP, brand tier, end-of-quarter flag, log-NewPrice
4. Preprocessing pipeline — median imputation, OrdinalEncoder (low-cardinality), TargetEncoder (high-cardinality)
5. Model comparison — Ridge, RandomForest, XGBoost, LightGBM (5-fold CV, MLflow tracked)
6. Hyperparameter tuning — Optuna 40-trial search on LightGBM
7. Ensemble — weighted average of tuned LightGBM + XGBoost
8. Diagnostics — residual plots, SHAP feature importance, error-by-price-bucket
9. Final predictions saved to `logs/predictions.csv` (actual vs predicted vs error)

---

### Data Cleaning Decisions

#### Redundant columns removed

Columns were dropped where they were exact aliases, rounded duplicates, or carried zero variance across the dataset. The more informative column of each pair was kept.

| Removed | Kept | Reason |
|---------|------|--------|
| `MakeCode` | `Make` | Perfect 1-to-1 alias (`HOLD` = `Holden`) |
| `FamilyCode` | `Model` | Perfect 1-to-1 alias (`COMMODO` = `Commodore`) |
| `DriveCode` | `DriveDescription` | Perfect 1-to-1 alias (`RWD` = `Rear Wheel Drive`) |
| `EngineDescription` | `EngineSize` | Description is `round(EngineSize/1000, 1)` — loses precision |
| `KerbWeight` | `TareMass` | Pearson r = 0.99; TareMass is the Australian standard |
| `CurrentRelease` | — | Constant value `'F'` across all rows — zero signal |
| `ImportFlag` | — | Constant value `'L'` across all rows — zero signal |
| `EngineCycleDescription` | — | Always `'4 Stroke'` — zero variance |
| `EngineLocation` | — | Always `'Front'` — zero variance |

#### Columns that looked redundant but were kept

| Pair | Why kept |
|------|----------|
| `BodyStyleDescription` / `BodyConfigDescription` | Different hierarchy levels — many body styles map to few configs |
| `GearTypeDescription` / `GearLocationDescription` | Orthogonal — *type* (auto/manual) vs *location* (floor/dash) |
| `Series` / `SeriesModelYear` | Different concepts — many-to-many relationship |
| `Model` / `ModelCode` | ModelCode is variant-level (more granular than Model) |

#### Other cleaning

| Issue | Fix |
|-------|-----|
| Literal `'NULL'` strings | Replaced with `NaN` before any processing |
| `Sold_Amount < $500` | Filtered from training set (data entry errors / internal transfers) |
| KM outliers (>99th pct) | Capped; `Is_High_KM` binary flag added |
| Electric/alt-engine columns | Dropped via >70% missing threshold (fleet is predominantly ICE) |

---

### Feature Engineering

| Feature | Description |
|---------|-------------|
| `Sold_Year`, `Sold_Month`, `Sold_Quarter` | Decomposed from `Sold_Date` |
| `Is_End_Of_Quarter` | Flag for months 3/6/9/12 — captures dealer clearance pricing |
| `Age_At_Sale` | Sale year minus `YearGroup` — vehicle age at point of sale |
| `KM_Per_Month` | `KM / Age_Comp_Months` — average usage intensity |
| `Is_High_KM` | Binary flag for vehicles above the 99th KM percentile |
| `NewPrice_log` | `log1p(NewPrice)` — stabilises right-skewed price distribution |
| `KM_Per_NewPrice_K` | `KM / (NewPrice / 1000)` — depreciation pressure per $1k of MSRP |
| `Brand_Tier` | `luxury` / `mainstream` / `budget` derived from `Make` |

---

### Encoding Strategy

| Column type | Threshold | Encoder |
|-------------|-----------|---------|
| Numeric | — | Median imputation + StandardScaler |
| Low-cardinality categorical | ≤ 15 unique values | OrdinalEncoder |
| High-cardinality categorical | > 15 unique values | TargetEncoder (mean log-price per category, fitted per CV fold — no leakage) |

High-cardinality columns that are now used rather than dropped include `Make`, `Model`, `Series`, `BadgeDescription`, `Colour`, and others.

---

### Metric Note — sMAPE replaces MAPE

The standard MAPE metric was producing astronomically large values (e.g. `524 trillion %`) due to records with near-zero `Sold_Amount`. The fix was two-fold:

1. Filter training rows with `Sold_Amount < $500`
2. Replace MAPE with **sMAPE** (Symmetric MAPE): `2 * |actual - pred| / (|actual| + |pred|) * 100`

sMAPE is bounded and stable even when true values approach zero.

---

### MLflow Experiments

| Experiment | Description |
|-----------|-------------|
| `vehicle_price` | Baseline runs (v1) — original features, OrdinalEncoder only |
| `vehicle_price_v2` | Enhanced runs — redundancy removed, TargetEncoder, Optuna tuning, ensemble |

---

## Section B – Vehicle Image Analysis

See `notebooks/02_section_b_visual_analysis.ipynb` for a full design report covering:
- Problem framing and actionable output signals
- Three modelling approaches (CNN classifier, YOLO detection, Vision-LLM zero-shot) with trade-offs
- Recommended hybrid pipeline architecture
- Evaluation strategy and business metrics
- MVP implementation roadmap

---

## VehiclePriceTrainer

```python
from src.trainer import VehiclePriceTrainer
from sklearn.pipeline import Pipeline

pipe = Pipeline([('pre', preprocessor), ('model', XGBRegressor())])
trainer = VehiclePriceTrainer(pipe, experiment_name='vehicle_price_v2')
trainer.fit(X_train, y_train, X_val=X_val, y_val=y_val)
trainer.evaluate(X_val, y_val)
```

Automatically logs to MLflow per run:
- CV metrics (MAE, RMSE, R², sMAPE — mean and std across folds)
- Validation metrics
- All pipeline parameters
- Fitted model artefact (`.pkl`)

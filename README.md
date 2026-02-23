# Datium Data Science Assessment

## Structure

```
datium-assessment/
├── data/
│   ├── train.rpt          # 50,703 labelled vehicle sales records
│   └── test.rpt           # 11,487 records for inference
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
Then open http://localhost:5000 to compare experiment runs.

## Section A – Vehicle Price Prediction

**Target:** `Sold_Amount`

**Banned features (per brief):** `AvgWholesale`, `AvgRetail`, `GoodWholesale`, `GoodRetail`, `TradeMin`, `TradeMax`, `PrivateMax`

**Workflow:**
1. EDA — missing rates, outlier analysis, target distribution
2. Feature engineering — date decomposition, age at sale, KM/month, log-NewPrice
3. Preprocessing pipeline — median imputation + OrdinalEncoder
4. Model comparison — Ridge, RandomForest, XGBoost, LightGBM (all with 5-fold CV)
5. Diagnostics — residual plots, SHAP feature importance, error-by-price-bucket
6. Final predictions saved to `logs/predictions.csv`

## Section B – Vehicle Image Analysis

See `notebooks/02_section_b_visual_analysis.ipynb` for a full design report covering:
- Problem framing and actionable output signals
- Three modelling approaches (CNN, YOLO, Vision-LLM) with trade-offs
- Recommended hybrid pipeline architecture
- Evaluation strategy and business metrics
- MVP implementation roadmap

## VehiclePriceTrainer

```python
from src.trainer import VehiclePriceTrainer
from sklearn.pipeline import Pipeline

pipe = Pipeline([('pre', preprocessor), ('model', XGBRegressor())])
trainer = VehiclePriceTrainer(pipe, experiment_name='vehicle_price')
trainer.fit(X_train, y_train, X_val=X_val, y_val=y_val)
trainer.evaluate(X_val, y_val)
```

All CV metrics, validation metrics, parameters, and model artefacts are automatically logged to MLflow.

# Exploration Options - Playground Series S6E4: Irrigation Prediction

## Competition Summary
- **Task**: Multi-class classification — predict `Irrigation_Need` (Low / Medium / High)
- **Metric**: Balanced Accuracy (average of per-class recall)
- **Train**: 630,000 rows, **Test**: 270,000 rows
- **No nulls** in either set
- **Class imbalance**: Low ~59% (369,917), Medium ~38% (239,074), High ~3.3% (21,009)
- **Dataset**: Synthetically generated from a deep learning model trained on the original Irrigation Prediction dataset

## Dataset Features (20 total, excluding id)

### Numeric (11)
| Feature | Correlation w/ Target | Notes |
|---------|----------------------|-------|
| Soil_Moisture | 0.455 | Strongest signal |
| Wind_Speed_kmh | 0.258 | Affects evaporation |
| Temperature_C | 0.253 | Affects evaporation |
| Rainfall_mm | 0.111 | Surprisingly low |
| Soil_pH | — | Below 0.1 |
| Organic_Carbon | — | Below 0.1 |
| Electrical_Conductivity | — | Below 0.1 |
| Humidity | — | Below 0.1 |
| Sunlight_Hours | — | Below 0.1 |
| Field_Area_hectare | — | Below 0.1 |
| Previous_Irrigation_mm | — | Below 0.1 |

### Categorical (8)
| Feature | Correlation (OHE) | Notes |
|---------|-------------------|-------|
| Crop_Growth_Stage | ~0.30 per level | All stages similarly correlated |
| Mulching_Used | 0.300 | Yes/No complement |
| Soil_Type | — | Not significant individually |
| Crop_Type | — | Not significant individually |
| Season | — | Not significant individually |
| Irrigation_Type | — | Not significant individually |
| Water_Source | — | Not significant individually |
| Region | — | Not significant individually |

## Key Observations
- Soil_Moisture is the dominant feature (0.455 correlation)
- Crop_Growth_Stage and Mulching_Used are the strongest categorical signals
- Most categorical features (soil type, crop type, region, season, etc.) are weak individually — may contribute through interactions
- High class is very rare (3.3%) — balanced accuracy means getting High right matters as much as the other two classes
- Correlation was computed via ordinal encoding of target (Low=1, Medium=2, High=3); used for exploration only, not for modeling

## Current State
- Pipeline in `common.py`, model code in `xgboost.ipynb`
- XGBoost with native categorical support (`enable_categorical=True`), default hyperparameters
- Mulching_Used mapped to 0/1, target mapped to Low=0, Medium=1, High=2
- Train/test split (80/20, stratified, random_state=123)
- Baseline scores: 98.5% accuracy, **96.1% balanced accuracy** (holdout)

---

## Things to Try

### Baseline (Priority: High)
- [x] Get a baseline XGBoost model working — 96.1% balanced accuracy (holdout)
- [ ] Set up cross-validation with balanced accuracy scoring
- [ ] Establish baseline CV score

### Class Imbalance (Priority: High — given metric)
- [ ] Investigate `sample_weight` or `scale_pos_weight` for the minority class
- [ ] Try oversampling High class (SMOTE or random)
- [ ] Check predicted probability distributions per class — is the model separating High from others?
- [ ] Threshold tuning for class assignment

### Feature Engineering (Priority: Medium)
- [ ] Interaction features (e.g., Soil_Moisture x Temperature, Crop_Growth_Stage x Mulching)
- [ ] Weak categoricals may gain signal through interactions with numeric features
- [ ] Service/feature counts or grouped combinations

### Encoding Strategies (Priority: Medium)
- [x] OHE — used for correlation analysis
- [x] XGBoost native categorical support — used for baseline model
- [ ] Target encoding (with proper CV fold separation)

### Model Options (Priority: Medium)
- [ ] XGBoost
- [ ] LightGBM
- [ ] CatBoost
- [ ] Hyperparameter tuning
- [ ] Ensemble/stacking

### Advanced (Priority: Low)
- [ ] Original dataset blending (original Irrigation Prediction dataset)
- [ ] Pseudo-labeling with confident predictions
- [ ] Feature selection (drop low-importance features)

---

## What We've Learned
- Balanced accuracy = average of per-class recall; each class weighted equally regardless of size
- Getting High class right is critical despite it being only 3.3% of data
- Ordinal encoding for correlation doesn't affect results (shift-invariant) but assumes linear relationship
- Trees don't need ordinal encoding of target — will use raw class labels
- Baseline XGBoost (default params): 98.5% accuracy but 96.1% balanced accuracy — confirms model is weaker on High class
- `sample_weight` expects one weight per sample, not one per class — use `balanced_accuracy_score` for evaluation
- Stratifying train/test split didn't change scores meaningfully at 630K rows — expected, but good practice

## Experiment Log

| # | Description | CV | LB | Notes |
|---|------------|----|----|-------|
| 01 | XGBoost baseline, default params, native categoricals | 0.9611 (holdout) | 0.9590 | Healthy holdout-to-LB gap |
| 02 | Same as 01 but trained on full training set | — | 0.9596 | Small bump from more training data |

See `submission_notes.ipynb` for per-submission details.

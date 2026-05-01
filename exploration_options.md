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
- 10-fold stratified CV: **0.96217 ± 0.00117**

---

## Things to Try

### Baseline (Priority: High)
- [x] Get a baseline XGBoost model working — 96.1% balanced accuracy (holdout)
- [x] Set up 10-fold stratified CV with balanced accuracy scoring
- [x] Establish baseline CV score — 0.96217 ± 0.00117

### Class Imbalance (Priority: High — given metric)
- [x] Investigate `sample_weight` — balanced weights gave +0.007 CV boost (0.9622 → 0.9695)
- [ ] Try oversampling High class (SMOTE or random)
- [ ] Check predicted probability distributions per class — is the model separating High from others?
- [ ] Threshold tuning for class assignment

### Feature Engineering (Priority: High — next focus)
- [ ] Interaction features between top correlated numerics (Soil_Moisture x Temperature, Soil_Moisture x Wind_Speed — evaporation is combinatorial)
- [ ] Crop_Growth_Stage x Mulching interaction — water retention varies by both
- [ ] Ratio features (e.g., Rainfall_mm / Temperature_C as net moisture proxy)
- [ ] Group-based features — weak categoricals (Region, Crop_Type, etc.) combined with numeric features (e.g., mean Soil_Moisture per Region)
- [ ] Weak categoricals may gain signal through interactions with numeric features

### Encoding Strategies (Priority: Medium)
- [x] OHE — used for correlation analysis
- [x] XGBoost native categorical support — used for baseline model
- [ ] Target encoding (with proper CV fold separation)

### Model Options (Priority: Medium)
- [x] XGBoost — primary model
- [ ] LightGBM
- [ ] CatBoost
- [x] Hyperparameter tuning — Optuna with TPE sampler + median pruner; +0.0026 CV gain
- [ ] Ensemble/stacking

### Advanced (Priority: Low)
- [x] Original dataset blending — +0.0007 CV, confirmed on LB
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
- 10-fold CV with 630K rows gives tight std — reliable for comparing experiments
- CV (0.9622) > LB (0.9596) gap is small and expected; track both to confirm they move together
- Balanced sample weights gave biggest single improvement so far (+0.007 CV) — model was undertreating High class
- CV-to-LB correlation is strong: 0.9695 CV → 0.9688 LB; can trust CV for iteration
- Original dataset (10K rows, same columns minus id) adds modest signal despite being small; std increases slightly from distribution differences
- CV can go up while LB goes down — warning sign of overfitting to the training distribution (seen with the forum-shared rule logits)
- Rules/thresholds tuned on the 10K original dataset don't transfer perfectly to the 900K synthetic data (generator introduces noise)
- XGBoost finds its own thresholds from continuous features — pre-engineered boolean flags add nothing. Useful feature engineering encodes things trees *can't* discover: ratios, products, cross-feature aggregations
- `feels_like` (Temperature × Humidity) didn't help — trees can already find any rectangular combination of those features
- Optuna with median pruning is efficient: 30 trials with 5-fold CV during tuning, validated on full 10-fold afterward. The winning trial had shallow trees (max_depth=3) with high regularization
- Final result: CV 0.97300 → public LB 0.97045, private LB 0.97209. Top 25% (1048/4316). CV-private gap was ~0.001, so CV was a faithful proxy for the leaderboard

## Experiment Log

| # | Description | CV | LB | Notes |
|---|------------|----|----|-------|
| 01 | XGBoost baseline, default params, native categoricals | 0.9611 (holdout) | 0.9590 | Healthy holdout-to-LB gap |
| 02 | Same as 01 but trained on full training set | — | 0.9596 | Small bump from more training data |
| — | 10-fold stratified CV baseline | 0.96217 ± 0.00117 | — | Stable, tight std; CV > LB gap is normal |
| 03 | Sample weights (balanced), full training set | 0.96949 ± 0.00158 | 0.96883 | Big jump; CV-to-LB gap very small |
| 04 | + original dataset (10K rows) | 0.97019 ± 0.00221 | 0.96986 | Small gain; original data adds some signal |
| 05 | + forum-shared rule logits (3 classes) + boolean thresholds | 0.97045 | 0.96841 | CV up, LB down — overfitting to original's rules |
| 06 | Only high-class logit + booleans | 0.97010 | 0.96831 | Worse on both — high logit is the worst offender |
| 07 | Booleans only (no logits) | 0.97019 | 0.96986 | Matches best — trees find splits themselves |
| 08 | Optuna-tuned XGBoost (30 trials, 5-fold tuning, validated on 10-fold) | 0.97300 ± 0.00153 | 0.97045 (public) / **0.97209 (private)** | **Final submission**; finished 1048/4316 (top 25%) |

See `submission_notes.ipynb` for per-submission details.

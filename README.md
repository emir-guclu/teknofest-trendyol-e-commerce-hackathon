# Trendyol E-Commerce Hackathon (Teknofest)

Modeling pipeline (feature engineering → gradient boosted ranking/classification → weighted ensemble) that achieved a Top 10% placement (28 / 302) in the competition. The repository intentionally exposes only code – original data files are NOT included.

## Repository Layout

```
trendyol-ecommerce-hackathon/
├── figer_denemeler/
│   ├── *.py , *.ipynb olarak yaptığımız birçok deme dosyasını içerir...
├── notebooks/
│   ├── 01_data_preparation.ipynb      # Multi‑stage feature pipeline (v2 → v5 datasets)
│   ├── 02_model_training.ipynb        # Local (validation) & full (competition) CatBoost training
│   └── 03_submission_generator.ipynb  # Local score estimation + final submission creation
├── config/
│   └── model_config.yaml              # Central paths & (base) model hyperparameters
├── requirements.txt
└── README.md
```

### Dataset Versioning Produced by the Pipeline
The feature pipeline is incremental; successive steps enrich previous versions:

| Version | Produced In Notebook Step | Description (high level) |
|---------|---------------------------|---------------------------|
| v2      | Data Prep – Step 1        | Time‑aware joins (asof) + static content/user meta + category/user/term aggregates |
| v3      | Data Prep – Step 2        | Session price stats, category price normalization, user–term affinity |
| v4      | Data Prep – Step 3        | Text concatenation & reranker (Qwen / MiniLM) scoring features |
| v5      | Data Prep – Final Step    | Additional product & user meta (age, discount), historical user–content interaction counts |

Final training & inference use `train_data_v5.parquet` / `test_data_v5.parquet`.

## Notebook Overview

1. **01_data_preparation.ipynb**
	- Robust, leakage‑safe time series feature generation using `merge_asof`.
	- Cumulative & rate features for content / term / user dimensions.
	- Session statistics, price normalization, user‑term affinity features.
	- External reranker (Qwen3 or MiniLM fallback) to derive semantic relevance score.
	- Additional engineered meta & interaction features → produces v2..v5.

2. **02_model_training.ipynb**
	- Two training regimes:
	  * Local (time split) for rapid iteration & feature importance.
	  * Full (all data) for competition submission models.
	- Trains two CatBoost classifiers (`ordered`, `clicked`) with class balancing via `scale_pos_weight`.

3. **03_submission_generator.ipynb**
	- Local validation: combines ordered & clicked probabilities via a weight `w` (grid search capable), evaluates with custom Trendyol AUC (group metric).
	- Final inference on test set (weight tuned, e.g. `w=0.45`) and creation of `submission.csv`.

## Configuration (`config/model_config.yaml`)
Key fields:
* `data.train_data_path`, `data.test_data_path`: point to latest enriched parquet (v5).
* `data.cat_features_json`: path to JSON list of categorical feature names (exported earlier in pipeline, e.g. v2 stage).
* `model.params`: baseline CatBoost hyperparameters (can be overridden per target if needed).
* `inference.ensemble_weight_w`: blend weight for ordered vs clicked probabilities.
* `output.*`: destination paths for models & submission file.

Adjust the paths to match your local project root (the notebooks currently use absolute placeholders like `C:/Projects/trendyol`). Prefer updating the config + modifying notebooks to read from config for portability.

## Installation

```bash
pip install -r requirements.txt
```

GPU acceleration (recommended for CatBoost + Transformer reranker) requires a proper CUDA / ROCm setup; the code auto‑falls back to CPU for the reranker if GPU not present.

## Running the Pipeline (Repro Order)

1. Open `01_data_preparation.ipynb` and execute sequentially to materialize v2 → v5 parquet datasets.
2. Open `02_model_training.ipynb` and run the local training section (optional) then the full training section to save final CatBoost models.
3. Open `03_submission_generator.ipynb`:
	- (Optional) run local validation weight tuning.
	- Run submission cell to produce `submission.csv`.

## Dependencies (Summary)
Core: pandas, numpy, polars, pyarrow, catboost, torch, transformers, tqdm
Utility / Notebook: jupyter, pyyaml
Optional (notebook exploration / plotting): matplotlib, seaborn

See `requirements.txt` for the authoritative list.

## Custom Metric
Notebook 03 references `trendyol_custom_auc.score` (official competition evaluator). The original file is NOT redistributed.

Included instead:
* `metrics/trendyol_custom_auc_stub.py` – raises `NotImplementedError` so you do not silently run with a wrong metric.
* `approx_group_map` (optional helper) – a rough MAP-like proxy; DO NOT treat as the official leaderboard metric.

To use the real metric locally:
1. Obtain `trendyol_custom_auc.py` from the competition package.
2. Place it at the project root (it is ignored by Git) or import it from its original location.
3. Ensure your notebook import points to the correct module.

## Reproducibility & Notes
* Time ordering is enforced before cumulative features and validation splits to mitigate leakage.
* Categorical features enumerated early (v2) are reused for all later versions—store them in JSON.
* Reranker model defaults to `Qwen/Qwen3-Reranker-0.6B` on GPU; otherwise falls back to `cross-encoder/ms-marco-MiniLM-L-6-v2`.

## License
MIT (feel free to reuse / adapt code; original competition data not distributed).

## Acknowledgements
Thanks to the Teknofest Trendyol Hackathon organizers & community.
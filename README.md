# BERRY-ML

Spectral preprocessing, seven-class raspberry phenology benchmarks, PCA-guided wavelength selection, and classifier-agnostic interval selection. The repository includes the raw spectra, processed spectra, five stratified allocations, and an example set of results.

## 1. Install and activate the environment

Install [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/) or another Conda distribution, then run in a terminal:

```bash
git clone https://github.com/pjafary/BERRY-ML.git
cd BERRY-ML
conda create -n BERRY-ML python=3.11 -y
conda activate BERRY-ML
conda install -c conda-forge numpy pandas scipy scikit-learn optuna joblib -y
```

Verify the packages:

```bash
python -c "import numpy, pandas, scipy, sklearn, optuna, joblib; print('Dependencies available')"
```

The scripts use only the Python standard library and these six installed packages. Saved `*_optimized.json` files record the Python, NumPy, pandas, scikit-learn, and Optuna versions used for each run. A checked-in example records Python 3.11.15, NumPy 2.4.6, pandas 3.0.1, scikit-learn 1.9.0, and Optuna 5.0.0. When loading an existing `.joblib` model, use a compatible environment; retraining in a newer environment is preferable if compatibility is uncertain.

## 2. Understand the input files

The `Data/` directory contains:

- `raw_data.csv`: original reflectance spectra.
- `processed_by_moving_average.csv`, `processed_by_sg_smoothing.csv`, `processed_by_sg_first_derivative.csv`, `processed_by_sg_second_derivative.csv`, and `processed_by_snv.csv`: alternative preprocessing outputs.
- `allocation_1.csv` through `allocation_5.csv`: the fixed train/validation/test assignments.

Spectral CSV files have wavelengths in the **first column** and one spectrum per subsequent **sample-ID column**. Allocation CSV files contain `sample_id`, `stage`, and `split`; `split` is `train`, `validation`, or `test`. Sample IDs in a spectral CSV must match those in the allocation files. Each allocation has 12 training, 4 validation, and 4 test spectra for each of seven stages (84/28/28 in total). Each of the 140 samples is in the test set of exactly one allocation.

All scripts resolve paths relative to the directory containing the script. Run the commands below from the repository root; alternatively use absolute paths in the configuration blocks.

## 3. Preprocess spectra (optional)

The processed CSVs are already included. To recreate them, edit the constants at the beginning of `preprocess_spectra.py`:

```python
INPUT_CSV = SCRIPT_DIR / "Data" / "raw_data.csv"
OUTPUT_DIR = SCRIPT_DIR / "Data"
OUTPUT_PREFIX = "processed_by"
METHODS = ["moving_average", "sg_smoothing", "sg_first_derivative", "snv", "sg_second_derivative"]
MOVING_AVERAGE_WINDOW = 3
SG_WINDOW = 5
SG_POLYORDER = 2
```

Then run:

```bash
python preprocess_spectra.py
```

Each method is applied independently to the raw spectrum. Moving average uses a three-point window; Savitzky–Golay smoothing and derivatives use a five-point, second-order polynomial. A preprocessing settings JSON is written alongside the output CSVs. The benchmarks below use the supplied moving-average CSV unless a different preprocessing method is explicitly selected.

## 4. Run the full-spectrum baseline

In `full_spectrum_benchmark.py`, set:

```python
DATA_CSV = SCRIPT_DIR / "Data" / "processed_by_moving_average.csv"
ALLOCATIONS_DIR = SCRIPT_DIR / "Data"
ALLOCATION_PATTERN = "allocation_*.csv"
OUTPUT_DIR = SCRIPT_DIR / "Results" / "full_spectrum_moving_average"
OUTPUT_PREFIX = "full_spectrum_moving_average"
INFERENCE_MODEL_PATH = None
```
You can run the file directly in your code editor or through you terminal by: 

```bash
python full_spectrum_benchmark.py
```

This script trains all seven classifiers on every measured wavelength, with no external wavelength selection. Note that PCA-LDA still fits PCA inside its own classifier pipeline, but it will have access to full spectral data.

## 5. Run PCA-based band selection

In `pca_band_benchmark.py`, confirm the same `DATA_CSV` and allocation directory, and set a separate output directory:

```python
DATA_CSV = SCRIPT_DIR / "Data" / "processed_by_moving_average.csv"
OUTPUT_DIR = SCRIPT_DIR / "Results" / "pca_moving_average"
OUTPUT_PREFIX = "pca_moving_average"
N_SELECTION_PCS = 6
N_WAVELENGTHS_PER_PC = 6
INFERENCE_MODEL_PATH = None
```
You can run the code in the terminal by:

```bash
python pca_band_benchmark.py
```

For each allocation, the script fits a pooled PCA using **training spectra only**, finds the largest absolute loading in each of the first six PCs, and takes the six measured wavelengths nearest each peak. Overlapping wavelengths are included once. Classifiers are trained on reflectance values at those wavelengths, **not on PCA scores**. The script also saves pooled and per-class training PCA means, scores, loadings, and explained-variance tables.

## 6. Run the classifier-agnostic interval experiments

Edit `classifier_agnostic_domain_optimization.py` before **each** experiment. Keep `DATA_CSV = SCRIPT_DIR / "Data" / "processed_by_moving_average.csv"`, `ALLOCATIONS_DIR = SCRIPT_DIR / "Data"`, and `ALLOCATION_PATTERN = "allocation_*.csv"`. Give every configuration a different `OUTPUT_DIR` and `OUTPUT_PREFIX` so results do not overwrite each other.

### Two-region search

```python
OUTPUT_DIR = SCRIPT_DIR / "Results" / "two_region_moving_average"
OUTPUT_PREFIX = "two_region_moving_average"
DOMAIN_SEARCH = [
    {"name": "domain_1", "min_nm": 350, "max_nm": 750, "width_nm": 40, "step_nm": 10},
    {"name": "domain_2", "min_nm": 750, "max_nm": 1550, "width_nm": 40, "step_nm": 10},
]
```

### Three-region search

```python
OUTPUT_DIR = SCRIPT_DIR / "Results" / "three_region_moving_average"
OUTPUT_PREFIX = "three_region_moving_average"
DOMAIN_SEARCH = [
    {"name": "domain_1", "min_nm": 350, "max_nm": 750, "width_nm": 40, "step_nm": 10},
    {"name": "domain_2", "min_nm": 750, "max_nm": 1150, "width_nm": 40, "step_nm": 10},
    {"name": "domain_3", "min_nm": 1150, "max_nm": 1550, "width_nm": 40, "step_nm": 10},
]
```

### Fixed band intervals
If you have one or more specific wavelength ranges that you want to test, you can adjust the `min_nm` ,  `max_nm` , `width_nm` , and `step_nm` to restrict the algorithm accordingly.

```python
OUTPUT_DIR = SCRIPT_DIR / "Results" / "cross_berry_moving_average"
OUTPUT_PREFIX = "cross_berry_moving_average"
DOMAIN_SEARCH = [
    {"name": "domain_1", "min_nm": 520, "max_nm": 560, "width_nm": 40, "step_nm": 10},
    {"name": "domain_2", "min_nm": 680, "max_nm": 720, "width_nm": 40, "step_nm": 10},
    {"name": "domain_3", "min_nm": 770, "max_nm": 810, "width_nm": 40, "step_nm": 10},
]
```

When `min_nm + width_nm == max_nm`, each domain has just one candidate interval. Thus the last configuration evaluates fixed intervals rather than searching for their locations. Replace the entire `DOMAIN_SEARCH` list with the desired configuration.

After each edit, run:

```bash
python classifier_agnostic_domain_optimization.py
```

The interval script evaluates all combinations of one candidate interval from each domain. It computes per-wavelength between-class/within-class (Fisher) relevance on the training samples, averages and normalizes relevance within each search domain, and subtracts a weighted mean absolute inter-interval Pearson correlation (`REDUNDANCY_WEIGHT = 0.25`). The highest-scoring combination is selected without fitting or comparing classifiers during band selection. All seven classifiers are then trained on the measured reflectance values inside the selected intervals. `MAX_GRID_CANDIDATES` limits the number of combinations allowed.

## 7. How model training and evaluation work

Each benchmark follows the same five allocation files:

1. Select features using only the training portion where applicable.
2. Optimize each classifier's hyperparameters with Optuna's TPE sampler (`N_TRIALS = 60` by default). Every trial fits on training data and maximizes macro-F1 on validation data.
3. With `REFIT_TRAIN_VALIDATION = True`, refit the selected configuration using the combined training and validation samples.
4. Evaluate once on that allocation's held-out test set, save the fitted model, and repeat for the remaining allocations.

The seven classifiers are PLS-DA, shrinkage LDA, PCA-LDA, linear SVM, RBF-SVM, logistic regression, and random forest. Scaling and estimator fitting occur in model pipelines; random forest does not use the common scaler. The five test sets do not overlap. Reported means and sample standard deviations are calculated across allocations; pooled confusion matrices can be assembled from the saved test predictions if each sample is counted once.

## 8. Find the outputs

Each script writes aggregate CSVs under its configured `OUTPUT_DIR`:

| File suffix | Contents |
| --- | --- |
| `_overall_by_fold.csv` | Test macro-precision, macro-recall, macro-F1, and overall accuracy for each model and allocation |
| `_overall_mean_std.csv` | Mean and sample standard deviation of those metrics across five allocations |
| `_per_class_by_fold.csv` and `_per_class_mean_std.csv` | Class-specific results by allocation and their summaries |
| `_test_predictions.csv` | Sample ID, actual stage, and predicted stage for each held-out prediction |
| `_selected_features_by_fold.csv` (or `_feature_inventory_by_fold.csv` for full spectrum) | Selected wavelength inventory per allocation |

Within each `<OUTPUT_PREFIX>_allocation_<n>/` subdirectory, look for `selected_features.json`, `<model>_optimized.json`, `<model>_trials.csv`, `<model>_model.joblib`, and count and row-normalized test confusion CSVs. The PCA run adds loading, score, variance, and PC-band files. The interval run adds `selected_bands.csv`, `feature_grid.csv`, `top_20_training_candidates.csv`, and aggregate band-frequency CSVs.

The repository currently contains an example `Results/cross-berry_mov_ave_v1/` run. Its saved configuration used the fixed 520–560, 680–720, and 770–810 nm intervals. The default domain script has since been set to a different third interval; configure the list above to reproduce that example. Other result directories are created by running the corresponding experiments.

## 9. Predict stages for new spectra

Use the same preprocessing method and wavelength grid as the saved model. The inference CSV must have wavelengths in its first column and new sample IDs as the remaining column headers. In the matching benchmark script, set, for example:

```python
INFERENCE_MODEL_PATH = OUTPUT_DIR / f"{OUTPUT_PREFIX}_allocation_1" / "PLS_DA_model.joblib"
INFERENCE_CSV = SCRIPT_DIR / "Data" / "new_spectra_processed_by_moving_average.csv"
INFERENCE_OUTPUT_CSV = OUTPUT_DIR / "new_predictions.csv"
```

Run that script again. When `INFERENCE_MODEL_PATH` is set, it loads the chosen saved model and writes `sample_id,predicted_stage` to `INFERENCE_OUTPUT_CSV`.

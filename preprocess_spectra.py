from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from scipy.signal import savgol_filter

SCRIPT_DIR = Path(__file__).resolve().parent

INPUT_CSV = SCRIPT_DIR / "Data" / "raw_data.csv"
OUTPUT_DIR = SCRIPT_DIR / "Data"
OUTPUT_PREFIX = "processed_by"
FLOAT_FORMAT = "%.10g"
METHODS = ["moving_average", "sg_smoothing", "sg_first_derivative", "snv", "sg_second_derivative", "msc"]

MOVING_AVERAGE_WINDOW = 3
SG_WINDOW = 5
SG_POLYORDER = 2
SNV_DDOF = 1
EPSILON = 1e-12

MSC_ALLOCATIONS_DIR = SCRIPT_DIR
MSC_ALLOCATION_PATTERN = "allocation_*.csv"
MSC_TRAIN_SAMPLE_IDS = []
MSC_REFERENCE_CSV = None


def load_spectra(path):
    frame = pd.read_csv(path)
    if frame.shape[1] < 2:
        raise ValueError("Expected wavelength column followed by sample columns")
    wavelengths = pd.to_numeric(frame.iloc[:, 0], errors="raise").to_numpy(float)
    sample_ids = frame.columns[1:].astype(str).to_numpy()
    spectra = frame.iloc[:, 1:].to_numpy(float).T
    if not np.isfinite(wavelengths).all() or not np.isfinite(spectra).all():
        raise ValueError("Input contains missing or nonfinite values")
    if len(np.unique(wavelengths)) != len(wavelengths):
        raise ValueError("Duplicate wavelengths")
    order = np.argsort(wavelengths)
    return frame.columns[0], wavelengths, sample_ids, spectra[:, order], order


def validate_window(window, n, name):
    if not isinstance(window, int) or window < 1 or window % 2 == 0 or window > n:
        raise ValueError(f"{name} must be an odd positive integer no larger than {n}")


def snv(X):
    if X.shape[1] <= SNV_DDOF:
        raise ValueError("Too few wavelengths for SNV")
    std = X.std(axis=1, ddof=SNV_DDOF, keepdims=True)
    if np.any(std <= EPSILON):
        raise ValueError("SNV is undefined for a constant or near-constant spectrum")
    return (X-X.mean(axis=1, keepdims=True))/std


def msc(X, reference):
    reference = np.asarray(reference, dtype=float)
    if reference.shape != (X.shape[1],) or not np.isfinite(reference).all():
        raise ValueError("Invalid MSC reference")
    centered = reference-reference.mean()
    denominator = centered @ centered
    if denominator <= EPSILON:
        raise ValueError("MSC reference has insufficient variation")
    slopes = ((X-X.mean(axis=1, keepdims=True)) @ centered)/denominator
    if np.any(abs(slopes) <= EPSILON):
        raise ValueError("MSC slope is zero or near zero")
    intercepts = X.mean(axis=1)-slopes*reference.mean()
    return (X-intercepts[:, None])/slopes[:, None]


def reference_sets(ids, X, sorted_wavelengths):
    sources = int(MSC_ALLOCATIONS_DIR is not None)+int(bool(MSC_TRAIN_SAMPLE_IDS))+int(MSC_REFERENCE_CSV is not None)
    if sources != 1:
        raise ValueError("For MSC, configure exactly one reference source: allocations directory, training sample IDs, or saved reference CSV")
    if MSC_REFERENCE_CSV is not None:
        frame = pd.read_csv(MSC_REFERENCE_CSV)
        wave = frame.iloc[:, 0].to_numpy(float)
        order = np.argsort(wave)
        if len(wave) != len(sorted_wavelengths) or not np.allclose(wave[order], sorted_wavelengths, rtol=0, atol=1e-8):
            raise ValueError("MSC reference wavelengths do not match input")
        return [("saved_reference", frame.iloc[:, 1].to_numpy(float)[order], [])]
    if MSC_TRAIN_SAMPLE_IDS:
        groups = [("training_reference", list(MSC_TRAIN_SAMPLE_IDS))]
    else:
        paths = sorted(Path(MSC_ALLOCATIONS_DIR).glob(MSC_ALLOCATION_PATTERN))
        if not paths:
            raise ValueError("No allocation CSVs found; set MSC_ALLOCATIONS_DIR correctly")
        groups = []
        for path in paths:
            allocation = pd.read_csv(path, usecols=["sample_id", "split"], dtype=str)
            if allocation.sample_id.duplicated().any() or set(allocation.sample_id) != set(ids):
                raise ValueError(f"{path}: allocations must match all source sample IDs exactly once")
            if not set(allocation['split']).issubset({'train', 'validation', 'test'}):
                raise ValueError(f"{path}: unknown split label")
            groups.append((path.stem, allocation.loc[allocation['split']=='train', 'sample_id'].tolist()))
    references = []
    for name, training_ids in groups:
        if not training_ids or len(set(training_ids)) != len(training_ids) or not set(training_ids).issubset(set(ids)):
            raise ValueError(f"Invalid training sample IDs for {name}")
        selected = np.isin(ids, training_ids)
        references.append((name, X[selected].mean(axis=0), training_ids))
    return references


def main():
    allowed = {'moving_average', 'sg_smoothing', 'sg_first_derivative', 'snv', 'sg_second_derivative', 'msc'}
    if not METHODS or set(METHODS)-allowed:
        raise ValueError("Unknown or empty METHODS")
    label, wavelengths, ids, X, order = load_spectra(INPUT_CSV)
    sorted_wavelengths = wavelengths[order]
    inverse_order = np.argsort(order)
    if 'moving_average' in METHODS:
        validate_window(MOVING_AVERAGE_WINDOW, X.shape[1], 'MOVING_AVERAGE_WINDOW')
    sg_methods = {'sg_smoothing': 0, 'sg_first_derivative': 1, 'sg_second_derivative': 2}
    sg_requested = set(METHODS) & set(sg_methods)
    delta = None
    if sg_requested:
        validate_window(SG_WINDOW, X.shape[1], 'SG_WINDOW')
        if not isinstance(SG_POLYORDER, int) or not 0 <= SG_POLYORDER < SG_WINDOW:
            raise ValueError("SG_POLYORDER must be a nonnegative integer below SG_WINDOW")
        if SG_POLYORDER < max(sg_methods[k] for k in sg_requested):
            raise ValueError("SG_POLYORDER must be at least the highest requested derivative order")
        differences = np.diff(sorted_wavelengths)
        if len(differences) == 0 or not np.allclose(differences, differences[0], rtol=1e-6, atol=1e-8):
            raise ValueError("Savitzky-Golay requires a uniformly spaced wavelength grid")
        delta = float(differences[0])
    references = reference_sets(ids, X, sorted_wavelengths) if 'msc' in METHODS else []
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = []

    def save(name, values):
        if not np.isfinite(values).all():
            raise ValueError(f"Nonfinite output: {name}")
        frame = pd.DataFrame(values[:, inverse_order].T, columns=ids)
        frame.insert(0, label, wavelengths)
        path = OUTPUT_DIR/f'{OUTPUT_PREFIX}_{name}.csv'
        frame.to_csv(path, index=False, float_format=FLOAT_FORMAT)
        outputs.append(str(path)); print(path)

    for method in METHODS:
        if method == 'moving_average':
            save(method, uniform_filter1d(X, size=MOVING_AVERAGE_WINDOW, axis=1, mode='reflect'))
        elif method in sg_methods:
            save(method, savgol_filter(X, SG_WINDOW, SG_POLYORDER, deriv=sg_methods[method], delta=delta, axis=1, mode='interp'))
        elif method == 'snv':
            save(method, snv(X))
        elif method == 'msc':
            for name, reference, training_ids in references:
                save(f'msc_{name}', msc(X, reference))
                reference_path = OUTPUT_DIR/f'{OUTPUT_PREFIX}_msc_reference_{name}.csv'
                pd.DataFrame({label:sorted_wavelengths, 'reference':reference}).to_csv(reference_path, index=False, float_format='%.17g')
                (OUTPUT_DIR/f'{OUTPUT_PREFIX}_msc_reference_{name}.json').write_text(json.dumps({'training_sample_ids':training_ids, 'reference_source':str(MSC_REFERENCE_CSV) if MSC_REFERENCE_CSV else 'mean_training_spectrum'}, indent=2))
    settings = dict(input=str(INPUT_CSV), methods=METHODS, moving_average_window=MOVING_AVERAGE_WINDOW,
                    sg_window=SG_WINDOW, sg_polyorder=SG_POLYORDER, wavelength_step_nm=delta,
                    snv_ddof=SNV_DDOF, outputs=outputs,
                    derivative_units='input reflectance units per nm or per nm^2',
                    processing='Each method independently applied to raw spectra; original row order restored')
    (OUTPUT_DIR/f'{OUTPUT_PREFIX}_preprocessing_settings.json').write_text(json.dumps(settings, indent=2))


if __name__ == '__main__':
    main()

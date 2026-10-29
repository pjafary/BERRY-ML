import csv
import hashlib
import itertools
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import optuna
import sklearn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_CSV = SCRIPT_DIR / "Data" / "processed_by_moving_average.csv"
ALLOCATIONS_DIR = SCRIPT_DIR / "Data"
ALLOCATION_PATTERN = "allocation_*.csv"
OUTPUT_DIR = SCRIPT_DIR / "Results" / "cross-berry_mov_ave_v2"
OUTPUT_PREFIX = "cross-berry_mov_ave_v2"

MODELS = ["PLS_DA", "Shrinkage_LDA", "PCA_LDA", "Linear_SVM", "RBF_SVM", "Logistic_Regression", "Random_Forest"]

SEED = 42
N_TRIALS = 60
REFIT_TRAIN_VALIDATION = True
MAX_GRID_CANDIDATES = 1000000
REDUNDANCY_WEIGHT = 0.25
EPSILON = 1e-12

INFERENCE_MODEL_PATH = None
INFERENCE_CSV = OUTPUT_DIR / "new_spectra.csv"
INFERENCE_OUTPUT_CSV = OUTPUT_DIR / "cross-berry_mov_ave_v2_predictions.csv"

DOMAIN_SEARCH = [
    {"name": "domain_1", "min_nm": 520, "max_nm": 560, "width_nm": 40, "step_nm": 10},
    {"name": "domain_2", "min_nm": 680, "max_nm": 720, "width_nm": 40, "step_nm": 10},
    #{"name": "domain_3", "min_nm": 770, "max_nm": 810, "width_nm": 40, "step_nm": 10},
    {"name": "domain_3", "min_nm": 1110, "max_nm": 1150, "width_nm": 40, "step_nm": 10},

]


class PLSDA(ClassifierMixin, BaseEstimator):
    def __init__(self, n_components=3):
        self.n_components = n_components

    def fit(self, X, y):
        self.classes_, encoded = np.unique(y, return_inverse=True)
        target = np.eye(len(self.classes_))[encoded]
        self.n_features_in_ = X.shape[1]
        self.model_ = PLSRegression(n_components=min(self.n_components, X.shape[1], len(X)-1), scale=False, max_iter=1000)
        self.model_.fit(X, target)
        return self

    def predict(self, X):
        return self.classes_[np.argmax(self.model_.predict(X), axis=1)]


def model(name, params):
    if name == "PLS_DA":
        estimator = PLSDA(**params)
    elif name == "Shrinkage_LDA":
        estimator = LinearDiscriminantAnalysis(solver="lsqr", **params)
    elif name == "PCA_LDA":
        return make_pipeline(StandardScaler(), PCA(n_components=params.get("variance", .99), svd_solver="full"), LinearDiscriminantAnalysis(solver="lsqr", shrinkage=params.get("shrinkage", "auto")))
    elif name in ("Linear_SVM", "RBF_SVM"):
        estimator = SVC(kernel="linear" if name == "Linear_SVM" else "rbf", **params)
    elif name == "Logistic_Regression":
        estimator = LogisticRegression(solver="lbfgs", max_iter=10000, **params)
    elif name == "Random_Forest":
        return RandomForestClassifier(random_state=SEED, n_jobs=-1, **params)
    else:
        raise ValueError(name)
    return make_pipeline(StandardScaler(), estimator)


def suggest(trial, name, n_features, n_train):
    if name == "PLS_DA":
        return {"n_components": trial.suggest_int("n_components", 1, min(15, n_features, n_train-1))}
    if name == "Shrinkage_LDA":
        mode = trial.suggest_categorical("shrinkage_mode", ["auto", "numeric"])
        return {"shrinkage": "auto" if mode == "auto" else trial.suggest_float("shrinkage", 0.0, 1.0)}
    if name == "PCA_LDA":
        return {"variance": trial.suggest_float("variance", .80, .9999), "shrinkage": "auto"}
    if name in ("Linear_SVM", "RBF_SVM", "Logistic_Regression"):
        result = {"C": trial.suggest_float("C", 1e-4, 1e4, log=True)}
        if name == "RBF_SVM":
            result["gamma"] = trial.suggest_float("gamma", 1e-5, 10., log=True)
        return result
    return {"n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
            "max_depth": trial.suggest_categorical("max_depth", [None, 3, 5, 10, 20]),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", .5, 1.0])}


def load_data(path):
    raw = pd.read_csv(path)
    wavelengths = pd.to_numeric(raw.iloc[:, 0], errors="raise").to_numpy(float)
    if len(np.unique(wavelengths)) != len(wavelengths):
        raise ValueError("Duplicate wavelengths")
    ids = raw.columns[1:].astype(str).to_numpy()
    X = raw.iloc[:, 1:].to_numpy(float).T
    if not np.isfinite(X).all() or len(set(ids)) != len(ids):
        raise ValueError("Nonfinite reflectance or duplicate sample IDs")
    return wavelengths, ids, X


def pack_fitted_model(fitted):
    if isinstance(fitted, Pipeline) and isinstance(fitted.steps[-1][1], PLSDA):
        return {"kind": "plsda", "scaler": fitted.steps[0][1],
                "regression": fitted.steps[-1][1].model_,
                "classes": fitted.steps[-1][1].classes_}
    return {"kind": "pipeline", "pipeline": fitted}


def predict_fitted(packed, features):
    if packed["kind"] == "plsda":
        scaled = packed["scaler"].transform(features)
        return packed["classes"][np.argmax(packed["regression"].predict(scaled), axis=1)]
    return packed["pipeline"].predict(features)


def inference():
    artifact = joblib.load(INFERENCE_MODEL_PATH)
    if artifact["feature_type"] != "domain":
        raise ValueError("This script only loads domain model artifacts")
    waves, ids, X = load_data(INFERENCE_CSV)
    expected = np.asarray(artifact["source_wavelengths_nm"], dtype=float)
    if len(waves) != len(expected) or not np.array_equal(np.sort(waves), np.sort(expected)):
        raise ValueError("Inference wavelength grid differs from the training data")
    row = {float(w): i for i, w in enumerate(waves)}
    X = X[:, [row[float(w)] for w in expected]]
    pred = predict_fitted(artifact["fitted_model"], transform(X, expected, artifact["specification"]))
    INFERENCE_OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"sample_id": ids, "predicted_stage": pred}).to_csv(INFERENCE_OUTPUT_CSV, index=False)
    print(f"Predictions written to {INFERENCE_OUTPUT_CSV}")


def load_allocation(path, ids):
    frame = pd.read_csv(path, usecols=["sample_id", "stage", "split"], dtype=str)
    if frame.sample_id.duplicated().any() or set(frame.sample_id) != set(ids):
        raise ValueError(f"{path}: allocation must contain each source sample exactly once")
    frame = frame.set_index("sample_id").loc[ids]
    if set(frame['split']) != {"train", "validation", "test"}:
        raise ValueError("Expected train/validation/test labels")
    counts = pd.crosstab(frame.stage, frame['split'])
    if not (counts[["train", "validation", "test"]].to_numpy() == [12, 4, 4]).all():
        raise ValueError(f"Expected 12/4/4 samples per stage; got {counts}")
    y = frame.stage.to_numpy()
    return y, {role: np.flatnonzero(frame['split'].to_numpy() == role) for role in ("train", "validation", "test")}


def summarize(frame, keys):
    values = ["precision", "recall", "f1", "accuracy"]
    result = frame.groupby(keys, sort=False)[values].agg(["mean", "std"])
    result.columns = ["_".join(c) for c in result.columns]
    return result.reset_index()


def evaluate(y, pred, labels, fold, name):
    p, r, f, support = precision_recall_fscore_support(y, pred, labels=labels, zero_division=0)
    cm = confusion_matrix(y, pred, labels=labels)
    n = cm.sum()
    rows = []
    for i, label in enumerate(labels):
        tp = cm[i, i]; fn = cm[i].sum()-tp; fp = cm[:, i].sum()-tp
        tn = n-tp-fn-fp
        rows.append(dict(fold=fold, model=name, stage=label, precision=p[i], recall=r[i], f1=f[i], accuracy=(tp+tn)/n, support=int(support[i])))
    overall = dict(fold=fold, model=name, precision=float(p.mean()), recall=float(r.mean()), f1=float(f.mean()), accuracy=accuracy_score(y, pred))
    return rows, overall, cm


def selection_statistics(X, y, waves):
    if not np.isfinite(REDUNDANCY_WEIGHT) or REDUNDANCY_WEIGHT < 0:
        raise ValueError("REDUNDANCY_WEIGHT must be finite and nonnegative")
    classes, counts = np.unique(y, return_counts=True)
    grand_mean = X.mean(axis=0)
    between = np.zeros(X.shape[1], dtype=float)
    within = np.zeros(X.shape[1], dtype=float)
    for label, count in zip(classes, counts):
        subset = X[y == label]
        mean = subset.mean(axis=0)
        between += count * (mean - grand_mean) ** 2
        within += ((subset - mean) ** 2).sum(axis=0)
    fisher = between / np.maximum(within, EPSILON)
    centered = X - grand_mean
    norms = np.linalg.norm(centered, axis=0)
    standardized = centered / np.maximum(norms, EPSILON)
    correlation = np.abs(standardized.T @ standardized)
    correlation = np.clip(correlation, 0.0, 1.0)
    options = domain_options(waves)
    indices = [[np.flatnonzero((waves >= d["start_nm"]-1e-8) &
                                   (waves <= d["end_nm"]+1e-8)) for d in group]
               for group in options]
    relevance = [np.array([fisher[idx].mean() for idx in group]) for group in indices]
    normalized = []
    for group in relevance:
        span = float(group.max() - group.min())
        normalized.append((group - group.min()) / span if span > EPSILON else np.zeros_like(group))
    redundancy = {}
    for a, b in itertools.combinations(range(len(options)), 2):
        redundancy[(a, b)] = np.array([
            [correlation[np.ix_(left, right)].mean() for right in indices[b]]
            for left in indices[a]
        ])
    return options, relevance, normalized, redundancy


def main():
    if INFERENCE_MODEL_PATH is not None:
        inference()
        return
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    waves, ids, X = load_data(DATA_CSV)
    paths = sorted(ALLOCATIONS_DIR.glob(ALLOCATION_PATTERN))
    if len(paths) != 5:
        raise ValueError(f"Expected five allocation CSVs; found {len(paths)}")
    allocations = [load_allocation(path, ids) for path in paths]
    test_counts = np.zeros(len(ids), dtype=int)
    for _, split in allocations:
        test_counts[split['test']] += 1
    if not np.all(test_counts == 1):
        raise ValueError("Expected each sample to be tested exactly once across allocations")
    class_rows, overall_rows, predictions, selections = [], [], [], []
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    for number, (path, (y, split)) in enumerate(zip(paths, allocations), 1):
        folder = OUTPUT_DIR / f"{OUTPUT_PREFIX}_{path.stem}"
        folder.mkdir(exist_ok=True)
        tr, va, te = (split[k] for k in ("train", "validation", "test"))
        labels = np.unique(y)
        options, relevance, normalized, redundancy = selection_statistics(X[tr], y[tr], waves)
        best_score, best_spec, best_choice, best_train = -np.inf, None, None, None
        total = candidate_count(waves)
        if total > MAX_GRID_CANDIDATES:
            raise ValueError(f"Grid has {total:,} combinations; change the search or MAX_GRID_CANDIDATES")
        print(f"{path.stem}: classifier-agnostic feature grid with {total:,} combinations", flush=True)
        ranked = []
        with (folder / "feature_grid.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["candidate", "specification", "mean_normalized_fisher", "mean_inter_interval_abs_correlation", "selection_score"])
            for index, choice in enumerate(itertools.product(
                    *(range(len(group)) for group in options)), 1):
                spec = [group[index_in_group] for group, index_in_group in zip(options, choice)]
                signal = float(np.mean([group[index_in_group]
                                        for group, index_in_group in zip(normalized, choice)]))
                dependence = float(np.mean([matrix[choice[a], choice[b]]
                                             for (a, b), matrix in redundancy.items()]))
                score = signal - REDUNDANCY_WEIGHT * dependence
                writer.writerow([index, json.dumps(spec), signal, dependence, score])
                ranked.append((score, index, spec, signal, dependence))
                if score > best_score:
                    best_score, best_spec, best_choice = score, spec, choice
        if best_spec is None:
            raise RuntimeError("No valid feature candidate; inspect feature_grid.csv")
        best_train = transform(X[tr], waves, best_spec)
        pd.DataFrame([{"rank": rank, "candidate": index, "selection_score": score,
                       "mean_normalized_fisher": signal,
                       "mean_inter_interval_abs_correlation": dependence,
                       "specification": json.dumps(spec)}
                      for rank, (score, index, spec, signal, dependence) in enumerate(
                          sorted(ranked, key=lambda item: (-item[0], item[1]))[:20], 1)
                      ]).to_csv(folder / "top_20_training_candidates.csv", index=False)
        validation = transform(X[va], waves, best_spec)
        selected_mask = np.zeros(len(waves), dtype=bool)
        for domain in best_spec:
            selected_mask |= (waves >= domain["start_nm"]-1e-8) & (waves <= domain["end_nm"]+1e-8)
        selected_wavelengths = waves[selected_mask].tolist()
        selected_relevance = float(np.mean([group[index_in_group]
                                            for group, index_in_group in zip(normalized, best_choice)]))
        selected_redundancy = float(np.mean([matrix[best_choice[a], best_choice[b]]
                                              for (a, b), matrix in redundancy.items()]))
        selected_band_rows = []
        for domain_number, (band, choice) in enumerate(zip(best_spec, best_choice)):
            included = waves[(waves >= band["start_nm"]-1e-8) & (waves <= band["end_nm"]+1e-8)]
            selected_band_rows.append({**band, "feature_count": len(included),
                                       "measured_wavelengths_nm": json.dumps(included.tolist()),
                                       "fisher_relevance": float(relevance[domain_number][choice]),
                                       "normalized_fisher_relevance": float(normalized[domain_number][choice])})
        selected = {"specification": best_spec, "feature_count": best_train.shape[1],
                    "measured_wavelengths_nm": selected_wavelengths,
                    "domain_search": DOMAIN_SEARCH,
                    "selection_method": "training-only Fisher relevance minus inter-interval absolute Pearson correlation",
                    "selection_score": best_score,
                    "mean_normalized_fisher": selected_relevance,
                    "mean_inter_interval_abs_correlation": selected_redundancy,
                    "fisher_normalization": "min-max within each search domain",
                    "redundancy_weight": REDUNDANCY_WEIGHT,
                    "training_sample_ids": ids[tr].tolist()}
        (folder / "selected_features.json").write_text(json.dumps(selected, indent=2))
        selections.append({"fold": path.stem, "domains": json.dumps(best_spec),
                           "measured_wavelengths_nm": json.dumps(selected_wavelengths),
                           "feature_count": best_train.shape[1], "selection_score": best_score,
                           **{f"{band['name']}_{key}": band[key]
                              for band in best_spec for key in ("start_nm", "end_nm")}})
        pd.DataFrame(selected_band_rows).to_csv(folder / "selected_bands.csv", index=False)
        pd.DataFrame({"wavelength_nm": selected_wavelengths}).to_csv(folder / "selected_wavelengths.csv", index=False)
        pd.DataFrame([{"domain": options[d][i]["name"],
                       "start_nm": options[d][i]["start_nm"],
                       "end_nm": options[d][i]["end_nm"],
                       "fisher_relevance": float(relevance[d][i]),
                       "normalized_fisher_relevance": float(normalized[d][i])}
                      for d in range(len(options)) for i in range(len(options[d]))
                      ]).to_csv(folder / "candidate_band_relevance.csv", index=False)
        for model_index, name in enumerate(MODELS):
            print(f"  {name}: TPE tuning", flush=True)
            def objective(trial):
                params = suggest(trial, name, best_train.shape[1], len(tr))
                trial.set_user_attr("resolved_parameters", params)
                m = model(name, params)
                m.fit(best_train, y[tr])
                return f1_score(y[va], m.predict(validation), labels=labels, average="macro", zero_division=0)
            study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED+number*100+model_index, n_startup_trials=min(10, max(1, N_TRIALS//3))))
            study.optimize(objective, n_trials=N_TRIALS, n_jobs=1, catch=(ValueError, np.linalg.LinAlgError))
            completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            if not completed:
                raise RuntimeError(f"All tuning trials failed: {name}")
            params = study.best_trial.user_attrs['resolved_parameters']
            fitted = model(name, params)
            fit_ids = np.concatenate([tr, va]) if REFIT_TRAIN_VALIDATION else tr
            fitted.fit(transform(X[fit_ids], waves, best_spec), y[fit_ids])
            artifact = {"feature_type": "domain", "model": name, "fold": path.stem,
                        "source_wavelengths_nm": waves.tolist(), "specification": best_spec,
                        "measured_wavelengths_nm": selected_wavelengths,
                        "fitted_model": pack_fitted_model(fitted)}
            model_path = folder / f"{name}_model.joblib"
            joblib.dump(artifact, model_path, compress=3)
            reloaded = joblib.load(model_path)
            pred = predict_fitted(reloaded["fitted_model"], transform(X[te], waves, reloaded["specification"]))
            rows, overall, cm = evaluate(y[te], pred, labels, path.stem, name)
            class_rows.extend(rows); overall_rows.append(overall)
            predictions.extend(dict(fold=path.stem, model=name, sample_id=ids[i], actual=y[i], predicted=p) for i, p in zip(te, pred))
            settings = dict(model=name, fold=path.stem, optimized_hyperparameters=params,
                            saved_model=str(model_path),
                            effective_pipeline_parameters=fitted.get_params(deep=True),
                            best_validation_macro_f1=study.best_value, features=selected,
                            refit_train_validation=REFIT_TRAIN_VALIDATION,
                            sample_ids={role:ids[idx].tolist() for role,idx in split.items()},
                            versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, sklearn=sklearn.__version__, optuna=optuna.__version__),
                            data_sha256=hashlib.sha256(DATA_CSV.read_bytes()).hexdigest(),
                            allocation_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            (folder / f"{name}_optimized.json").write_text(json.dumps(settings, indent=2, default=str))
            study.trials_dataframe().to_csv(folder / f"{name}_trials.csv", index=False)
            pd.DataFrame(cm, index=labels, columns=labels).to_csv(folder / f"{name}_test_confusion_counts.csv")
            pd.DataFrame(cm/cm.sum(axis=1, keepdims=True), index=labels, columns=labels).to_csv(folder / f"{name}_test_confusion_normalized.csv")
            pd.DataFrame(class_rows).to_csv(OUTPUT_DIR / f"{OUTPUT_PREFIX}_per_class_by_fold.csv", index=False)
            pd.DataFrame(overall_rows).to_csv(OUTPUT_DIR / f"{OUTPUT_PREFIX}_overall_by_fold.csv", index=False)
            pd.DataFrame(predictions).to_csv(OUTPUT_DIR / f"{OUTPUT_PREFIX}_test_predictions.csv", index=False)
    summarize(pd.DataFrame(class_rows), ['model', 'stage']).to_csv(OUTPUT_DIR / f"{OUTPUT_PREFIX}_per_class_mean_std.csv", index=False)
    summarize(pd.DataFrame(overall_rows), ['model']).to_csv(OUTPUT_DIR / f"{OUTPUT_PREFIX}_overall_mean_std.csv", index=False)
    pd.DataFrame(selections).to_csv(OUTPUT_DIR / f"{OUTPUT_PREFIX}_selected_features_by_fold.csv", index=False)
    selection_frame = pd.DataFrame([
        {"allocation": row["fold"], "domain": band["name"],
         "start_nm": band["start_nm"], "end_nm": band["end_nm"],
         "selection_score": row["selection_score"]}
        for row in selections for band in json.loads(row["domains"])
    ])
    selection_frame.to_csv(OUTPUT_DIR / f"{OUTPUT_PREFIX}_selected_bands_by_allocation.csv", index=False)
    selection_frame.groupby(["domain", "start_nm", "end_nm"]).size().rename("times_selected").reset_index().to_csv(
        OUTPUT_DIR / f"{OUTPUT_PREFIX}_selected_band_frequency.csv", index=False)
    wavelength_frame = pd.DataFrame([{"allocation": row["fold"], "wavelength_nm": wavelength}
                                     for row in selections for wavelength in json.loads(row["measured_wavelengths_nm"])])
    wavelength_frame.groupby("wavelength_nm").size().rename("times_selected").reset_index().to_csv(
        OUTPUT_DIR / f"{OUTPUT_PREFIX}_selected_wavelength_frequency.csv", index=False)
    print(f"Finished: {OUTPUT_DIR.resolve()}")



def domain_options(waves):
    if len(DOMAIN_SEARCH) < 2:
        raise ValueError("DOMAIN_SEARCH must contain at least two domains")
    if len({d['name'] for d in DOMAIN_SEARCH}) != len(DOMAIN_SEARCH):
        raise ValueError("DOMAIN_SEARCH names must be unique")
    result = []
    for d in DOMAIN_SEARCH:
        lo, hi, width, step = (float(d[k]) for k in ('min_nm', 'max_nm', 'width_nm', 'step_nm'))
        if step <= 0 or width <= 0 or hi-lo < width:
            raise ValueError(f"Invalid domain configuration: {d}")
        options, seen = [], set()
        for start in np.arange(lo, hi-width+1e-8, step):
            end = start+width
            idx = tuple(np.flatnonzero((waves >= start-1e-8) & (waves <= end+1e-8)))
            if idx and idx not in seen:
                options.append(dict(name=d['name'], start_nm=float(start), end_nm=float(end)))
                seen.add(idx)
        if not options:
            raise ValueError(f"Domain contains no measured wavelengths: {d}")
        result.append(options)
    return result


def candidate_count(waves):
    return int(np.prod([len(v) for v in domain_options(waves)]))


def candidates(waves):
    yield from (list(v) for v in itertools.product(*domain_options(waves)))


def transform(X, waves, spec):
    mask = np.zeros(len(waves), dtype=bool)
    for d in spec:
        mask |= (waves >= d['start_nm']-1e-8) & (waves <= d['end_nm']+1e-8)
    if not mask.any():
        raise ValueError("Empty feature set")
    return X[:, mask]


if __name__ == "__main__":
    main()

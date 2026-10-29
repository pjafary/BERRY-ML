import hashlib
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
OUTPUT_DIR = SCRIPT_DIR / "Results" / "pca_mov_ave_v1"
OUTPUT_PREFIX = "pca_mov_ave_v1"

MODELS = ["PLS_DA", "Shrinkage_LDA", "PCA_LDA", "Linear_SVM", "RBF_SVM", "Logistic_Regression", "Random_Forest"]

SEED = 42
N_TRIALS = 60
REFIT_TRAIN_VALIDATION = True
INFERENCE_MODEL_PATH = None
INFERENCE_CSV = OUTPUT_DIR / "new_spectra.csv"
INFERENCE_OUTPUT_CSV = OUTPUT_DIR / "pca_band_predictions.csv"

N_SELECTION_PCS = 6
N_WAVELENGTHS_PER_PC = 6
N_CLASS_PCS = 6
N_TOP_LOADINGS_PER_PC = 6

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
    if artifact["feature_type"] != "pca_bands":
        raise ValueError("This script only loads PCA-band model artifacts")
    waves, ids, X = load_data(INFERENCE_CSV)
    expected = np.asarray(artifact["source_wavelengths_nm"], dtype=float)
    if len(waves) != len(expected) or not np.array_equal(np.sort(waves), np.sort(expected)):
        raise ValueError("Inference wavelength grid differs from the training data")
    row = {float(w): i for i, w in enumerate(waves)}
    X = X[:, [row[float(w)] for w in expected]]
    indices = selected_indices(expected, artifact["selected_wavelengths_nm"])
    pred = predict_fitted(artifact["fitted_model"], X[:, indices])
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


def selected_indices(wavelengths, selected_wavelengths):
    indices = []
    for value in selected_wavelengths:
        hits = np.flatnonzero(np.isclose(wavelengths, value, atol=1e-8, rtol=0))
        if len(hits) != 1:
            raise ValueError(f"Expected exactly one spectral channel at {value} nm")
        indices.append(int(hits[0]))
    return np.asarray(indices, dtype=int)


def save_pca_tables(pca, spectra, ids, wavelengths, folder, prefix):
    joblib.dump(pca, folder / f"{prefix}_pca.joblib", compress=3)
    pd.DataFrame({"wavelength_nm": wavelengths, "training_mean": pca.mean_}).to_csv(
        folder / f"{prefix}_mean_spectrum.csv", index=False)
    columns = [f"PC{k+1}" for k in range(pca.n_components_)]
    loadings = pd.DataFrame(pca.components_.T, columns=columns)
    loadings.insert(0, "wavelength_nm", wavelengths)
    loadings.to_csv(folder / f"{prefix}_loadings.csv", index=False)
    scores = pd.DataFrame(pca.transform(spectra), columns=columns)
    scores.insert(0, "sample_id", ids)
    scores.to_csv(folder / f"{prefix}_scores.csv", index=False)
    variance = pd.DataFrame({"PC": columns,
                             "explained_variance_ratio": pca.explained_variance_ratio_,
                             "cumulative_variance_ratio": np.cumsum(pca.explained_variance_ratio_),
                             "eigenvalue": pca.explained_variance_})
    variance.to_csv(folder / f"{prefix}_variance.csv", index=False)
    top = []
    for k, component in enumerate(pca.components_):
        order = np.argsort(-np.abs(component), kind="stable")[:N_TOP_LOADINGS_PER_PC]
        for rank, index in enumerate(order, 1):
            top.append({"PC": columns[k], "rank": rank,
                        "wavelength_nm": float(wavelengths[index]),
                        "loading": float(component[index]),
                        "absolute_loading": float(abs(component[index]))})
    pd.DataFrame(top).to_csv(folder / f"{prefix}_top_loadings.csv", index=False)
    return top


def choose_bands(components, wavelengths):
    if N_WAVELENGTHS_PER_PC < 1 or N_WAVELENGTHS_PER_PC > len(wavelengths):
        raise ValueError("Invalid N_WAVELENGTHS_PER_PC")
    selected = set()
    bands = []
    for k, component in enumerate(components, 1):
        peak = int(np.argmax(np.abs(component)))
        nearest = np.argsort(abs(wavelengths-wavelengths[peak]), kind="stable")[:N_WAVELENGTHS_PER_PC]
        nearest = sorted(int(i) for i in nearest)
        selected.update(nearest)
        bands.append({"PC": f"PC{k}", "peak_nm": float(wavelengths[peak]),
                      "peak_loading": float(component[peak]),
                      "start_nm": float(min(wavelengths[nearest])),
                      "end_nm": float(max(wavelengths[nearest])),
                      "measured_wavelengths_nm": wavelengths[nearest].tolist()})
    return np.asarray(sorted(selected), dtype=int), bands


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
    class_rows, overall_rows, predictions, selections, appendix_top = [], [], [], [], []
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    for number, (path, (y, split)) in enumerate(zip(paths, allocations), 1):
        folder = OUTPUT_DIR / f"{OUTPUT_PREFIX}_{path.stem}"
        folder.mkdir(exist_ok=True)
        tr, va, te = (split[k] for k in ("train", "validation", "test"))
        labels = np.unique(y)
        if N_SELECTION_PCS > min(len(tr), X.shape[1]) or N_CLASS_PCS > min(12, X.shape[1]):
            raise ValueError("Requested PCA components exceed available training samples or wavelengths")
        pooled_pca = PCA(n_components=N_SELECTION_PCS, svd_solver="full").fit(X[tr])
        save_pca_tables(pooled_pca, X[tr], ids[tr], waves, folder, "pooled_training")
        chosen, bands = choose_bands(pooled_pca.components_, waves)
        selected_wavelengths = waves[chosen].tolist()
        best_train = X[tr][:, chosen]
        validation = X[va][:, chosen]
        class_summary = {}
        for stage in labels:
            within = tr[y[tr] == stage]
            if len(within) <= N_CLASS_PCS:
                raise ValueError(f"Not enough training spectra for class {stage}")
            class_pca = PCA(n_components=N_CLASS_PCS, svd_solver="full").fit(X[within])
            top = save_pca_tables(class_pca, X[within], ids[within], waves, folder,
                                  f"class_{stage}_training")
            appendix_top.extend({"fold": path.stem, "stage": stage, **record} for record in top)
            class_summary[stage] = {"training_sample_ids": ids[within].tolist(),
                                    "explained_variance_ratio": class_pca.explained_variance_ratio_.tolist(),
                                    "cumulative_variance_ratio": np.cumsum(class_pca.explained_variance_ratio_).tolist()}
        (folder / "class_pca_summary.json").write_text(json.dumps(class_summary, indent=2))
        selected = {"feature_type": "pca_bands", "feature_count": len(chosen),
                    "measured_wavelengths_nm": selected_wavelengths,
                    "band_selection": "peak absolute pooled loading for each PC, nearest measured wavelengths",
                    "n_selection_pcs": N_SELECTION_PCS,
                    "n_wavelengths_per_pc": N_WAVELENGTHS_PER_PC,
                    "training_sample_ids": ids[tr].tolist(),
                    "pooled_variance_ratio": pooled_pca.explained_variance_ratio_.tolist(),
                    "pooled_cumulative_variance_ratio": np.cumsum(pooled_pca.explained_variance_ratio_).tolist(),
                    "bands_by_pc": bands}
        (folder / "selected_features.json").write_text(json.dumps(selected, indent=2))
        pd.DataFrame(bands).to_csv(folder / "pc_band_locations.csv", index=False)
        pd.DataFrame({"wavelength_nm": selected_wavelengths}).to_csv(folder / "selected_wavelengths.csv", index=False)
        selections.append({"fold": path.stem, "feature_type": "pca_bands",
                           "feature_count": len(chosen),
                           "pooled_six_pc_cumulative_variance": float(np.sum(pooled_pca.explained_variance_ratio_)),
                           "bands_by_pc": json.dumps(bands),
                           "measured_wavelengths_nm": json.dumps(selected_wavelengths)})
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
            fitted.fit(X[fit_ids][:, chosen], y[fit_ids])
            artifact = {"feature_type": "pca_bands", "model": name, "fold": path.stem,
                        "source_wavelengths_nm": waves.tolist(),
                        "selected_wavelengths_nm": selected_wavelengths,
                        "bands_by_pc": bands,
                        "fitted_model": pack_fitted_model(fitted)}
            model_path = folder / f"{name}_model.joblib"
            joblib.dump(artifact, model_path, compress=3)
            reloaded = joblib.load(model_path)
            pred = predict_fitted(reloaded["fitted_model"],
                                  X[te][:, selected_indices(waves, reloaded["selected_wavelengths_nm"])])
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
    pd.DataFrame(appendix_top).to_csv(OUTPUT_DIR / f"{OUTPUT_PREFIX}_class_top_loadings_by_fold.csv", index=False)
    print(f"Finished: {OUTPUT_DIR.resolve()}")



if __name__ == "__main__":
    main()

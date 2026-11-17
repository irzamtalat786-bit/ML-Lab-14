import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    f1_score, precision_score, recall_score, classification_report,
    confusion_matrix
)
from .config import (FEATURES, REPORT_DIR, SEED, TRAIN_CAP, WINDOW_SIZE, RETRAIN_DELAY, RETRAIN_EVERY, UNKNOWN_REAL, MIN_RETRAIN_SAMPLES, MAX_FEEDBACK_BUFFER)
from .models import classifiers, fit_anomaly_model, predict_system
from .drift import DDM


def sample(frame, n):
    """Class-balanced sample so frequent classes cannot consume the cap."""
    if frame.empty or n <= 0:
        return frame.iloc[:0].copy()
    groups = [g for _, g in frame.groupby("Label", sort=False)]
    quota = max(1, n // max(len(groups), 1))
    pieces = []
    for g in groups:
        pieces.append(g.sample(min(len(g), quota), random_state=SEED))
    result = pd.concat(pieces)
    if len(result) < min(n, len(frame)):
        remaining = frame.loc[~frame.index.isin(result.index)]
        if len(remaining):
            result = pd.concat([result, remaining.sample(
                min(n-len(result), len(remaining)), random_state=SEED)])
    return result.sample(frac=1, random_state=SEED).reset_index(drop=True)


def select_classifier(train, val):
    rows = []
    train_X = train[FEATURES].replace([np.inf, -np.inf], np.nan)
    val_X = val[FEATURES].replace([np.inf, -np.inf], np.nan)
    for name, model in classifiers().items():
        start = time.time()
        model.fit(train_X, train["Label"])
        train_time = time.time() - start
        start = time.time()
        pred = model.predict(val_X)
        predict_time = time.time() - start
        rows.append({
            "model": name,
            "macro_f1": f1_score(val["Label"], pred, average="macro", zero_division=0),
            "precision": precision_score(val["Label"], pred, average="macro", zero_division=0),
            "recall": recall_score(val["Label"], pred, average="macro", zero_division=0),
            "train_sec": train_time, "predict_sec": predict_time, "estimator": model
        })
    rows.sort(key=lambda x: (-x["macro_f1"], x["train_sec"]))
    table = pd.DataFrame([{k: round(v, 4) if isinstance(v, (float, np.floating)) else v
                           for k, v in r.items() if k != "estimator"} for r in rows])
    return rows[0]["model"], rows[0]["estimator"], table


def anomaly_threshold(iso, benign_validation):
    if len(benign_validation) == 0:
        raise ValueError("Validation data has no benign records for threshold calibration.")
    return float(np.percentile(-iso.score_samples(benign_validation[FEATURES]), 95))


def stream_run(stream, base_train, initial_clf, initial_iso, threshold,
               known, best_name, strategy, use_if, use_ddm):
    clf, iso, thr = initial_clf, (initial_iso if use_if else None), threshold
    detector = DDM()
    preds, accuracies, alarms, retrain_log = [], [], [], []
    pending = None
    n_windows = (len(stream) + WINDOW_SIZE - 1) // WINDOW_SIZE

    for w in range(n_windows):
        start = w * WINDOW_SIZE
        chunk = stream.iloc[start:start + WINDOW_SIZE]
        if chunk.empty:
            continue
        X, y = chunk[FEATURES], chunk["Label"].to_numpy()
        X_safe = X.replace([np.inf, -np.inf], np.nan)
        classifier_pred = clf.predict(X_safe)
        system_pred = predict_system(clf, iso, thr, X)
        preds.extend(system_pred.tolist())

        known_mask = np.isin(y, known)
        accuracies.append(float(np.mean(classifier_pred[known_mask] == y[known_mask]))
                          if known_mask.any() else np.nan)

        alarm = False
        if use_ddm and known_mask.any():
            for err in (classifier_pred[known_mask] != y[known_mask]).astype(int):
                if detector.update(err):
                    alarm = True
                    alarms.append(w)
                    detector.reset()
                    break
        if alarm and strategy == "drift" and pending is None:
            pending = w + RETRAIN_DELAY

        periodic_due = (w + 1) % RETRAIN_EVERY == 0
        drift_due = pending is not None and pending <= w
        due = ((strategy == "periodic" and periodic_due) or
               (strategy == "drift" and (drift_due or periodic_due)))
        if not due:
            continue
        if drift_due:
            pending = None

        recent = stream.iloc[max(0, start - 2*WINDOW_SIZE):start + len(chunk)].copy()
        recent = recent[recent["Label"].isin(known)].tail(MAX_FEEDBACK_BUFFER)
        if len(recent) < MIN_RETRAIN_SAMPLES or recent["Label"].nunique() < 2:
            retrain_log.append({
                "window": w, "attempted": False, "accepted": False,
                "reason": f"insufficient recent data: {len(recent)} rows / "
                          f"{recent['Label'].nunique()} classes"
            })
            continue

        # Temporal holdout: train on earlier feedback, validate on the latest.
        cut = int(0.70 * len(recent))
        recent_train, recent_val = recent.iloc[:cut], recent.iloc[cut:]
        if recent_val.empty or recent_train["Label"].nunique() < 2:
            retrain_log.append({
                "window": w, "attempted": False, "accepted": False,
                "reason": "insufficient class diversity for validation"
            })
            continue

        old_sample = sample(base_train, min(500, len(base_train)))
        fit_data = pd.concat([old_sample, recent_train], ignore_index=True)
        candidate = classifiers()[best_name]
        fit_X = fit_data[FEATURES].replace([np.inf, -np.inf], np.nan)
        candidate.fit(fit_X, fit_data["Label"])

        candidate_iso, candidate_thr = None, thr
        if use_if:
            benign_fit = fit_data[fit_data["Label"] == "Benign"]
            if len(benign_fit) >= 20:
                candidate_iso = fit_anomaly_model(
                    benign_fit[FEATURES].replace([np.inf, -np.inf], np.nan)
                )
                benign_holdout = recent_val[recent_val["Label"] == "Benign"]
                if len(benign_holdout):
                    candidate_thr = anomaly_threshold(candidate_iso, benign_holdout[FEATURES])

        old_p = predict_system(clf, iso, thr, recent_val[FEATURES])
        new_p = predict_system(candidate, candidate_iso, candidate_thr, recent_val[FEATURES])
        old_f1 = f1_score(recent_val["Label"], old_p, labels=known,
                          average="macro", zero_division=0)
        new_f1 = f1_score(recent_val["Label"], new_p, labels=known,
                          average="macro", zero_division=0)
        accepted = new_f1 >= old_f1
        retrain_log.append({
            "window": w, "attempted": True,
            "old_f1": round(old_f1, 4), "new_f1": round(new_f1, 4),
            "accepted": bool(accepted),
            "reason": "candidate accepted" if accepted else "candidate rejected: validation F1 decreased"
        })
        if accepted:
            clf, iso, thr = candidate, (candidate_iso if use_if else None), candidate_thr

    return np.asarray(preds, dtype=object), accuracies, alarms, retrain_log


def metrics(y, pred, known, unknown):
    known_mask, unknown_mask, benign_mask = (
        np.isin(y, known), y == unknown, y == "Benign")
    return {
        "known_macro_f1": round(f1_score(y[known_mask], pred[known_mask],
                                        labels=known, average="macro", zero_division=0), 3),
        "unknown_recall": round(float(np.mean(pred[unknown_mask] == "Suspicious/Unknown")), 3)
                          if unknown_mask.any() else np.nan,
        "benign_fpr": round(float(np.mean(pred[benign_mask] != "Benign")), 3)
                      if benign_mask.any() else np.nan
    }


def run_experiment(df, drift_at, synthetic_mode):
    unknown = "Infiltration" if synthetic_mode else UNKNOWN_REAL
    if unknown not in set(df["Label"]):
        raise ValueError(
            f"Held-out class {unknown!r} not found. Check labels or change UNKNOWN_REAL in config.py."
        )

    
    # Chronological development and streaming split
    n = len(df)
    val_end = int(0.60 * n)

    development = df.iloc[:val_end].copy()
    stream = df.iloc[val_end:].reset_index(drop=True)

    # Exclude unknown attacks from training and validation
    known = sorted(set(df["Label"].unique()) - {unknown})

    development = development[
        development["Label"].isin(known)
    ].copy()

    # Stratified training and validation split
    from sklearn.model_selection import train_test_split

    train, val = train_test_split(
        development,
        test_size=0.20,
        random_state=SEED,
        stratify=development["Label"]
    )

    print("\n=== DATA SPLIT SUMMARY ===")
    print("Training:", len(train))
    print("Validation:", len(val))
    print("Streaming:", len(stream))

    print("\nTraining distribution:")
    print(train["Label"].value_counts())

    print("\nValidation distribution:")
    print(val["Label"].value_counts())


    best_name, best_clf, comparison = select_classifier(sample(train, TRAIN_CAP), val)
    val_pred = best_clf.predict(
        val[FEATURES].replace([np.inf, -np.inf], np.nan)
    )
    print("\n=== CLASSIFIER COMPARISON ===\n", comparison.to_string(index=False))
    print("\nSelected:", best_name)
    print("\n=== VALIDATION REPORT ===\n",
          classification_report(val["Label"], val_pred, zero_division=0))

    benign_train = train[train["Label"] == "Benign"]
    iso = fit_anomaly_model(sample(benign_train, TRAIN_CAP)[FEATURES])
    threshold = anomaly_threshold(iso, val[val["Label"] == "Benign"])

    configs = [
        ("A", "Classifier only", "static", False, False),
        ("B", "Classifier + Isolation Forest", "static", True, False),
        ("C", "Classifier + DDM", "static", False, True),
        ("D", "Classifier + IF + DDM", "static", True, True),
        ("E", "Full adaptive system", "drift", True, True),
        ("P", "Periodic retraining baseline", "periodic", True, False),
    ]
    rows, curves, alarm_sets, logs, full_pred = [], {}, {}, {}, None
    for cid, title, strategy, use_if, use_ddm in configs:
        pred, curve, alarms, log = stream_run(
            stream, train, best_clf, iso, threshold, known,
            best_name, strategy, use_if, use_ddm
        )
        y = stream["Label"].to_numpy()[:len(pred)]
        rows.append({
            "ID": cid, "Configuration": title, **metrics(y, pred, known, unknown),
            "Drift alarms": len(alarms),
            "Retraining attempts": sum(x.get("attempted", False) for x in log),
            "Accepted retrains": sum(x.get("accepted", False) for x in log)
        })
        curves[cid], alarm_sets[cid], logs[cid] = curve, alarms, log
        if cid == "E":
            full_pred = pred

    ablation = pd.DataFrame(rows)
    print("\n=== ABLATION / STREAM RESULTS ===\n", ablation.to_string(index=False))
    print("\nFull-system drift alarms:", alarm_sets["E"])
    print("Full-system retraining attempts:",
          sum(x.get("attempted", False) for x in logs["E"]))
    print("Full-system accepted retrains:",
          sum(x.get("accepted", False) for x in logs["E"]))
    print("Full-system retraining log:", logs["E"])
    valid_accuracy = [x for x in curves["E"] if np.isfinite(x)]
    if valid_accuracy:
        print("Mean known-class classifier accuracy:",
              f"{np.mean(valid_accuracy):.4f}")

    true_window = None
    if synthetic_mode and drift_at is not None:
        true_window = max(0, (drift_at - val_end) // WINDOW_SIZE)
        detected = next((x for x in alarm_sets["E"] if x >= true_window), None)
        print("Synthetic drift window:", true_window,
              "| detection delay:", None if detected is None else detected - true_window)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(2, 2, figsize=(15, 11))
    cm = confusion_matrix(val["Label"], val_pred, labels=known, normalize="true")
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=known, yticklabels=known, ax=ax[0, 0])
    ax[0, 0].set(title="Validation confusion matrix", xlabel="Predicted", ylabel="Actual")

    melted = comparison.melt(id_vars="model",
                             value_vars=["macro_f1", "precision", "recall"],
                             var_name="metric", value_name="score")
    sns.barplot(data=melted, x="model", y="score", hue="metric", ax=ax[0, 1])
    ax[0, 1].set(title="Five classifier comparison", ylim=(0, 1))
    ax[0, 1].tick_params(axis="x", rotation=20)

    for cid, label in [("A", "Static"), ("P", "Periodic"), ("E", "Adaptive")]:
        ax[1, 0].plot(curves[cid], label=label)
    if true_window is not None:
        ax[1, 0].axvline(true_window, linestyle="--", label="Synthetic drift")
    ax[1, 0].set(title="Known-class accuracy by stream window",
                 xlabel="Window", ylabel="Accuracy")
    ax[1, 0].legend()

    y = stream["Label"].to_numpy()[:len(full_pred)]
    table = pd.crosstab(pd.Series(y, name="Actual"),
                        pd.Series(full_pred, name="Predicted"), normalize="index")
    sns.heatmap(table, annot=True, fmt=".2f", cmap="Oranges", ax=ax[1, 1])
    ax[1, 1].set_title("Full system stream results")
    plt.tight_layout()
    output = REPORT_DIR / "results.png"
    plt.savefig(output, dpi=140)
    print("\nSaved plot:", output)
    plt.show()
    return comparison, ablation

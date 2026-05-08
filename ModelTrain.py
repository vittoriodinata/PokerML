
import argparse
import json
import warnings
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    learning_curve,
    train_test_split,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

# CLI
parser = argparse.ArgumentParser()
parser.add_argument("--data",      default="poker_dataset.csv")
parser.add_argument("--out",       default="poker_model")
parser.add_argument("--trees",     type=int,  default=300)
parser.add_argument("--depth",     type=int,  default=None)
parser.add_argument("--seed",      type=int,  default=42)
parser.add_argument("--cv",        type=int,  default=0,
                    help="Stratified k-fold CV folds (0 = skip CV)")
parser.add_argument("--tune",      action="store_true",
                    help="Run RandomizedSearchCV before final fit")
parser.add_argument("--n-iter",    type=int,  default=30,
                    help="Iterations for RandomizedSearchCV (--tune only)")
parser.add_argument("--calibrate", action="store_true",
                    help="Isotonic calibration of predicted probabilities")
parser.add_argument("--curves",    action="store_true",
                    help="Compute learning curves (adds ~1 min)")
args = parser.parse_args()
MODEL_DIR = Path(args.out)
MODEL_DIR.mkdir(exist_ok=True)

# LOAD
df = pd.read_csv(args.data)
print(f"Loaded {len(df):,} rows × {df.shape[1]} columns")
print(f"\nAction distribution:\n{df['action'].value_counts().to_string()}\n")
CAT_COLS = ["stage", "position"]
label_encoders: dict = {}
for col in CAT_COLS:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
action_le = LabelEncoder()
df["action_id"] = action_le.fit_transform(df["action"])
label_encoders["action"] = action_le
CLASS_NAMES = list(action_le.classes_)
FEATURE_COLS = [c for c in df.columns if c not in ("action", "action_id")]
X = df[FEATURE_COLS].values.astype(np.float32)
y = df["action_id"].values.astype(np.int64)
print(f"Classes  : {CLASS_NAMES}")
print(f"Features : {len(FEATURE_COLS)}")

# SPLIT & SCALING
X_tv, X_test, y_tv, y_test = train_test_split(
    X, y, test_size=0.15, random_state=args.seed, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_tv, y_tv,
    test_size=0.15 / 0.85,
    random_state=args.seed,
    stratify=y_tv,
)
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)

# TUNING
base_params = dict(
    n_estimators      = args.trees,
    max_depth         = args.depth,
    min_samples_split = 4,
    min_samples_leaf  = 2,
    max_features      = "sqrt",
    class_weight      = "balanced",
    n_jobs            = -1,
    random_state      = args.seed,
    oob_score         = True,
)
if args.tune:
    param_dist = {
        "n_estimators":      [100, 200, 300, 400, 500],
        "max_depth":         [None, 10, 15, 20, 25, 30],
        "min_samples_split": [2, 4, 6, 8],
        "min_samples_leaf":  [1, 2, 4],
        "max_features":      ["sqrt", "log2", 0.3, 0.5],
        "class_weight":      ["balanced", "balanced_subsample"],
    }

    search_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=args.seed)
    search = RandomizedSearchCV(
        RandomForestClassifier(n_jobs=-1, random_state=args.seed, oob_score=False),
        param_distributions=param_dist,
        n_iter=args.n_iter,
        cv=search_cv,
        scoring="accuracy",
        n_jobs=-1,
        random_state=args.seed,
        verbose=1,
        refit=True,
    )
    search.fit(X_train, y_train)

    print(f"\nBest CV accuracy : {search.best_score_:.4f}")
    print(f"Best params      : {search.best_params_}")
    best_params = search.best_params_.copy()
    best_params.update({"n_jobs": -1, "random_state": args.seed, "oob_score": True})
    base_params.update(best_params)

# CROSS VALIDATION
cv_acc = cv_f1 = None

if args.cv > 1:
    print(f"\nRunning {args.cv}-fold stratified cross-validation ...")
    cv_params = {k: v for k, v in base_params.items() if k != "oob_score"}
    cv_rf     = RandomForestClassifier(**cv_params, oob_score=False)
    cv_skf    = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=args.seed)
    cv_acc = cross_val_score(cv_rf, X_train, y_train, cv=cv_skf, scoring="accuracy", n_jobs=-1)
    cv_f1  = cross_val_score(cv_rf, X_train, y_train, cv=cv_skf, scoring="f1_macro",  n_jobs=-1)

    print(f"  CV accuracy : {cv_acc.mean():.4f} ± {cv_acc.std():.4f}")
    print(f"  CV F1 macro : {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")

# TRAIN

print(f"\nTraining final Random Forest ...")
rf = RandomForestClassifier(**base_params)
rf.fit(X_train, y_train)
print(f"Done.  OOB accuracy: {rf.oob_score_:.4f}")

# LEARNING CURVES
if args.curves:
    print("\nComputing learning curves (this may take a minute) ...")
    lc_params = {k: v for k, v in base_params.items() if k != "oob_score"}
    lc_rf     = RandomForestClassifier(**lc_params, oob_score=False)
    lc_skf    = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)

    train_sizes, train_scores, val_scores = learning_curve(
        lc_rf, X_train, y_train,
        cv=lc_skf,
        scoring="accuracy",
        train_sizes=np.linspace(0.1, 1.0, 8),
        n_jobs=-1,
    )
    print(f"  {'Train size':<12}  {'Train acc':<12}  {'Val acc'}")
    for sz, tr, vl in zip(train_sizes, train_scores.mean(axis=1), val_scores.mean(axis=1)):
        print(f"  {sz:<12d}  {tr:<12.4f}  {vl:.4f}")

# PROBABILITY CALIBRATION
if args.calibrate:
    print("\nCalibrating probabilities (isotonic regression) ...")
    calibrated_rf = CalibratedClassifierCV(rf, method="isotonic", cv=3)
    calibrated_rf.fit(np.vstack([X_train, X_val]), np.concatenate([y_train, y_val]))
    raw_proba = rf.predict_proba(X_test)
    cal_proba = calibrated_rf.predict_proba(X_test)
    ll_raw    = log_loss(y_test, raw_proba)
    ll_cal    = log_loss(y_test, cal_proba)
    print(f"  Log-loss before calibration : {ll_raw:.4f}")
    print(f"  Log-loss after  calibration : {ll_cal:.4f}")
    model_for_inference = calibrated_rf
else:
    model_for_inference = rf

# EVALUATE
val_preds  = model_for_inference.predict(X_val)
val_acc    = (val_preds == y_val).mean()
test_preds = model_for_inference.predict(X_test)
test_proba = model_for_inference.predict_proba(X_test)
test_acc   = (test_preds == y_test).mean()
test_ll    = log_loss(y_test, test_proba)

try:
    test_auc = roc_auc_score(y_test, test_proba, multi_class="ovr", average="macro")
except Exception:
    test_auc = float("nan")

print(f"\n{'='*55}")
print(f"  Val  accuracy : {val_acc:.4f}")
print(f"  Test accuracy : {test_acc:.4f}")
print(f"  Test log-loss : {test_ll:.4f}")
print(f"  Test AUC (OvR): {test_auc:.4f}")
print(f"{'='*55}")
print(classification_report(y_test, test_preds, target_names=CLASS_NAMES, digits=4))

cm = confusion_matrix(y_test, test_preds)
print("Confusion matrix (rows=actual, cols=predicted):")
print(pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_string())

# Save
joblib.dump(model_for_inference, MODEL_DIR / "rf_model.pkl")
joblib.dump(scaler,              MODEL_DIR / "scaler.pkl")
joblib.dump(label_encoders,      MODEL_DIR / "label_encoders.pkl")

cv_results = {}
if cv_acc is not None and cv_f1 is not None:
    cv_results = {
        "cv_folds":    args.cv,
        "cv_acc_mean": round(float(cv_acc.mean()), 6),
        "cv_acc_std":  round(float(cv_acc.std()),  6),
        "cv_f1_mean":  round(float(cv_f1.mean()),  6),
        "cv_f1_std":   round(float(cv_f1.std()),   6),
    }

meta = {
    "feature_cols":  FEATURE_COLS,
    "cat_cols":      CAT_COLS,
    "class_names":   CLASS_NAMES,
    "n_estimators":  base_params["n_estimators"],
    "max_depth":     base_params["max_depth"],
    "calibrated":    args.calibrate,
    "oob_score":     round(rf.oob_score_, 6),
    "val_acc":       round(float(val_acc),   6),
    "test_acc":      round(float(test_acc),  6),
    "test_log_loss": round(float(test_ll),   6),
    "test_auc_ovr":  round(float(test_auc),  6) if not np.isnan(test_auc) else None,
    "top_features":  feat_imp.round(6).to_dict(),
}

with open(MODEL_DIR / "meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print(f"\nSaved to {MODEL_DIR}/")
print(f"  rf_model.pkl         {'Calibrated RF' if args.calibrate else 'Random Forest'}")
print(f"  scaler.pkl           StandardScaler")
print(f"  label_encoders.pkl   stage / position / action")
print(f"  meta.json            accuracy + feature importance")

# Inference
def load_model(model_dir: Path = MODEL_DIR):
    _meta = json.load(open(model_dir / "meta.json"))
    _les  = joblib.load(model_dir / "label_encoders.pkl")
    _sc   = joblib.load(model_dir / "scaler.pkl")
    _rf   = joblib.load(model_dir / "rf_model.pkl")

    def predict_action(hand: dict) -> tuple[str, dict]:
        row = pd.DataFrame([hand])
        for col in _meta["cat_cols"]:
            row[col] = _les[col].transform(row[col])
        X_in   = row[_meta["feature_cols"]].values.astype(float)
        X_sc   = _sc.transform(X_in)
        proba  = _rf.predict_proba(X_sc)[0]
        action = _les["action"].inverse_transform([int(proba.argmax())])[0]
        return action, dict(zip(_meta["class_names"], proba.round(4).tolist()))

    return predict_action

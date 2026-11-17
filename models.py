import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from .config import SEED, SUSPICIOUS


def classifiers():
    # Imputation is fitted on training data only, preventing validation leakage.
    return {
        "Random Forest": make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            RandomForestClassifier(n_estimators=150, n_jobs=-1,
                                   class_weight="balanced", random_state=SEED)
        ),
        "KNN": make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            StandardScaler(), KNeighborsClassifier(5)
        ),
        "Logistic Regression": make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            StandardScaler(),
            LogisticRegression(max_iter=1500, class_weight="balanced",
                               random_state=SEED)
        ),
        "Decision Tree": make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            DecisionTreeClassifier(max_depth=12, class_weight="balanced",
                                   random_state=SEED)
        ),
        "SVM": make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            StandardScaler(),
            SVC(class_weight="balanced", random_state=SEED)
        ),
    }


def fit_anomaly_model(X):
    if len(X) < 20:
        raise ValueError("At least 20 benign training records are needed for Isolation Forest.")
    model = make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        StandardScaler(),
        IsolationForest(n_estimators=200, contamination="auto",
                        random_state=SEED, n_jobs=-1)
    )
    return model.fit(X)


def predict_system(clf, iso, threshold, X):
    # Convert infinities to missing values; pipeline imputers handle them.
    X_safe = X.replace([np.inf, -np.inf], np.nan)
    pred = clf.predict(X_safe).astype(object)
    if iso is not None:
        anomaly = -iso.score_samples(X_safe) > threshold
        pred[(pred == "Benign") & anomaly] = SUSPICIOUS
    return pred

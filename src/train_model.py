import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)
import numpy as np
import joblib
import matplotlib.pyplot as plt

RANDOM_SEED = 42


def load_data():
    path = Path("data/processed/hospit_risk_dataset.csv")
    df = pd.read_csv(path)
    return df


def build_preprocessor():
    numeric_features = [
        "age",
        "temperature",
        "spo2",
        "freq_resp",
        "severity_score",
    ]
    categorical_features = [
        "sexe",
        "hta",
        "diabete",
        "cardiopathie",
        "infection_respi",
        "vaccinated",
    ]

    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])

    # 🔴 AVANT : categorical_transformer = "passthrough"
    # ✅ MAINTENANT : OneHotEncoder
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor, numeric_features + categorical_features


def train_and_evaluate():
    df = load_data()
    target = "hospitalisation"

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    preprocessor, feature_names = build_preprocessor()

    # ======== 1) Modèle de base : Régression logistique ========
    logreg_clf = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)),
        ]
    )

    logreg_clf.fit(X_train, y_train)

    y_proba_lr = logreg_clf.predict_proba(X_test)[:, 1]
    y_pred_lr = (y_proba_lr >= 0.5).astype(int)

    auc_lr = roc_auc_score(y_test, y_proba_lr)
    acc_lr = accuracy_score(y_test, y_pred_lr)
    recall_lr = recall_score(y_test, y_pred_lr)
    precision_lr = precision_score(y_test, y_pred_lr)
    f1_lr = f1_score(y_test, y_pred_lr)

    print("=== Régression Logistique ===")
    print(f"AUC       : {auc_lr:.3f}")
    print(f"Accuracy  : {acc_lr:.3f}")
    print(f"Recall    : {recall_lr:.3f}")
    print(f"Précision : {precision_lr:.3f}")
    print(f"F1-score  : {f1_lr:.3f}")
    print("\nClassification report :")
    print(classification_report(y_test, y_pred_lr))
    print("Matrice de confusion :")
    print(confusion_matrix(y_test, y_pred_lr))

    # ======== 2) Modèle avancé : Random Forest ========
    rf_clf = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=300,
                    random_state=RANDOM_SEED,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    rf_clf.fit(X_train, y_train)

    y_proba_rf = rf_clf.predict_proba(X_test)[:, 1]
    y_pred_rf = (y_proba_rf >= 0.5).astype(int)

    auc_rf = roc_auc_score(y_test, y_proba_rf)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    recall_rf = recall_score(y_test, y_pred_rf)
    precision_rf = precision_score(y_test, y_pred_rf)
    f1_rf = f1_score(y_test, y_pred_rf)

    print("\n=== Random Forest ===")
    print(f"AUC       : {auc_rf:.3f}")
    print(f"Accuracy  : {acc_rf:.3f}")
    print(f"Recall    : {recall_rf:.3f}")
    print(f"Précision : {precision_rf:.3f}")
    print(f"F1-score  : {f1_rf:.3f}")
    print("\nClassification report :")
    print(classification_report(y_test, y_pred_rf))
    print("Matrice de confusion :")
    print(confusion_matrix(y_test, y_pred_rf))

    # ======== 3) Sauvegarde des modèles ========
    Path("models").mkdir(exist_ok=True)
    joblib.dump(logreg_clf, "models/logreg_hospit.pkl")
    joblib.dump(rf_clf, "models/rf_hospit.pkl")
    print("\n✅ Modèles sauvegardés dans le dossier 'models/'.")


if __name__ == "__main__":
    train_and_evaluate()

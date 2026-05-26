"""""
Loan Default Risk Classifier
=============================
Trains a Random Forest model to predict loan defaults. Addresses class
imbalance with SMOTE oversampling on the training set, lowers the
decision threshold to reduce false negatives, and evaluates performance
with a classification report and annotated confusion matrix.

Model  : Random Forest Classifier
Author : Nathaniel Morrison
Date   : 05/20/2026
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


# ── Constants ─────────────────────────────────────────────────────────────────

DATA_PATH         = "data/loan_data.csv"
TEST_SIZE         = 0.20
RANDOM_STATE      = 42
DEFAULT_THRESHOLD = 0.30    # Lower than 0.5 to reduce costly false negatives

GRID_CLR  = "#E5E7EB"
FIG_SIZE  = (6, 5)


# ── Data preparation ──────────────────────────────────────────────────────────

def load_features(path: str) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load loan data and separate numeric features from the target label.

    Categorical columns and the target ('default') are excluded from X.

    Returns
    -------
    X : pd.DataFrame  — numeric feature matrix
    y : pd.Series     — binary default label (0 = no default, 1 = default)
    """
    df = pd.read_csv(path)

    X = df.select_dtypes(exclude=["object"]).drop(columns=["default"])
    y = df["default"]

    print("Feature matrix shape :", X.shape)
    print("Class distribution:\n", y.value_counts().to_string(), "\n")

    return X, y


def split_and_resample(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple:
    """
    Perform an 80/20 train-test split, then apply SMOTE to the training
    set to balance the minority (default) class.

    SMOTE is fitted only on training data to prevent data leakage.

    Returns
    -------
    X_train_res, X_test, y_train_res, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print("Before SMOTE:", y_train.value_counts().to_string())

    sm = SMOTE(random_state=random_state)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

    print("After SMOTE :", pd.Series(y_train_res).value_counts().to_string(), "\n")

    return X_train_res, X_test, y_train_res, y_test


# ── Model training & evaluation ───────────────────────────────────────────────

def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = RANDOM_STATE,
) -> RandomForestClassifier:
    """Train a Random Forest classifier on the resampled training data."""
    clf = RandomForestClassifier(random_state=random_state)
    clf.fit(X_train, y_train)
    return clf


def predict_with_threshold(
    clf: RandomForestClassifier,
    X_test: pd.DataFrame,
    threshold: float = DEFAULT_THRESHOLD,
) -> pd.Series:
    """
    Generate binary predictions using a custom probability threshold.

    A threshold below 0.5 increases recall for the default class,
    reducing costly false negatives at the expense of some precision.

    Parameters
    ----------
    threshold : float
        Minimum predicted probability for classifying a loan as defaulted.
    """
    proba    = clf.predict_proba(X_test)[:, 1]
    y_pred   = (proba >= threshold).astype(int)
    return pd.Series(y_pred, index=X_test.index)


def print_evaluation(y_test: pd.Series, y_pred: pd.Series) -> None:
    """Print a classification report to stdout."""
    print("=" * 60)
    print(f"CLASSIFICATION REPORT  (threshold = {DEFAULT_THRESHOLD})")
    print("=" * 60)
    print(classification_report(y_test, y_pred, target_names=["No Default", "Default"]))


def plot_confusion_matrix(y_test: pd.Series, y_pred: pd.Series) -> None:
    """Display an annotated confusion matrix heatmap."""
    cm     = confusion_matrix(y_test, y_pred)
    labels = ["No Default", "Default"]

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        linewidths=0.5,
        linecolor=GRID_CLR,
        ax=ax,
    )

    ax.set_title(
        f"Confusion Matrix  (threshold = {DEFAULT_THRESHOLD})",
        fontsize=12, fontweight="bold", pad=12,
    )
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    ax.tick_params(axis="both", length=0)

    fig.tight_layout()
    plt.show()


def plot_feature_importance(
    clf: RandomForestClassifier, feature_names: list[str], top_n: int = 10
) -> None:
    """Bar chart of the top-N features by mean Gini impurity decrease."""
    importances = (
        pd.Series(clf.feature_importances_, index=feature_names)
        .sort_values(ascending=True)
        .tail(top_n)
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    importances.plot(kind="barh", ax=ax, color="#2563EB", edgecolor="#1E3A8A",
                     linewidth=0.6)

    ax.set_title(f"Top {top_n} Feature Importances", fontsize=12,
                 fontweight="bold", pad=12)
    ax.set_xlabel("Mean Decrease in Gini Impurity", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.xaxis.grid(True, color=GRID_CLR, linewidth=0.8)
    ax.set_axisbelow(True)

    fig.tight_layout()
    plt.show()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    X, y = load_features(DATA_PATH)

    X_train, X_test, y_train, y_test = split_and_resample(X, y)

    clf    = train_model(X_train, y_train)
    y_pred = predict_with_threshold(clf, X_test, threshold=DEFAULT_THRESHOLD)

    print_evaluation(y_test, y_pred)
    plot_confusion_matrix(y_test, y_pred)
    plot_feature_importance(clf, feature_names=X.columns.tolist())


if __name__ == "__main__":
    main()

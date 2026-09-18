"""Auditable five-fold cross-validation for a binary GeneMLP classifier.

The script exposes the complete calculation trail requested for one reported
cross-validation cell:

* validation-fold class totals;
* the 2 x 2 confusion matrix for every fold;
* class-specific precision and fold macro precision;
* fold accuracy, macro recall, and macro F1;
* the five unrounded macro-precision values;
* the arithmetic mean and sample standard deviation (ddof=1); and
* an optional CSV containing the fold-level audit records.

Important
---------
Pass development data only. Do not include an external held-out test set.
This file contains a synthetic example for calculation verification; it cannot
reproduce a manuscript result unless it is run with the exact development
features, labels, representation, fixed configuration, and random seed used
for that result.
"""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import make_classification
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class Config:
    """Fixed cross-validation and GeneMLP settings."""

    n_splits: int = 5
    seed: int = 42
    hidden1: int = 512
    hidden2: int = 256
    dropout: float = 0.30
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 8
    epochs: int = 100


class GeneMLP(nn.Module):
    """Conventional two-hidden-layer fully connected classifier."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden1: int = 512,
        hidden2: int = 256,
        dropout: float = 0.30,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def set_seed(seed: int) -> None:
    """Set reproducibility controls for Python, NumPy, and PyTorch."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    torch.use_deterministic_algorithms(True, warn_only=True)


def _is_missing_label(value: object) -> bool:
    if value is None:
        return True

    try:
        return bool(np.isnan(value))
    except (TypeError, ValueError):
        return False


def validate_inputs(X: np.ndarray, y: np.ndarray, config: Config) -> None:
    """Validate data and settings before creating cross-validation folds."""

    if X.ndim != 2:
        raise ValueError("X must have shape (samples, features).")
    if y.ndim != 1:
        raise ValueError("y must be one-dimensional.")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must contain the same number of samples.")
    if X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("X cannot be empty.")
    if not np.isfinite(X).all():
        raise ValueError("X contains NaN or infinite values.")
    if any(_is_missing_label(value) for value in y):
        raise ValueError("y contains missing labels.")

    classes, counts = np.unique(y, return_counts=True)
    if len(classes) != 2:
        raise ValueError("This audit script requires exactly two classes.")
    if config.n_splits != 5:
        raise ValueError("This audit script is fixed to five-fold CV.")
    if counts.min() < config.n_splits:
        raise ValueError(
            "Each class must contain at least five development observations."
        )

    if config.hidden1 < 1 or config.hidden2 < 1:
        raise ValueError("Hidden-layer widths must be positive integers.")
    if not 0.0 <= config.dropout < 1.0:
        raise ValueError("dropout must be in the interval [0, 1).")
    if config.learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive.")
    if config.weight_decay < 0.0:
        raise ValueError("weight_decay cannot be negative.")
    if config.batch_size < 1 or config.epochs < 1:
        raise ValueError("batch_size and epochs must be positive integers.")


def train_model(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epochs: int,
) -> None:
    """Fit one fold-specific model using only that fold's training data."""

    for _ in range(epochs):
        model.train()

        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()


@torch.no_grad()
def predict_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Return validation labels and predictions without parameter updates."""

    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []

    for x_batch, y_batch in loader:
        logits = model(x_batch.to(device))
        predictions = torch.argmax(logits, dim=1)

        y_true.extend(y_batch.numpy().tolist())
        y_pred.extend(predictions.cpu().numpy().tolist())

    return (
        np.asarray(y_true, dtype=np.int64),
        np.asarray(y_pred, dtype=np.int64),
    )


def _divide_or_zero(numerator: int, denominator: int) -> float:
    """Match scikit-learn's zero_division=0 convention."""

    return float(numerator / denominator) if denominator else 0.0


def calculate_binary_fold_audit(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: tuple[str, str],
) -> dict[str, int | float | str]:
    """Calculate and cross-check all counts and metrics for one fold.

    Confusion-matrix rows are actual classes and columns are predicted classes:

        [[actual class 0 predicted class 0, actual class 0 predicted class 1],
         [actual class 1 predicted class 0, actual class 1 predicted class 1]]
    """

    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if matrix.shape != (2, 2):
        raise RuntimeError("Expected a 2 x 2 confusion matrix.")

    true0_pred0 = int(matrix[0, 0])
    true0_pred1 = int(matrix[0, 1])
    true1_pred0 = int(matrix[1, 0])
    true1_pred1 = int(matrix[1, 1])

    actual0 = true0_pred0 + true0_pred1
    actual1 = true1_pred0 + true1_pred1
    predicted0 = true0_pred0 + true1_pred0
    predicted1 = true0_pred1 + true1_pred1

    # Each class is treated as the positive class in turn.
    class0_tp = true0_pred0
    class0_fp = true1_pred0
    class0_fn = true0_pred1
    class0_tn = true1_pred1

    class1_tp = true1_pred1
    class1_fp = true0_pred1
    class1_fn = true1_pred0
    class1_tn = true0_pred0

    precision0 = _divide_or_zero(class0_tp, class0_tp + class0_fp)
    precision1 = _divide_or_zero(class1_tp, class1_tp + class1_fp)
    manual_macro_precision = (precision0 + precision1) / 2.0

    sklearn_macro_precision = float(
        precision_score(
            y_true,
            y_pred,
            labels=[0, 1],
            average="macro",
            zero_division=0,
        )
    )
    if not np.isclose(
        manual_macro_precision,
        sklearn_macro_precision,
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError("Manual and scikit-learn macro precision disagree.")

    manual_accuracy = (true0_pred0 + true1_pred1) / len(y_true)
    sklearn_accuracy = float(accuracy_score(y_true, y_pred))
    if not np.isclose(
        manual_accuracy,
        sklearn_accuracy,
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError("Manual and scikit-learn accuracy disagree.")

    return {
        "class_0_name": class_names[0],
        "class_1_name": class_names[1],
        "validation_n": int(len(y_true)),
        "actual_class_0_n": actual0,
        "actual_class_1_n": actual1,
        "predicted_class_0_n": predicted0,
        "predicted_class_1_n": predicted1,
        "true0_pred0": true0_pred0,
        "true0_pred1": true0_pred1,
        "true1_pred0": true1_pred0,
        "true1_pred1": true1_pred1,
        "class_0_tp": class0_tp,
        "class_0_fp": class0_fp,
        "class_0_fn": class0_fn,
        "class_0_tn": class0_tn,
        "class_1_tp": class1_tp,
        "class_1_fp": class1_fp,
        "class_1_fn": class1_fn,
        "class_1_tn": class1_tn,
        "class_0_precision": precision0,
        "class_1_precision": precision1,
        "macro_precision": sklearn_macro_precision,
        "macro_recall": float(
            recall_score(
                y_true,
                y_pred,
                labels=[0, 1],
                average="macro",
                zero_division=0,
            )
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=[0, 1],
                average="macro",
                zero_division=0,
            )
        ),
        "accuracy": sklearn_accuracy,
    }


def mean_and_sample_sd(values: Iterable[float]) -> dict[str, float | list[float]]:
    """Calculate mean and sample SD using the unrounded fold values."""

    array = np.asarray(list(values), dtype=np.float64)
    if array.ndim != 1 or array.size < 2:
        raise ValueError("At least two one-dimensional values are required.")
    if not np.isfinite(array).all():
        raise ValueError("Metric values must be finite.")

    mean = float(array.mean())
    squared_deviations = np.square(array - mean)
    sum_squared_deviations = float(squared_deviations.sum())
    sample_variance = sum_squared_deviations / (array.size - 1)
    manual_sample_sd = float(np.sqrt(sample_variance))
    numpy_sample_sd = float(array.std(ddof=1))

    if not np.isclose(
        manual_sample_sd,
        numpy_sample_sd,
        rtol=0.0,
        atol=1e-15,
    ):
        raise RuntimeError("Manual and NumPy sample SD disagree.")

    return {
        "fold_values": array.tolist(),
        "mean": mean,
        "sum_squared_deviations": sum_squared_deviations,
        "sample_variance": float(sample_variance),
        "sample_sd": numpy_sample_sd,
    }


def print_fold_audit(record: dict[str, int | float | str]) -> None:
    """Print one fold's raw counts and explicit metric equations."""

    fold = int(record["fold"])
    class0 = str(record["class_0_name"])
    class1 = str(record["class_1_name"])

    a = int(record["true0_pred0"])
    b = int(record["true0_pred1"])
    c = int(record["true1_pred0"])
    d = int(record["true1_pred1"])

    precision0 = float(record["class_0_precision"])
    precision1 = float(record["class_1_precision"])
    macro_precision = float(record["macro_precision"])
    accuracy = float(record["accuracy"])

    print("=" * 78)
    print(f"FOLD {fold}")
    print(
        f"Validation total = {record['validation_n']}; "
        f"actual {class0} = {record['actual_class_0_n']}; "
        f"actual {class1} = {record['actual_class_1_n']}"
    )
    print("Confusion matrix (rows=actual, columns=predicted):")
    print(f"                         predicted {class0}   predicted {class1}")
    print(f"actual {class0:<16} {a:>15} {b:>18}")
    print(f"actual {class1:<16} {c:>15} {d:>18}")
    print()
    print(
        f"Precision({class0}) = {a}/({a}+{c}) "
        f"= {precision0:.12f}"
    )
    print(
        f"Precision({class1}) = {d}/({d}+{b}) "
        f"= {precision1:.12f}"
    )
    print(
        "Macro precision = "
        f"({precision0:.12f}+{precision1:.12f})/2 "
        f"= {macro_precision:.12f}"
    )
    print(
        f"Accuracy = ({a}+{d})/{a+b+c+d} "
        f"= {accuracy:.12f}"
    )


def print_macro_precision_summary(
    result: dict[str, float | list[float]],
) -> None:
    """Print the exact five-value mean and sample-SD calculations."""

    values = np.asarray(result["fold_values"], dtype=np.float64)
    mean = float(result["mean"])
    sum_sq = float(result["sum_squared_deviations"])
    sample_variance = float(result["sample_variance"])
    sample_sd = float(result["sample_sd"])

    value_text = ", ".join(f"{value:.12f}" for value in values)
    sum_text = " + ".join(f"{value:.12f}" for value in values)
    deviation_text = " + ".join(
        f"({value:.12f}-{mean:.12f})^2" for value in values
    )

    print("=" * 78)
    print("FIVE-FOLD MACRO-PRECISION SUMMARY")
    print(f"Unrounded fold values: [{value_text}]")
    print(
        f"Mean = ({sum_text})/{len(values)} "
        f"= {mean:.12f}"
    )
    print(f"Sum of squared deviations = {deviation_text} = {sum_sq:.12f}")
    print(
        f"Sample variance = {sum_sq:.12f}/({len(values)}-1) "
        f"= {sample_variance:.12f}"
    )
    print(
        f"Sample SD = sqrt({sample_variance:.12f}) "
        f"= {sample_sd:.12f}"
    )
    print(f"Report after calculation: {mean:.2f} ± {sample_sd:.2f}")


def export_fold_audit_csv(
    fold_records: list[dict[str, int | float | str]],
    output_file: str | Path,
) -> None:
    """Export full-precision fold-level audit records to a CSV file."""

    if not fold_records:
        raise ValueError("fold_records cannot be empty.")

    destination = Path(output_file)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fold_records[0].keys()))
        writer.writeheader()
        writer.writerows(fold_records)


def run_cross_validation(
    X: np.ndarray,
    y: np.ndarray,
    config: Config = Config(),
    audit_csv: str | Path | None = None,
    verbose: bool = True,
) -> tuple[
    dict[str, dict[str, float | list[float]]],
    list[dict[str, int | float | str]],
]:
    """Run leakage-aware five-fold CV on development data only."""

    X = np.asarray(X, dtype=np.float32)
    y_raw = np.asarray(y)
    validate_inputs(X, y_raw, config)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_raw).astype(np.int64)
    class_names = tuple(str(value) for value in label_encoder.classes_)

    if len(class_names) != 2:
        raise RuntimeError("Expected two encoded class names.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splitter = StratifiedKFold(
        n_splits=config.n_splits,
        shuffle=True,
        random_state=config.seed,
    )

    fold_records: list[dict[str, int | float | str]] = []

    for fold, (train_index, validation_index) in enumerate(
        splitter.split(X, y_encoded),
        start=1,
    ):
        if np.intersect1d(train_index, validation_index).size:
            raise RuntimeError("Training and validation indices overlap.")

        fold_seed = config.seed + fold
        set_seed(fold_seed)

        X_train_raw = X[train_index]
        X_validation_raw = X[validation_index]
        y_train = y_encoded[train_index]
        y_validation = y_encoded[validation_index]

        # Leakage control: learn feature means and scales from training only.
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw).astype(np.float32)
        X_validation = scaler.transform(X_validation_raw).astype(np.float32)

        train_dataset = TensorDataset(
            torch.from_numpy(X_train),
            torch.from_numpy(y_train),
        )
        validation_dataset = TensorDataset(
            torch.from_numpy(X_validation),
            torch.from_numpy(y_validation),
        )

        loader_generator = torch.Generator()
        loader_generator.manual_seed(fold_seed)

        train_loader = DataLoader(
            train_dataset,
            batch_size=min(config.batch_size, len(train_dataset)),
            shuffle=True,
            generator=loader_generator,
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=min(config.batch_size, len(validation_dataset)),
            shuffle=False,
        )

        # Leakage control: new model and optimizer for every fold.
        model = GeneMLP(
            input_dim=X.shape[1],
            num_classes=2,
            hidden1=config.hidden1,
            hidden2=config.hidden2,
            dropout=config.dropout,
        ).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        train_model(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epochs=config.epochs,
        )

        y_true, y_pred = predict_model(model, validation_loader, device)
        audit = calculate_binary_fold_audit(y_true, y_pred, class_names)
        record: dict[str, int | float | str] = {
            "fold": fold,
            "train_n": int(len(train_index)),
        }
        record.update(audit)
        fold_records.append(record)

        if verbose:
            print_fold_audit(record)

    metrics = ("accuracy", "macro_precision", "macro_recall", "macro_f1")
    summary = {
        metric: mean_and_sample_sd(
            float(record[metric]) for record in fold_records
        )
        for metric in metrics
    }

    if verbose:
        print_macro_precision_summary(summary["macro_precision"])

    if audit_csv is not None:
        export_fold_audit_csv(fold_records, audit_csv)

    return summary, fold_records


def make_synthetic_example() -> tuple[np.ndarray, np.ndarray]:
    """Create an anonymized 41-sample binary development example."""

    X_pool, label_pool = make_classification(
        n_samples=200,
        n_features=30,
        n_informative=12,
        n_redundant=6,
        n_classes=2,
        weights=[0.40, 0.60],
        flip_y=0.0,
        class_sep=1.0,
        random_state=42,
    )

    # Select exact fictional class totals, then shuffle the selected rows.
    selected = np.concatenate(
        (
            np.flatnonzero(label_pool == 0)[:16],
            np.flatnonzero(label_pool == 1)[:25],
        )
    )
    rng = np.random.default_rng(42)
    rng.shuffle(selected)

    X = X_pool[selected]
    numeric_labels = label_pool[selected]
    y = np.where(
        numeric_labels == 0,
        "Example_Group_A",
        "Example_Group_B",
    )
    return X, y


def main() -> None:
    """Run the anonymized demonstration and create a fold-audit CSV."""

    X_development, y_development = make_synthetic_example()

    # Five epochs keep this demonstration fast. Use the prespecified study
    # configuration when auditing an actual reported result.
    demonstration_config = Config(epochs=5)

    run_cross_validation(
        X=X_development,
        y=y_development,
        config=demonstration_config,
        audit_csv="cv_fold_audit_example.csv",
        verbose=True,
    )


if __name__ == "__main__":
    main()

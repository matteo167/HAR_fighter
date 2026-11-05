"""Rotina de treino: carrega o dataset, treina o modelo configurado e salva
os artefatos (modelo + metadados) em data/models/.
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit

from har_fighter.classification.movements import MovementRegistry
from har_fighter.features.motion import append_velocity
from har_fighter.features.tilt import build_feature_fn
from har_fighter.training.dataset_loader import load_dataset
from har_fighter.training.model_factory import build_model

logger = logging.getLogger(__name__)


def run_training(settings: dict, registry: MovementRegistry, axis: str) -> Path:
    features_cfg = settings["features"]
    dataset_cfg = settings["dataset"]
    training_cfg = settings["training"]

    feature_mode = features_cfg.get("feature_mode", "full")
    use_velocity = features_cfg.get("use_velocity", False)
    X, y, clip_ids = load_dataset(
        keypoints_dir=Path(dataset_cfg["keypoints_dir"]),
        registry=registry,
        axis=axis,
        sequence_length=features_cfg["sequence_length"],
        stride=features_cfg.get("stride", 1),
        feature_fn=build_feature_fn(feature_mode),
        window_fn=append_velocity if use_velocity else None,
    )
    logger.info("Dataset carregado (eixo=%s): %d amostras, %d classes", axis, len(X), len(registry))

    # Split por CLIPE (não por janela): janelas de um mesmo clipe se
    # sobrepõem muito (sliding window), então um split aleatório por janela
    # deixa quase-duplicatas em treino e validação, inflando a acurácia.
    splitter = GroupShuffleSplit(n_splits=1, test_size=training_cfg["validation_split"], random_state=42)
    train_idx, val_idx = next(splitter.split(X, y, groups=clip_ids))
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    model = build_model(
        architecture=training_cfg["architecture"],
        seq_len=X.shape[1],
        num_landmarks=X.shape[2],
        num_classes=len(registry),
    )
    model.summary()

    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=training_cfg["epochs"],
        batch_size=training_cfg["batch_size"],
    )

    val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
    logger.info("Validação: loss=%.4f accuracy=%.4f", val_loss, val_accuracy)

    y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
    present_labels = sorted(set(y_val.tolist()) | set(y_pred.tolist()))
    label_names = [registry.by_index(i).name for i in present_labels]

    confusion = confusion_matrix(y_val, y_pred, labels=present_labels)
    report = classification_report(
        y_val, y_pred, labels=present_labels, target_names=label_names, output_dict=True, zero_division=0
    )
    logger.info(
        "\n%s", classification_report(y_val, y_pred, labels=present_labels, target_names=label_names, zero_division=0)
    )

    output_dir = Path(training_cfg["output_dir"]) / axis
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    run_dir = output_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    model_path = run_dir / "model.keras"
    model.save(model_path)

    metadata = {
        "axis": axis,
        "movement_names": list(registry.names),
        "sequence_length": X.shape[1],
        "num_landmarks": X.shape[2],
        "architecture": training_cfg["architecture"],
        "feature_mode": feature_mode,
        "use_velocity": use_velocity,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy,
        "trained_at": timestamp,
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    (run_dir / "confusion_matrix.json").write_text(
        json.dumps({"labels": label_names, "matrix": confusion.tolist()}, indent=2)
    )
    (run_dir / "classification_report.json").write_text(json.dumps(report, indent=2))

    latest_dir = output_dir / "latest"
    shutil.rmtree(latest_dir, ignore_errors=True)
    shutil.copytree(run_dir, latest_dir)

    logger.info("Modelo salvo em %s (e copiado para %s)", run_dir, latest_dir)
    return run_dir

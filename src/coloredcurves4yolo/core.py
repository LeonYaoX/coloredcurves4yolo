from __future__ import annotations

import glob
import math
import shutil
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import numpy as np
import yaml

import matplotlib.pyplot as plt


def resolve_model_paths(models: list[str], weight_globs: list[str]) -> list[Path]:
    paths: list[Path] = []

    for model in models:
        model_path = Path(model).expanduser().resolve()
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        paths.append(model_path)

    patterns = weight_globs or ["*.pt", "*.pth", "*.weights"]
    for pattern in patterns:
        expanded_pattern = str(Path(pattern).expanduser())
        for match in sorted(glob.glob(expanded_pattern)):
            model_path = Path(match).expanduser().resolve()
            if model_path.exists():
                paths.append(model_path)

    unique_paths: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        unique_paths.append(path)
    return unique_paths


def load_class_names(yaml_path: Path) -> list[str]:
    with yaml_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    names = config.get("names")
    if isinstance(names, list):
        return [str(name) for name in names]

    if isinstance(names, dict):
        normalized: dict[int, str] = {}
        for key, value in names.items():
            try:
                normalized[int(key)] = str(value)
            except (TypeError, ValueError):
                continue

        if not normalized:
            raise ValueError(f'No valid class names found in "names" of {yaml_path}')

        max_index = max(normalized)
        return [normalized.get(index, f"Class_{index}") for index in range(max_index + 1)]

    raise ValueError(f'Field "names" is missing or invalid in {yaml_path}')


def _reset_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _resolve_device(device: str) -> Any:
    if device != "auto":
        return int(device) if device.isdigit() else device

    try:
        import torch
    except ImportError:  # pragma: no cover
        torch = None

    if torch is not None and torch.cuda.is_available():
        return 0
    return "cpu"


def _extract_curve_data(results: Any) -> dict[str, Any]:
    box = getattr(results, "box", None)
    if box is None:
        raise ValueError("Ultralytics results do not expose a box metrics object.")

    p_curve = getattr(box, "p_curve", None)
    r_curve = getattr(box, "r_curve", None)
    f1_curve = getattr(box, "f1_curve", None)

    arrays = {"P": p_curve, "R": r_curve, "F1": f1_curve}
    for name, value in arrays.items():
        if not isinstance(value, np.ndarray):
            raise ValueError(f"results.box.{name.lower()}_curve is not a NumPy array.")
        if value.ndim != 2:
            raise ValueError(f"results.box.{name.lower()}_curve must be 2D.")

    if p_curve.shape != r_curve.shape or p_curve.shape != f1_curve.shape:
        raise ValueError("Precision, recall, and F1 curves have mismatched shapes.")

    num_thresholds = p_curve.shape[1]
    if num_thresholds == 0:
        raise ValueError("No confidence thresholds were returned by validation.")

    px = getattr(box, "px", None)
    if isinstance(px, np.ndarray) and px.ndim == 1 and px.shape[0] == num_thresholds:
        confidence = px
    else:
        confidence = np.linspace(0.0, 1.0, num_thresholds, dtype=np.float32)

    return {
        "P": p_curve.T,
        "R": r_curve.T,
        "F1": f1_curve.T,
        "confidence": confidence,
        "actual_num_classes": p_curve.shape[0],
    }


def _save_curve_metrics(curve_data: dict[str, Any], output_dir: Path) -> Path:
    metrics_dir = output_dir / "metrics_for_custom_plots"
    _reset_directory(metrics_dir)

    np.save(metrics_dir / "P.npy", curve_data["P"])
    np.save(metrics_dir / "R.npy", curve_data["R"])
    np.save(metrics_dir / "F1.npy", curve_data["F1"])
    np.save(metrics_dir / "confidence.npy", curve_data["confidence"])
    return metrics_dir


def _effective_class_names(class_names: list[str], num_classes: int) -> list[str]:
    names: list[str] = []
    for index in range(num_classes):
        if index < len(class_names):
            names.append(str(class_names[index]))
        else:
            names.append(f"Class_{index}")
    return names


def _get_color_palette(num_classes: int) -> list[Any]:
    colors: list[Any] = []
    for palette in (plt.cm.tab20, plt.cm.Set3, plt.cm.tab20b, plt.cm.tab20c):
        colors.extend(list(palette.colors))

    if num_classes > len(colors):
        colors *= math.ceil(num_classes / len(colors))
    return colors[:num_classes]


def _plot_metrics_vs_confidence(
    confidence: np.ndarray,
    metric_name: str,
    metric_data: np.ndarray,
    class_names: list[str],
    colors: list[Any],
    plots_dir: Path,
    chunk_size: int,
    dpi: int,
) -> None:
    safe_name = metric_name.lower().replace("-", "_")

    for start in range(0, len(class_names), chunk_size):
        stop = min(start + chunk_size, len(class_names))
        figure, axis = plt.subplots(figsize=(12, 8))

        for class_index in range(start, stop):
            if class_index >= metric_data.shape[1]:
                continue
            series = metric_data[:, class_index]
            if np.all(np.isnan(series)):
                continue
            axis.plot(
                confidence,
                series,
                color=colors[class_index],
                linewidth=1.5,
                label=class_names[class_index],
            )

        axis.set_xlabel("Confidence Threshold")
        axis.set_ylabel(metric_name)
        axis.set_title(f"{metric_name} vs Confidence (Classes {start + 1}-{stop})")
        axis.grid(True, alpha=0.5)
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1.05)
        handles, _ = axis.get_legend_handles_labels()
        if handles:
            axis.legend(loc="best", fontsize="small")
        figure.tight_layout()
        figure.savefig(
            plots_dir / f"{safe_name}_vs_conf_chunk_{(start // chunk_size) + 1}.png",
            dpi=dpi,
            bbox_inches="tight",
        )
        plt.close(figure)


def _plot_pr_curves(
    precision: np.ndarray,
    recall: np.ndarray,
    class_names: list[str],
    colors: list[Any],
    plots_dir: Path,
    chunk_size: int,
    dpi: int,
) -> None:
    for start in range(0, len(class_names), chunk_size):
        stop = min(start + chunk_size, len(class_names))
        figure, axis = plt.subplots(figsize=(12, 8))

        for class_index in range(start, stop):
            if class_index >= precision.shape[1] or class_index >= recall.shape[1]:
                continue
            p_values = precision[:, class_index]
            r_values = recall[:, class_index]
            if np.all(np.isnan(p_values)) or np.all(np.isnan(r_values)):
                continue

            sort_indices = np.argsort(r_values)
            axis.plot(
                r_values[sort_indices],
                p_values[sort_indices],
                color=colors[class_index],
                linewidth=1.5,
                label=class_names[class_index],
            )

        axis.set_xlabel("Recall")
        axis.set_ylabel("Precision")
        axis.set_title(f"Precision-Recall Curve (Classes {start + 1}-{stop})")
        axis.grid(True, alpha=0.5)
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1.05)
        handles, _ = axis.get_legend_handles_labels()
        if handles:
            axis.legend(loc="best", fontsize="small")
        figure.tight_layout()
        figure.savefig(
            plots_dir / f"pr_curve_chunk_{(start // chunk_size) + 1}.png",
            dpi=dpi,
            bbox_inches="tight",
        )
        plt.close(figure)


def _plot_colorful_curves(
    metrics_dir: Path,
    class_names: list[str],
    output_dir: Path,
    actual_num_classes: int,
    chunk_size: int,
    dpi: int,
) -> Path:
    plots_dir = output_dir / "colorful_illustrative_curves"
    _reset_directory(plots_dir)

    precision = np.load(metrics_dir / "P.npy")
    recall = np.load(metrics_dir / "R.npy")
    f1_score = np.load(metrics_dir / "F1.npy")
    confidence = np.load(metrics_dir / "confidence.npy")

    num_classes = min(actual_num_classes, precision.shape[1], recall.shape[1], f1_score.shape[1])
    effective_names = _effective_class_names(class_names, num_classes)
    colors = _get_color_palette(num_classes)

    _plot_metrics_vs_confidence(
        confidence=confidence,
        metric_name="Precision",
        metric_data=precision,
        class_names=effective_names,
        colors=colors,
        plots_dir=plots_dir,
        chunk_size=chunk_size,
        dpi=dpi,
    )
    _plot_metrics_vs_confidence(
        confidence=confidence,
        metric_name="Recall",
        metric_data=recall,
        class_names=effective_names,
        colors=colors,
        plots_dir=plots_dir,
        chunk_size=chunk_size,
        dpi=dpi,
    )
    _plot_metrics_vs_confidence(
        confidence=confidence,
        metric_name="F1-score",
        metric_data=f1_score,
        class_names=effective_names,
        colors=colors,
        plots_dir=plots_dir,
        chunk_size=chunk_size,
        dpi=dpi,
    )
    _plot_pr_curves(
        precision=precision,
        recall=recall,
        class_names=effective_names,
        colors=colors,
        plots_dir=plots_dir,
        chunk_size=chunk_size,
        dpi=dpi,
    )
    return plots_dir


def export_model_curves(
    model_path: Path,
    data_yaml: Path,
    output_dir: Path,
    split: str = "val",
    imgsz: int = 640,
    batch: int = 16,
    device: str = "auto",
    chunk_size: int = 10,
    dpi: int = 300,
) -> dict[str, Path]:
    model_path = Path(model_path).expanduser().resolve()
    data_yaml = Path(data_yaml).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data_yaml}")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")

    output_dir.mkdir(parents=True, exist_ok=True)
    native_output_dir = output_dir / "ultralytics_native_outputs"
    _reset_directory(native_output_dir)

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "ultralytics is required to run validation. Install it with "
            '`pip install ultralytics` or `pip install -e .`.'
        ) from exc

    class_names = load_class_names(data_yaml)
    resolved_device = _resolve_device(device)
    model = YOLO(str(model_path))

    with tempfile.TemporaryDirectory(prefix="ultralytics-val-") as temp_dir:
        results = model.val(
            data=str(data_yaml),
            split=split,
            imgsz=imgsz,
            batch=batch,
            project=str(output_dir),
            name="ultralytics_native_outputs",
            exist_ok=True,
            save_json=True,
            plots=True,
            verbose=False,
            device=resolved_device,
        )

    curve_data = _extract_curve_data(results)
    metrics_dir = _save_curve_metrics(curve_data, output_dir)
    plots_dir = _plot_colorful_curves(
        metrics_dir=metrics_dir,
        class_names=class_names,
        output_dir=output_dir,
        actual_num_classes=curve_data["actual_num_classes"],
        chunk_size=chunk_size,
        dpi=dpi,
    )

    return {
        "metrics_dir": metrics_dir,
        "plots_dir": plots_dir,
    }

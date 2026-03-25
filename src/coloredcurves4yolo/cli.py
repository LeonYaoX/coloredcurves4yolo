from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export Ultralytics native validation artifacts together with "
            "metrics_for_custom_plots and colorful_illustrative_curves."
        )
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="Path to a model weight file. Can be passed multiple times.",
    )
    parser.add_argument(
        "--weights-glob",
        action="append",
        default=[],
        help='Glob pattern for model discovery, for example "weights/*.pt".',
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Path to dataset YAML.",
    )
    parser.add_argument(
        "--output",
        default="outputs",
        help="Directory to write exported artifacts into.",
    )
    parser.add_argument(
        "--split",
        default="val",
        help="Dataset split for validation. Default: val.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Validation image size. Default: 640.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Validation batch size. Default: 16.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help='Validation device such as "auto", "cpu", "cuda", "cuda:0", or "0".',
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=10,
        help="Number of classes per figure. Default: 10.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Saved figure DPI. Default: 300.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.chunk_size <= 0:
        parser.error("--chunk-size must be greater than 0.")
    if args.batch <= 0:
        parser.error("--batch must be greater than 0.")
    if args.imgsz <= 0:
        parser.error("--imgsz must be greater than 0.")
    if args.dpi <= 0:
        parser.error("--dpi must be greater than 0.")

    data_path = Path(args.data).expanduser().resolve()
    if not data_path.exists():
        parser.error(f"Dataset YAML not found: {data_path}")

    from .core import export_model_curves, resolve_model_paths

    try:
        model_paths = resolve_model_paths(args.model, args.weights_glob)
    except FileNotFoundError as exc:
        parser.error(str(exc))

    if not model_paths:
        parser.error(
            "No model files found. Pass --model or --weights-glob to select weights."
        )

    output_root = Path(args.output).expanduser().resolve()
    single_model = len(model_paths) == 1

    for model_path in model_paths:
        target_dir = output_root if single_model else output_root / model_path.stem
        print(f"[export] {model_path} -> {target_dir}")
        export_model_curves(
            model_path=model_path,
            data_yaml=data_path,
            output_dir=target_dir,
            split=args.split,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            chunk_size=args.chunk_size,
            dpi=args.dpi,
        )

    print(f"[done] Exported {len(model_paths)} model(s) into {output_root}")
    return 0

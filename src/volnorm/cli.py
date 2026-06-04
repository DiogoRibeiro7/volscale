import argparse
import json
from pathlib import Path

import pandas as pd
from .features import FeatureConfig, build_normalized_features


def _parse_args(args=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate volatility-normalized features from a CSV file"
    )
    parser.add_argument("input_csv", help="Input CSV containing a price column")
    parser.add_argument(
        "--column",
        default="close",
        help="Name of the price column to use (default: close)",
    )
    parser.add_argument(
        "--window", type=int, default=20, help="Rolling window size (default: 20)"
    )
    parser.add_argument(
        "--output", help="Optional output CSV path. Prints to stdout if omitted"
    )
    parser.add_argument(
        "--input-sep",
        default=",",
        help="Field separator for the input file (default: ,)",
    )
    parser.add_argument(
        "--output-sep",
        default=",",
        help="Field separator for CSV output (default: ,)",
    )
    parser.add_argument(
        "--features",
        help="Comma-separated feature names to include. Supported: price_minus_sma, atr_proxy, log_return",
    )
    parser.add_argument(
        "--include-volatility",
        action="store_true",
        help="Append the rolling volatility column used for normalization",
    )
    parser.add_argument(
        "--date-column",
        help="Optional date column to parse and set as the index",
    )
    parser.add_argument(
        "--keep-index",
        action="store_true",
        help="Preserve the parsed index in the output",
    )
    parser.add_argument(
        "--dropna",
        action="store_true",
        help="Drop rows with missing feature values before output",
    )
    parser.add_argument(
        "--sort-index",
        action="store_true",
        help="Sort by the parsed index before feature generation",
    )
    parser.add_argument(
        "--output-format",
        choices=("csv", "json"),
        default="csv",
        help="Output format to write or print (default: csv)",
    )
    return parser.parse_args(args)


def _load_input_frame(args: argparse.Namespace) -> pd.DataFrame:
    try:
        df = pd.read_csv(args.input_csv, sep=args.input_sep)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc
    except pd.errors.EmptyDataError as exc:
        raise SystemExit(f"Input file '{args.input_csv}' is empty") from exc
    except pd.errors.ParserError as exc:
        raise SystemExit(f"Failed to parse '{args.input_csv}': {exc}") from exc

    if df.empty:
        raise SystemExit(f"Input file '{args.input_csv}' contains no rows")

    if args.date_column:
        if args.date_column not in df.columns:
            raise SystemExit(
                f"Date column '{args.date_column}' not found in {args.input_csv}"
            )
        try:
            df[args.date_column] = pd.to_datetime(df[args.date_column], errors="raise")
        except (ValueError, TypeError) as exc:
            raise SystemExit(
                f"Failed to parse date column '{args.date_column}': {exc}"
            ) from exc
        df = df.set_index(args.date_column)
        if args.sort_index:
            df = df.sort_index()

    return df


def _build_feature_frame(args: argparse.Namespace, df: pd.DataFrame) -> pd.DataFrame:
    if args.column not in df.columns:
        raise SystemExit(f"Column '{args.column}' not found in {args.input_csv}")

    include = None
    if args.features:
        include = [
            feature.strip() for feature in args.features.split(",") if feature.strip()
        ]

    try:
        config = FeatureConfig.from_inputs(
            window=args.window,
            include=include,
            append_volatility=args.include_volatility,
        )
        features = build_normalized_features(df[args.column], config=config)
    except (TypeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    if args.keep_index:
        features.index = df.index
    if args.dropna:
        features = features.dropna()
    return features


def _render_output(args: argparse.Namespace, features: pd.DataFrame) -> str:
    if args.output_format == "json":
        payload = features.reset_index(drop=not args.keep_index).to_dict(
            orient="records"
        )
        return json.dumps(payload, default=str, indent=2)

    return features.to_csv(index=args.keep_index, sep=args.output_sep)


def _write_output(args: argparse.Namespace, rendered: str) -> None:
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered, encoding="utf-8")
    else:
        print(rendered)


def main(argv=None) -> None:
    args = _parse_args(argv)
    df = _load_input_frame(args)
    features = _build_feature_frame(args, df)
    rendered = _render_output(args, features)
    _write_output(args, rendered)


if __name__ == "__main__":
    main()

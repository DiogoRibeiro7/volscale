import argparse
import pandas as pd
from .features import build_normalized_features


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
    return parser.parse_args(args)


def main(argv=None) -> None:
    args = _parse_args(argv)
    try:
        df = pd.read_csv(args.input_csv)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc

    if args.date_column:
        if args.date_column not in df.columns:
            raise SystemExit(
                f"Date column '{args.date_column}' not found in {args.input_csv}"
            )
        df[args.date_column] = pd.to_datetime(df[args.date_column], errors="raise")
        df = df.set_index(args.date_column)

    if args.column not in df.columns:
        raise SystemExit(f"Column '{args.column}' not found in {args.input_csv}")

    include = None
    if args.features:
        include = [
            feature.strip() for feature in args.features.split(",") if feature.strip()
        ]

    try:
        features = build_normalized_features(
            df[args.column],
            window=args.window,
            include=include,
            append_volatility=args.include_volatility,
        )
    except (TypeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    if args.output:
        features.to_csv(args.output, index=False)
    else:
        print(features.to_csv(index=False))


if __name__ == "__main__":
    main()

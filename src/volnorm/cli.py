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
    return parser.parse_args(args)


def main(argv=None) -> None:
    args = _parse_args(argv)
    df = pd.read_csv(args.input_csv)
    if args.column not in df.columns:
        raise SystemExit(f"Column '{args.column}' not found in {args.input_csv}")

    features = build_normalized_features(df[args.column], window=args.window)

    if args.output:
        features.to_csv(args.output, index=False)
    else:
        print(features.to_csv(index=False))


if __name__ == "__main__":
    main()

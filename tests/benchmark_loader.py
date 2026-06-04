from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


DATA_DIR = Path(__file__).resolve().parent / "data"


def _load_json(name: str) -> dict:
    with (DATA_DIR / name).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_atr_benchmark() -> dict:
    return _load_json("atr_benchmark.json")


def load_black_scholes_benchmarks() -> list[dict]:
    payload = _load_json("black_scholes_benchmarks.json")
    return list(payload["cases"])


def load_realized_volatility_benchmark() -> dict:
    payload = _load_json("realized_volatility_benchmark.json")
    payload["index"] = pd.DatetimeIndex(payload["timestamps"])
    payload["expected_daily"] = pd.Series(
        payload["expected_daily"],
        index=pd.DatetimeIndex(payload["expected_daily_index"], freq="D"),
    )
    return payload

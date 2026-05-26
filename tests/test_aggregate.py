"""Test the headline aggregation invariants."""

import pandas as pd
from pathlib import Path

from src.aggregate import HEADLINE_ORDER, TRACEABILITY, build_traceability

ROOT = Path(__file__).resolve().parent.parent


def test_headline_schema_has_20_rows():
    assert len(HEADLINE_ORDER) == 20


def test_headline_metric_ids_unique():
    ids = [r[0] for r in HEADLINE_ORDER]
    assert len(set(ids)) == 20
    expected = [f"M{i:02d}" for i in range(1, 21)]
    assert ids == expected


def test_traceability_has_20_rows():
    df = build_traceability()
    assert len(df) == 20
    assert set(df["metric_id"]) == set(f"M{i:02d}" for i in range(1, 21))
    assert (df["application_phrase"].str.len() > 0).all()


def test_headline_file_if_present_has_20_rows():
    p = ROOT / "outputs" / "tables" / "headline_numbers.csv"
    if p.exists():
        df = pd.read_csv(p)
        assert len(df) == 20
        assert df["value"].notna().all(), \
            f"Null values in headline: {df[df['value'].isna()]['metric_id'].tolist()}"

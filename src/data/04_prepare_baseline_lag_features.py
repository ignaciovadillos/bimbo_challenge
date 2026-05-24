"""Leakage-safe lag feature builder for the Bimbo baseline model.

This script follows section 6.1.1 of docs/PT_HW2_Bimbo_Kaggle-1.pdf:
lag-1, lag-2, lag-3, and short rolling demand features. It computes the
lags at weekly aggregate grain first, then merges them back to row grain so
duplicate same-week rows cannot leak into each other.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "csv"
DEFAULT_INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
DEFAULT_TRAIN_OUTPUT_PATH = DEFAULT_INTERIM_DIR / "train_baseline_lag_features.parquet"
DEFAULT_TEST_OUTPUT_PATH = DEFAULT_INTERIM_DIR / "test_baseline_lag_features.parquet"

WEEK_COL = "Semana"
TARGET_COL = "Demanda_uni_equil"
SALES_UNITS_COL = "Venta_uni_hoy"
RETURN_UNITS_COL = "Dev_uni_proxima"
ID_COL = "id"

BASE_COLUMNS = [
    WEEK_COL,
    "Agencia_ID",
    "Canal_ID",
    "Ruta_SAK",
    "Cliente_ID",
    "Producto_ID",
]

TRAIN_DTYPES = {
    WEEK_COL: "int8",
    "Agencia_ID": "int32",
    "Canal_ID": "int8",
    "Ruta_SAK": "int32",
    "Cliente_ID": "int32",
    "Producto_ID": "int32",
    SALES_UNITS_COL: "int32",
    RETURN_UNITS_COL: "int32",
    TARGET_COL: "int32",
}

TEST_DTYPES = {
    ID_COL: "int32",
    WEEK_COL: "int8",
    "Agencia_ID": "int32",
    "Canal_ID": "int8",
    "Ruta_SAK": "int32",
    "Cliente_ID": "int32",
    "Producto_ID": "int32",
}

PRECISE_LAG_KEY = ["Producto_ID", "Agencia_ID", "Cliente_ID"]
AGGREGATE_GROUPINGS = [
    ("sku_agency", ["Producto_ID", "Agencia_ID"]),
    ("sku_channel", ["Producto_ID", "Canal_ID"]),
    ("sku_route", ["Producto_ID", "Agencia_ID", "Ruta_SAK"]),
    ("sku", ["Producto_ID"]),
    ("agency_channel", ["Agencia_ID", "Canal_ID"]),
]
RETURN_RATE_KEY = ["Producto_ID", "Agencia_ID", "Canal_ID"]
LAGS = (1, 2, 3)


def prefixed_name(prefix: str, suffix: str) -> str:
    return f"{prefix}_{suffix}" if prefix else suffix


def read_train(train_path: Path, nrows: int | None) -> pd.DataFrame:
    usecols = BASE_COLUMNS + [SALES_UNITS_COL, RETURN_UNITS_COL, TARGET_COL]
    return pd.read_csv(train_path, usecols=usecols, dtype=TRAIN_DTYPES, nrows=nrows)


def read_test(test_path: Path, nrows: int | None) -> pd.DataFrame:
    usecols = [ID_COL] + BASE_COLUMNS
    return pd.read_csv(test_path, usecols=usecols, dtype=TEST_DTYPES, nrows=nrows)


def add_week_features(rows: pd.DataFrame) -> pd.DataFrame:
    output = rows.copy()
    output["week_index"] = output[WEEK_COL].astype("int16")
    output["is_week_10"] = (output[WEEK_COL] == 10).astype("int8")
    output["is_week_11"] = (output[WEEK_COL] == 11).astype("int8")
    return output


def merge_lagged_target_aggregates(
    rows: pd.DataFrame,
    train: pd.DataFrame,
    keys: list[str],
    prefix: str,
    agg: str,
    lags: tuple[int, ...] = LAGS,
) -> pd.DataFrame:
    weekly_col = prefixed_name(prefix, "weekly_demand")
    weekly = (
        train.groupby(keys + [WEEK_COL], observed=True)[TARGET_COL]
        .agg(agg)
        .reset_index(name=weekly_col)
    )

    output = rows
    lag_columns = []
    for lag in lags:
        feature_col = prefixed_name(prefix, f"demand_lag{lag}")
        lagged = weekly.copy()
        lagged[WEEK_COL] = lagged[WEEK_COL] + lag
        lagged = lagged.rename(columns={weekly_col: feature_col})
        output = output.merge(lagged, on=keys + [WEEK_COL], how="left")
        lag_columns.append(feature_col)

    lag_count_col = prefixed_name(prefix, "demand_lag_count")
    output[prefixed_name(prefix, "demand_roll2_mean")] = output[lag_columns[:2]].mean(
        axis=1
    )
    output[prefixed_name(prefix, "demand_roll3_mean")] = output[lag_columns].mean(axis=1)
    output[prefixed_name(prefix, "demand_roll3_std")] = output[lag_columns].std(axis=1)
    output[lag_count_col] = output[lag_columns].notna().sum(axis=1).astype("int8")
    output[prefixed_name(prefix, "demand_cold_start")] = (
        output[lag_count_col] == 0
    ).astype("int8")
    return output


def merge_lagged_return_rate_features(
    rows: pd.DataFrame,
    train: pd.DataFrame,
    keys: list[str] = RETURN_RATE_KEY,
    prefix: str = "sku_agency_channel",
    lags: tuple[int, ...] = LAGS,
) -> pd.DataFrame:
    weekly = (
        train.groupby(keys + [WEEK_COL], observed=True)
        .agg(
            weekly_sales_units=(SALES_UNITS_COL, "sum"),
            weekly_return_units=(RETURN_UNITS_COL, "sum"),
        )
        .reset_index()
    )
    weekly[f"{prefix}_return_rate"] = (
        weekly["weekly_return_units"] / (weekly["weekly_sales_units"] + 1.0)
    )

    output = rows
    rate_columns = []
    for lag in lags:
        lagged = weekly.copy()
        lagged[WEEK_COL] = lagged[WEEK_COL] + lag
        lagged = lagged.rename(
            columns={
                "weekly_sales_units": f"{prefix}_sales_units_lag{lag}",
                "weekly_return_units": f"{prefix}_return_units_lag{lag}",
                f"{prefix}_return_rate": f"{prefix}_return_rate_lag{lag}",
            }
        )
        output = output.merge(
            lagged[
                keys
                + [WEEK_COL]
                + [
                    f"{prefix}_sales_units_lag{lag}",
                    f"{prefix}_return_units_lag{lag}",
                    f"{prefix}_return_rate_lag{lag}",
                ]
            ],
            on=keys + [WEEK_COL],
            how="left",
        )
        rate_columns.append(f"{prefix}_return_rate_lag{lag}")

    output[f"{prefix}_return_rate_roll3_mean"] = output[rate_columns].mean(axis=1)
    output[f"{prefix}_return_rate_lag_count"] = (
        output[rate_columns].notna().sum(axis=1).astype("int8")
    )
    return output


def merge_lookup_features(rows: pd.DataFrame, interim_dir: Path) -> pd.DataFrame:
    output = rows

    product_path = interim_dir / "product_regex_features.parquet"
    if product_path.exists():
        product_features = pd.read_parquet(
            product_path,
            columns=[
                "Producto_ID",
                "product_weight_g",
                "product_volume_ml",
                "product_units_per_pack",
                "product_weight_per_unit_g",
            ],
        )
        product_features = product_features.drop_duplicates("Producto_ID")
        output = output.merge(
            product_features,
            on="Producto_ID",
            how="left",
            validate="many_to_one",
        )

    geolocation_path = interim_dir / "agency_geolocation_features.parquet"
    if geolocation_path.exists():
        geolocation_features = pd.read_parquet(
            geolocation_path,
            columns=[
                "Agencia_ID",
                "lat",
                "lon",
                "geo_cluster",
                "dist_to_hub_km",
                "lat_zone",
            ],
        )
        geolocation_features["lat_zone_code"] = geolocation_features["lat_zone"].map(
            {"south": 0, "central": 1, "north": 2}
        )
        geolocation_features = geolocation_features.drop(columns=["lat_zone"])
        geolocation_features = geolocation_features.drop_duplicates("Agencia_ID")
        output = output.merge(
            geolocation_features,
            on="Agencia_ID",
            how="left",
            validate="many_to_one",
        )

    client_path = interim_dir / "client_chain_features.parquet"
    if client_path.exists():
        client_features = pd.read_parquet(client_path)
        client_columns = [
            column
            for column in client_features.columns
            if column == "Cliente_ID" or column.startswith("client_chain_")
        ]
        client_columns.append("client_is_chain")
        client_columns = list(dict.fromkeys(client_columns))
        client_features = client_features[client_columns]
        for column in client_features.columns:
            if column != "Cliente_ID":
                client_features[column] = client_features[column].astype("int8")
        client_features = client_features.groupby("Cliente_ID", as_index=False).max()
        output = output.merge(
            client_features,
            on="Cliente_ID",
            how="left",
            validate="many_to_one",
        )

    return output


def downcast_feature_columns(features: pd.DataFrame) -> pd.DataFrame:
    output = features.copy()
    for column in output.columns:
        if column in BASE_COLUMNS or column == ID_COL or column == TARGET_COL:
            continue
        if pd.api.types.is_float_dtype(output[column]):
            output[column] = output[column].astype("float32")
        elif pd.api.types.is_integer_dtype(output[column]):
            if output[column].isna().any():
                output[column] = output[column].astype("float32")
                continue
            minimum = output[column].min()
            maximum = output[column].max()
            if minimum >= np.iinfo(np.int8).min and maximum <= np.iinfo(np.int8).max:
                output[column] = output[column].astype("int8")
            elif minimum >= np.iinfo(np.int16).min and maximum <= np.iinfo(np.int16).max:
                output[column] = output[column].astype("int16")
            elif minimum >= np.iinfo(np.int32).min and maximum <= np.iinfo(np.int32).max:
                output[column] = output[column].astype("int32")
    return output


def build_baseline_lag_features(
    train: pd.DataFrame,
    test: pd.DataFrame | None,
    interim_dir: Path,
    include_lookup_features: bool,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    train_rows = add_week_features(train[BASE_COLUMNS + [TARGET_COL]].copy())
    train_rows["is_test_row"] = 0

    if test is None:
        all_rows = train_rows
    else:
        test_rows = add_week_features(test[[ID_COL] + BASE_COLUMNS].copy())
        test_rows[TARGET_COL] = np.nan
        test_rows["is_test_row"] = 1
        all_rows = pd.concat([train_rows, test_rows], ignore_index=True, sort=False)

    all_rows = merge_lagged_target_aggregates(
        all_rows,
        train,
        keys=PRECISE_LAG_KEY,
        prefix="",
        agg="sum",
    )

    for prefix, keys in AGGREGATE_GROUPINGS:
        all_rows = merge_lagged_target_aggregates(
            all_rows,
            train,
            keys=keys,
            prefix=f"{prefix}_mean",
            agg="mean",
        )

    all_rows = merge_lagged_return_rate_features(all_rows, train)

    if include_lookup_features:
        all_rows = merge_lookup_features(all_rows, interim_dir)

    all_rows = downcast_feature_columns(all_rows)

    train_features = all_rows[all_rows["is_test_row"] == 0].copy().drop(
        columns=["is_test_row"]
    )
    train_features = train_features.drop(columns=[ID_COL], errors="ignore")
    train_features[TARGET_COL] = train_features[TARGET_COL].astype(
        TRAIN_DTYPES[TARGET_COL]
    )

    test_features = None
    if test is not None:
        test_features = all_rows[all_rows["is_test_row"] == 1].copy().drop(
            columns=["is_test_row"]
        )
        test_features = test_features.drop(columns=[TARGET_COL], errors="ignore")
        test_features[ID_COL] = test_features[ID_COL].astype(TEST_DTYPES[ID_COL])

    return train_features, test_features


def write_table(data: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix == ".parquet":
        data.to_parquet(output_path, index=False)
    elif output_path.suffix == ".csv":
        data.to_csv(output_path, index=False)
    else:
        raise ValueError("Output path must end in .parquet or .csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create leakage-safe baseline lag features from section 6.1.1 of the "
            "Bimbo homework PDF, plus sparse-history backoff features."
        )
    )
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--interim-dir", type=Path, default=DEFAULT_INTERIM_DIR)
    parser.add_argument(
        "--train-output-path",
        type=Path,
        default=DEFAULT_TRAIN_OUTPUT_PATH,
    )
    parser.add_argument(
        "--test-output-path",
        type=Path,
        default=DEFAULT_TEST_OUTPUT_PATH,
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Only write train features; useful for validation experiments.",
    )
    parser.add_argument(
        "--no-lookup-features",
        action="store_true",
        help="Do not merge product, geolocation, or client-chain interim features.",
    )
    parser.add_argument(
        "--nrows",
        type=int,
        default=None,
        help="Optional row limit for quick local checks.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train = read_train(args.raw_dir / "train.csv", nrows=args.nrows)
    test = None
    if not args.skip_test:
        test = read_test(args.raw_dir / "test.csv", nrows=args.nrows)

    train_features, test_features = build_baseline_lag_features(
        train=train,
        test=test,
        interim_dir=args.interim_dir,
        include_lookup_features=not args.no_lookup_features,
    )

    write_table(train_features, args.train_output_path)
    print(f"Wrote {len(train_features):,} train rows to {args.train_output_path}")

    if test_features is not None:
        write_table(test_features, args.test_output_path)
        print(f"Wrote {len(test_features):,} test rows to {args.test_output_path}")

    feature_count = len(
        [column for column in train_features.columns if column != TARGET_COL]
    )
    print(f"Feature columns including IDs: {feature_count:,}")
    print(
        "Core lag columns: demand_lag1, demand_lag2, "
        "demand_lag3, demand_roll2_mean"
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
RAW_CSV_DIR = RAW_DIR / "csv"
INTERIM_DIR = DATA_DIR / "interim"
RAW_ZIP_PATH = RAW_DIR / "grupo-bimbo-inventory-demand.zip"

SCRIPT_DIR = Path(__file__).resolve().parent
UNZIP_SCRIPT = SCRIPT_DIR / "01_unzip_bimbo.py"
REGEX_SCRIPT = SCRIPT_DIR / "02_raw_regex_features.py"
GEOLOCATION_SCRIPT = SCRIPT_DIR / "03_prepare_geolocation_features.py"


def run_step(name: str, command: list[str]) -> None:
    print(f"\n=== {name} ===", flush=True)
    print(" ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def validate_raw_zip() -> None:
    if not RAW_ZIP_PATH.exists():
        raise FileNotFoundError(
            "Missing raw Kaggle zip. Place it at "
            f"{RAW_ZIP_PATH.relative_to(PROJECT_ROOT)}"
        )


def validate_extracted_csvs() -> None:
    required_files = [
        "train.csv",
        "test.csv",
        "producto_tabla.csv",
        "cliente_tabla.csv",
        "town_state.csv",
        "sample_submission.csv",
    ]
    missing = [name for name in required_files if not (RAW_CSV_DIR / name).exists()]
    if missing:
        missing_list = ", ".join(missing)
        raise FileNotFoundError(
            "Missing extracted raw CSV files in "
            f"{RAW_CSV_DIR.relative_to(PROJECT_ROOT)}: {missing_list}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Bimbo data preparation scripts in order."
    )
    parser.add_argument(
        "--skip-unzip",
        action="store_true",
        help="Skip 01_unzip_bimbo.py when raw CSVs are already extracted.",
    )
    parser.add_argument(
        "--regex-output-format",
        choices=["parquet", "csv"],
        default="parquet",
        help="Output format for 02_raw_regex_features.py.",
    )
    parser.add_argument(
        "--geo-output-path",
        type=Path,
        default=INTERIM_DIR / "agency_geolocation_features.parquet",
    )
    parser.add_argument(
        "--geocode-cache-path",
        type=Path,
        default=INTERIM_DIR / "geocode_cache.json",
    )
    parser.add_argument("--n-geo-clusters", type=int, default=8)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--skip-online-geocoding",
        action="store_true",
        help="Use cache and state-centroid fallback only; do not call Nominatim.",
    )
    parser.add_argument(
        "--max-geocode-requests",
        type=int,
        default=None,
        help="Optional cap for online Nominatim requests while testing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.skip_unzip:
        print("\n=== 01 unzip raw Kaggle data ===", flush=True)
        print("Skipped because --skip-unzip was provided.", flush=True)
        validate_extracted_csvs()
    else:
        validate_raw_zip()
        run_step("01 unzip raw Kaggle data", [sys.executable, str(UNZIP_SCRIPT)])
        validate_extracted_csvs()

    run_step(
        "02 extract regex and keyword features",
        [
            sys.executable,
            str(REGEX_SCRIPT),
            "--raw-dir",
            str(RAW_CSV_DIR),
            "--output-dir",
            str(INTERIM_DIR),
            "--output-format",
            args.regex_output_format,
        ],
    )

    geolocation_command = [
        sys.executable,
        str(GEOLOCATION_SCRIPT),
        "--raw-path",
        str(RAW_CSV_DIR / "town_state.csv"),
        "--output-path",
        str(args.geo_output_path),
        "--cache-path",
        str(args.geocode_cache_path),
        "--n-clusters",
        str(args.n_geo_clusters),
        "--random-state",
        str(args.random_state),
    ]

    if args.skip_online_geocoding:
        geolocation_command.append("--skip-online")
    if args.max_geocode_requests is not None:
        geolocation_command.extend(["--max-requests", str(args.max_geocode_requests)])

    run_step("03 prepare geolocation features", geolocation_command)
    print("\nPipeline complete.")


if __name__ == "__main__":
    main()

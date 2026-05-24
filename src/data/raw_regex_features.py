from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "interim"

PRODUCT_ID_COL = "Producto_ID"
PRODUCT_NAME_COL = "NombreProducto"
CLIENT_ID_COL = "Cliente_ID"
CLIENT_NAME_COL = "NombreCliente"
AGENCY_ID_COL = "Agencia_ID"
TOWN_COL = "Town"
STATE_COL = "State"

PRODUCT_ID_SUFFIX_RE = re.compile(r"\s+\d+\s*$")
SIZE_RE = re.compile(
    r"(?<!\d)(\d+(?:[.,]\d+|\s+\d{1,2})?)\s*(kg|gr|g|ml)\b",
    flags=re.IGNORECASE,
)
UNITS_RE = re.compile(r"(?<!\d)(\d+)\s*(?:pzas?|pz|pq|p)\b", flags=re.IGNORECASE)
BRAND_CODE_RE = re.compile(r"\b([A-Z]{2,5})\b$")

PRODUCT_FORMAT_PATTERNS = {
    "caja": re.compile(r"\b(?:CAJA|CJ)\b", flags=re.IGNORECASE),
    "bolsa": re.compile(r"\bBOLSA\b", flags=re.IGNORECASE),
    "tira": re.compile(r"\bTIRA\b", flags=re.IGNORECASE),
    "lata": re.compile(r"\bLATA\b", flags=re.IGNORECASE),
    "cubeta": re.compile(r"\bCUBETA\b", flags=re.IGNORECASE),
    "tableta": re.compile(r"\bTAB\b", flags=re.IGNORECASE),
    "multipack": re.compile(r"\b(?:MTA|PROM|PACK)\b", flags=re.IGNORECASE),
}

UNKNOWN_CLIENT_RE = re.compile(r"\b(?:NO IDENTIFICADO|SIN NOMBRE)\b")
CHAIN_PATTERNS = {
    "oxxo": re.compile(r"\bOXXO\b"),
    "7_eleven": re.compile(r"\b(?:7\s+ELEVEN|SEVEN\s+ELEVEN|7\s+SEVEN\s+ELEVEN)\b"),
    "walmart": re.compile(r"\b(?:WAL\s*MART|WALMART)\b"),
    "bodega_aurrera": re.compile(r"\bBODEGA\s+AURRERA\b"),
    "soriana": re.compile(r"\bSORIANA\b"),
    "chedraui": re.compile(r"\bCHEDRAUI\b"),
    "comercial_mexicana": re.compile(r"\bCOMERCIAL\s+MEXICANA\b"),
    "farmacia_guadalajara": re.compile(r"\bFARMACIA\s+GUADALAJARA\b"),
    "extra": re.compile(r"\bEXTRA\b"),
    "kiosko": re.compile(r"\bKIOSKO\b"),
    "calimax": re.compile(r"\bCALIMAX\b"),
    "superama": re.compile(r"\bSUPERAMA\b"),
    "sams_club": re.compile(r"\bSAMS\s+CLUB\b"),
    "costco": re.compile(r"\bCOSTCO\b"),
}

TOWN_NUMERIC_PREFIX_RE = re.compile(r"^\s*\d+\s*")
TOWN_TAG_PREFIX_RE = re.compile(
    r"^(?:(?:AG|SUC|BODEGA|BOD|CEDIS|DEP)\.?\s*)+",
    flags=re.IGNORECASE,
)


def normalize_match_text(value: object) -> str:
    """Normalize accents, punctuation, and spacing for keyword matching."""
    if pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKD", str(value))
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^A-Za-z0-9]+", " ", text.upper())
    return re.sub(r"\s+", " ", text).strip()


def strip_product_id_suffix(name: object) -> str:
    if pd.isna(name):
        return ""
    return PRODUCT_ID_SUFFIX_RE.sub("", str(name)).strip()


def _parse_decimal(raw_value: str) -> float:
    value = raw_value.strip().replace(",", ".")
    parts = value.split()
    if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
        value = f"{parts[0]}.{parts[1]}"
    return float(value)


def _last_size_match(name: object) -> re.Match[str] | None:
    core_name = strip_product_id_suffix(name)
    matches = list(SIZE_RE.finditer(core_name))
    return matches[-1] if matches else None


def extract_weight_g(name: object) -> float:
    """Extract product weight in grams from NombreProducto."""
    match = _last_size_match(name)
    if not match:
        return np.nan

    value = _parse_decimal(match.group(1))
    unit = match.group(2).lower()
    if unit == "kg":
        return value * 1000.0
    if unit in {"g", "gr"}:
        return value
    return np.nan


def extract_volume_ml(name: object) -> float:
    """Extract product volume in milliliters when NombreProducto uses ml."""
    match = _last_size_match(name)
    if not match or match.group(2).lower() != "ml":
        return np.nan
    return _parse_decimal(match.group(1))


def extract_units_per_pack(name: object) -> float:
    """Extract pack count such as 6p, 25pz, or 2pq from NombreProducto."""
    core_name = strip_product_id_suffix(name)
    matches = list(UNITS_RE.finditer(core_name))
    if not matches:
        return 1.0
    return float(matches[-1].group(1))


def extract_product_format(name: object) -> str | None:
    core_name = strip_product_id_suffix(name)
    for format_name, pattern in PRODUCT_FORMAT_PATTERNS.items():
        if pattern.search(core_name):
            return format_name
    return None


def extract_brand_code(name: object) -> str | None:
    core_name = strip_product_id_suffix(name).strip()
    match = BRAND_CODE_RE.search(core_name)
    return match.group(1) if match else None


def extract_chain_membership(name: object) -> str:
    """Return a canonical chain label, unknown, or independent."""
    normalized = normalize_match_text(name)
    if not normalized or UNKNOWN_CLIENT_RE.search(normalized):
        return "unknown"

    for chain_name, pattern in CHAIN_PATTERNS.items():
        if pattern.search(normalized):
            return chain_name

    return "independent"


def clean_town_for_geocoding(raw_town: object) -> str:
    """Strip internal agency code and leading abbreviations from Town."""
    if pd.isna(raw_town):
        return ""

    town = str(raw_town).replace("_", " ")
    town = TOWN_NUMERIC_PREFIX_RE.sub("", town)
    town = TOWN_TAG_PREFIX_RE.sub("", town)
    town = re.sub(r"\s+", " ", town).strip(" .,-")
    return town


def make_geocode_query(raw_town: object, state: object) -> str:
    clean_town = clean_town_for_geocoding(raw_town)
    clean_state = "" if pd.isna(state) else re.sub(r"\s+", " ", str(state)).strip()
    return ", ".join(part for part in [clean_town, clean_state, "Mexico"] if part)


def add_product_regex_features(
    products: pd.DataFrame,
    name_col: str = PRODUCT_NAME_COL,
) -> pd.DataFrame:
    features = products.copy()
    names = features[name_col]

    features["product_name_core"] = names.map(strip_product_id_suffix)
    features["product_weight_g"] = names.map(extract_weight_g)
    features["product_volume_ml"] = names.map(extract_volume_ml)
    features["product_units_per_pack"] = names.map(extract_units_per_pack)
    features["product_weight_per_unit_g"] = (
        features["product_weight_g"] / features["product_units_per_pack"]
    )
    features["product_format"] = names.map(extract_product_format)
    features["product_brand_code"] = names.map(extract_brand_code)
    return features


def add_client_chain_features(
    clients: pd.DataFrame,
    name_col: str = CLIENT_NAME_COL,
) -> pd.DataFrame:
    features = clients.copy()
    normalized_names = features[name_col].map(normalize_match_text)

    features["client_name_normalized"] = normalized_names
    features["client_chain"] = features[name_col].map(extract_chain_membership)
    features["client_is_chain"] = ~features["client_chain"].isin(
        ["independent", "unknown"]
    )

    for chain_name, pattern in CHAIN_PATTERNS.items():
        features[f"client_chain_{chain_name}"] = normalized_names.map(
            lambda name, chain_pattern=pattern: bool(chain_pattern.search(name))
        )

    return features


def add_town_geocode_features(
    agencies: pd.DataFrame,
    town_col: str = TOWN_COL,
    state_col: str = STATE_COL,
) -> pd.DataFrame:
    features = agencies.copy()
    features["agency_town_clean"] = features[town_col].map(clean_town_for_geocoding)
    features["agency_state_clean"] = features[state_col].map(
        lambda value: "" if pd.isna(value) else re.sub(r"\s+", " ", str(value)).strip()
    )
    features["agency_geocode_query"] = features.apply(
        lambda row: make_geocode_query(row[town_col], row[state_col]),
        axis=1,
    )
    return features


def build_raw_regex_feature_tables(raw_dir: Path) -> dict[str, pd.DataFrame]:
    products = pd.read_csv(raw_dir / "producto_tabla.csv")
    clients = pd.read_csv(raw_dir / "cliente_tabla.csv")
    agencies = pd.read_csv(raw_dir / "town_state.csv")

    return {
        "product_regex_features": add_product_regex_features(products),
        "client_chain_features": add_client_chain_features(clients),
        "agency_town_geocode_features": add_town_geocode_features(agencies),
    }


def write_feature_tables(
    feature_tables: dict[str, pd.DataFrame],
    output_dir: Path,
    output_format: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for table_name, table in feature_tables.items():
        output_path = output_dir / f"{table_name}.{output_format}"
        if output_format == "parquet":
            table.to_parquet(output_path, index=False)
        elif output_format == "csv":
            table.to_csv(output_path, index=False)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
        print(f"Wrote {len(table):,} rows to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract regex and keyword features from raw Bimbo lookup tables."
    )
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-format", choices=["parquet", "csv"], default="parquet")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    feature_tables = build_raw_regex_feature_tables(args.raw_dir)
    write_feature_tables(feature_tables, args.output_dir, args.output_format)


if __name__ == "__main__":
    main()

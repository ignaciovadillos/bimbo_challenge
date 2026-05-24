from __future__ import annotations

import argparse
import json
import re
import time
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from geopy.extra.rate_limiter import RateLimiter
from geopy.geocoders import Nominatim

try:
    from sklearn.cluster import KMeans
except ImportError:  # pragma: no cover - depends on local environment
    KMeans = None

try:
    from src.data.raw_regex_features import clean_town_for_geocoding
except ModuleNotFoundError:  # Allows running this file directly.
    from raw_regex_features import clean_town_for_geocoding


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_PATH = PROJECT_ROOT / "data" / "raw" / "csv" / "town_state.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "interim" / "agency_geolocation_features.parquet"
DEFAULT_CACHE_PATH = PROJECT_ROOT / "data" / "interim" / "geocode_cache.json"

AGENCY_ID_COL = "Agencia_ID"
TOWN_COL = "Town"
STATE_COL = "State"
MEXICO_CITY_LAT = 19.43
MEXICO_CITY_LON = -99.13
EARTH_RADIUS_KM = 6371.0

# Approximate state centroids used only when town-level geocoding fails.
STATE_CENTROIDS = {
    "AGUASCALIENTES": (21.8853, -102.2916),
    "BAJA CALIFORNIA NORTE": (30.8406, -115.2838),
    "BAJA CALIFORNIA": (30.8406, -115.2838),
    "BAJA CALIFORNIA SUR": (26.0444, -111.6661),
    "CAMPECHE": (19.8301, -90.5349),
    "CHIAPAS": (16.7569, -93.1292),
    "CHIHUAHUA": (28.6320, -106.0691),
    "COAHUILA": (27.0587, -101.7068),
    "COLIMA": (19.2452, -103.7241),
    "DURANGO": (24.0277, -104.6532),
    "ESTADO DE MEXICO": (19.4969, -99.7233),
    "GUANAJUATO": (21.0190, -101.2574),
    "GUERRERO": (17.4392, -99.5451),
    "HIDALGO": (20.0911, -98.7624),
    "JALISCO": (20.6597, -103.3496),
    "MEXICO DF": (19.4326, -99.1332),
    "MEXICO D F": (19.4326, -99.1332),
    "MICHOACAN": (19.5665, -101.7068),
    "MORELOS": (18.6813, -99.1013),
    "NAYARIT": (21.7514, -104.8455),
    "NUEVO LEON": (25.5922, -99.9962),
    "OAXACA": (17.0732, -96.7266),
    "PUEBLA": (19.0414, -98.2063),
    "QUERETARO": (20.5888, -100.3899),
    "QUERETARO DE ARTEAGA": (20.5888, -100.3899),
    "QUINTANA ROO": (19.1817, -88.4791),
    "SAN LUIS POTOSI": (22.1565, -100.9855),
    "SINALOA": (25.1721, -107.4795),
    "SONORA": (29.2972, -110.3309),
    "TABASCO": (17.8409, -92.6189),
    "TAMAULIPAS": (23.7369, -99.1411),
    "TLAXCALA": (19.3182, -98.2375),
    "VERACRUZ": (19.1738, -96.1342),
    "YUCATAN": (20.7099, -89.0943),
    "ZACATECAS": (22.7709, -102.5832),
}


def normalize_state(value: object) -> str:
    if pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKD", str(value))
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^A-Za-z0-9]+", " ", text.upper())
    return re.sub(r"\s+", " ", text).strip()


def clean_state_for_query(value: object) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def make_geocode_query(clean_town: object, state: object) -> str:
    town = "" if pd.isna(clean_town) else str(clean_town).strip()
    clean_state = clean_state_for_query(state)
    return ", ".join(part for part in [town, clean_state, "Mexico"] if part)


def load_geocode_cache(cache_path: Path) -> dict[str, dict[str, Any]]:
    if not cache_path.exists():
        return {}
    with cache_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_geocode_cache(cache: dict[str, dict[str, Any]], cache_path: Path) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as file:
        json.dump(cache, file, indent=2, sort_keys=True, ensure_ascii=False)


def geocode_queries(
    queries: pd.Series,
    cache_path: Path,
    user_agent: str,
    min_delay_seconds: float,
    max_requests: int | None,
    skip_online: bool,
) -> dict[str, dict[str, Any]]:
    """Geocode unique queries once, respecting Nominatim's public rate limit."""
    cache = load_geocode_cache(cache_path)
    unique_queries = [query for query in queries.dropna().drop_duplicates() if query]
    missing_queries = [query for query in unique_queries if query not in cache]

    if skip_online or not missing_queries:
        return cache

    geolocator = Nominatim(user_agent=user_agent, timeout=10)
    geocode = RateLimiter(
        geolocator.geocode,
        min_delay_seconds=min_delay_seconds,
        swallow_exceptions=True,
        return_value_on_exception=None,
    )

    request_count = 0
    for query in missing_queries:
        if max_requests is not None and request_count >= max_requests:
            break

        location = geocode(query)
        request_count += 1

        if location is None:
            cache[query] = {"lat": None, "lon": None, "source": "not_found"}
        else:
            cache[query] = {
                "lat": float(location.latitude),
                "lon": float(location.longitude),
                "source": "nominatim",
                "address": location.address,
            }

        write_geocode_cache(cache, cache_path)
        print(f"Geocoded {request_count:,}/{len(missing_queries):,}: {query}")

    return cache


def apply_cached_coordinates(towns: pd.DataFrame, cache: dict[str, dict[str, Any]]) -> pd.DataFrame:
    output = towns.copy()
    cached = output["geocode_query"].map(lambda query: cache.get(query, {}))
    output["lat"] = cached.map(lambda item: item.get("lat")).astype("float64")
    output["lon"] = cached.map(lambda item: item.get("lon")).astype("float64")
    output["geocode_source"] = cached.map(lambda item: item.get("source", "missing_cache"))
    return output


def fill_missing_with_state_centroids(towns: pd.DataFrame) -> pd.DataFrame:
    output = towns.copy()
    missing_mask = output[["lat", "lon"]].isna().any(axis=1)

    for idx, row in output.loc[missing_mask].iterrows():
        centroid = STATE_CENTROIDS.get(row["state_normalized"])
        if centroid is None:
            continue

        output.at[idx, "lat"] = centroid[0]
        output.at[idx, "lon"] = centroid[1]
        output.at[idx, "geocode_source"] = "state_centroid"

    return output


def haversine_km(
    lat1: pd.Series | np.ndarray,
    lon1: pd.Series | np.ndarray,
    lat2: float,
    lon2: float,
) -> np.ndarray:
    lat1_rad = np.radians(lat1.astype(float))
    lon1_rad = np.radians(lon1.astype(float))
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    )
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def numpy_kmeans(
    coordinates: np.ndarray,
    n_clusters: int,
    random_state: int,
    max_iter: int = 100,
) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    if len(coordinates) < n_clusters:
        raise ValueError("n_clusters cannot exceed the number of coordinate rows")

    centroids = coordinates[rng.choice(len(coordinates), size=n_clusters, replace=False)]
    labels = np.full(len(coordinates), fill_value=-1, dtype=int)

    for _ in range(max_iter):
        distances = np.linalg.norm(coordinates[:, None, :] - centroids[None, :, :], axis=2)
        new_labels = distances.argmin(axis=1)

        if np.array_equal(labels, new_labels):
            break

        labels = new_labels
        for cluster_id in range(n_clusters):
            cluster_points = coordinates[labels == cluster_id]
            if len(cluster_points):
                centroids[cluster_id] = cluster_points.mean(axis=0)

    return labels


def assign_geo_clusters(
    towns: pd.DataFrame,
    n_clusters: int,
    random_state: int,
) -> pd.Series:
    valid_mask = towns[["lat", "lon"]].notna().all(axis=1)
    labels = pd.Series(pd.NA, index=towns.index, dtype="Int64")
    coordinates = towns.loc[valid_mask, ["lat", "lon"]].to_numpy(dtype=float)

    if len(coordinates) == 0:
        return labels

    if KMeans is not None:
        model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
        cluster_labels = model.fit_predict(coordinates)
    else:
        cluster_labels = numpy_kmeans(coordinates, n_clusters, random_state)

    labels.loc[valid_mask] = cluster_labels
    return labels


def add_spatial_features(
    towns: pd.DataFrame,
    n_clusters: int,
    random_state: int,
) -> pd.DataFrame:
    output = towns.copy()
    output["geo_cluster"] = assign_geo_clusters(output, n_clusters, random_state)
    output["dist_to_hub_km"] = haversine_km(
        output["lat"],
        output["lon"],
        MEXICO_CITY_LAT,
        MEXICO_CITY_LON,
    )
    output["lat_zone"] = pd.cut(
        output["lat"],
        bins=[0, 20, 23, 90],
        labels=["south", "central", "north"],
    )
    return output


def prepare_town_geolocation_frame(towns: pd.DataFrame) -> pd.DataFrame:
    output = towns.copy()
    output["town_clean"] = output[TOWN_COL].map(clean_town_for_geocoding)
    output["state_clean"] = output[STATE_COL].map(clean_state_for_query)
    output["state_normalized"] = output[STATE_COL].map(normalize_state)
    output["geocode_query"] = output.apply(
        lambda row: make_geocode_query(row["town_clean"], row[STATE_COL]),
        axis=1,
    )
    return output


def prepare_geolocation_features(
    raw_path: Path,
    cache_path: Path,
    n_clusters: int,
    random_state: int,
    user_agent: str,
    min_delay_seconds: float,
    max_requests: int | None,
    skip_online: bool,
) -> pd.DataFrame:
    towns = pd.read_csv(raw_path)
    towns = prepare_town_geolocation_frame(towns)
    cache = geocode_queries(
        queries=towns["geocode_query"],
        cache_path=cache_path,
        user_agent=user_agent,
        min_delay_seconds=min_delay_seconds,
        max_requests=max_requests,
        skip_online=skip_online,
    )
    towns = apply_cached_coordinates(towns, cache)
    towns = fill_missing_with_state_centroids(towns)
    towns = add_spatial_features(towns, n_clusters, random_state)
    return towns


def write_output(data: pd.DataFrame, output_path: Path) -> None:
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
            "Geocode Bimbo agencies and create spatial features from town_state.csv. "
            "Online geocoding uses Nominatim at one request per second and caches results."
        )
    )
    parser.add_argument("--raw-path", type=Path, default=DEFAULT_RAW_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--cache-path", type=Path, default=DEFAULT_CACHE_PATH)
    parser.add_argument("--n-clusters", type=int, default=8)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--user-agent", default="bimbo_hw_geolocation")
    parser.add_argument("--min-delay-seconds", type=float, default=1.0)
    parser.add_argument(
        "--max-requests",
        type=int,
        default=None,
        help="Optional cap for online geocoding requests during test runs.",
    )
    parser.add_argument(
        "--skip-online",
        action="store_true",
        help="Use only existing cache and state-centroid fallbacks; do not call Nominatim.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started_at = time.time()
    features = prepare_geolocation_features(
        raw_path=args.raw_path,
        cache_path=args.cache_path,
        n_clusters=args.n_clusters,
        random_state=args.random_state,
        user_agent=args.user_agent,
        min_delay_seconds=args.min_delay_seconds,
        max_requests=args.max_requests,
        skip_online=args.skip_online,
    )
    write_output(features, args.output_path)

    town_geocoded = (features["geocode_source"] == "nominatim").sum()
    centroid_fallback = (features["geocode_source"] == "state_centroid").sum()
    elapsed = time.time() - started_at
    print(f"Wrote {len(features):,} agency rows to {args.output_path}")
    print(f"Town-level geocoded rows: {town_geocoded:,}")
    print(f"State-centroid fallback rows: {centroid_fallback:,}")
    print(f"Elapsed seconds: {elapsed:.1f}")


if __name__ == "__main__":
    main()

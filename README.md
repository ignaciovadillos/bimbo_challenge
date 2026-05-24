# Grupo Bimbo Inventory Demand: Feature Engineering and Geospatial Intelligence

An end-to-end data preparation project for the Kaggle Grupo Bimbo Inventory Demand challenge. The goal is to turn messy retail transaction lookup tables into modeling-ready features that capture product attributes, client chain behavior, and agency geography.

This repository focuses on the part of machine learning that quietly decides whether the model has a chance: clean joins, leakage-aware feature engineering, and clear exploratory analysis.

## What This Project Shows

- A reproducible data pipeline for unpacking and preparing the Bimbo raw files.
- Regex-based product parsing from Spanish SKU names, including weight, units per pack, format, and brand code.
- Client chain detection from high-cardinality customer names such as `OXXO`, `WALMART`, and `7 ELEVEN`.
- Geospatial enrichment of agency locations using cleaned town/state strings, Nominatim geocoding, cached results, and state-centroid fallbacks.
- Spatial features for modeling: latitude, longitude, geocode quality flag, geographic cluster, distance to Mexico City, and latitude zone.
- Exploratory notebooks that explain what the dataset contains, where it is incomplete, and how reliable each geospatial signal is.

## Why It Matters

Grupo Bimbo delivers fresh bakery products across a large and diverse retail network. Demand depends on more than historical sales: product perishability, store type, route geography, and regional patterns all matter.

The raw Kaggle files provide useful signals, but many are hidden inside text fields:

- `NombreProducto` encodes product size, pack count, and brand.
- `NombreCliente` often reveals whether a client belongs to a modern retail chain.
- `Town` includes internal agency codes that must be stripped before geocoding.

This project extracts those signals in a structured, reusable way.

## Repository Structure

```text
.
├── README.md
├── data/
│   └── README.md                         # Data placement and pipeline guide
├── docs/
│   └── PT_HW2_Bimbo_Kaggle-1.pdf         # Assignment notes and feature hints
├── experiments/
│   ├── eda_bimbo.ipynb                   # Dataset overview and sanity checks
│   └── geospatial_exploration.ipynb      # Geospatial coverage and visualizations
├── requirements.txt
└── src/
    └── data/
        ├── 01_unzip_bimbo.py
        ├── 02_raw_regex_features.py
        ├── 03_prepare_geolocation_features.py
        └── pipeline.py
```

Raw Kaggle data and generated interim files are intentionally kept out of version control.

## Data Pipeline

Place the Kaggle archive here:

```text
data/raw/grupo-bimbo-inventory-demand.zip
```

Then run the full preparation pipeline:

```bash
python src/data/pipeline.py
```

For fast local development when the CSVs are already extracted and you do not want to call the online geocoder:

```bash
python src/data/pipeline.py --skip-unzip --skip-online-geocoding
```

The pipeline runs:

1. `01_unzip_bimbo.py`: extracts the Kaggle archive and nested CSV zips.
2. `02_raw_regex_features.py`: creates product, client-chain, and town-cleaning feature tables.
3. `03_prepare_geolocation_features.py`: geocodes agencies, applies fallbacks, and creates spatial features.

More detailed setup instructions are in [data/README.md](data/README.md).

## Generated Feature Tables

After running the pipeline, the main outputs are:

```text
data/interim/product_regex_features.parquet
data/interim/client_chain_features.parquet
data/interim/agency_town_geocode_features.parquet
data/interim/agency_geolocation_features.parquet
data/interim/geocode_cache.json
```

These tables are designed to be joined back into train/test data by `Producto_ID`, `Cliente_ID`, and `Agencia_ID`.

## Geospatial Work

The geospatial preparation is deliberately conservative:

- Town strings are cleaned before geocoding, removing prefixes like `2008 AG.`.
- Nominatim requests are rate-limited to one request per second.
- Geocoding results are cached to avoid repeated public-service calls.
- Failed or ambiguous town matches fall back to approximate Mexican state centroids.
- The output keeps `geocode_source` so downstream analysis can distinguish precise town-level matches from approximate fallbacks.

In the current prepared table:

- `790` agency rows are represented.
- `561` rows use town-level Nominatim coordinates.
- `229` rows use state-centroid fallback coordinates.

The notebook [experiments/geospatial_exploration.ipynb](experiments/geospatial_exploration.ipynb) visualizes this coverage with maps, state-level quality charts, cluster plots, and fallback examples.

## Notebooks

- [experiments/eda_bimbo.ipynb](experiments/eda_bimbo.ipynb): quick EDA covering file availability, schemas, row grain, time span, target distribution, sales/returns, and train/test coverage.
- [experiments/geospatial_exploration.ipynb](experiments/geospatial_exploration.ipynb): visual guide to what geospatial data exists, what is inferred, and which states need caution because of fallback coordinates.

## Tech Stack

- Python
- pandas and NumPy
- GeoPy and Nominatim
- matplotlib, seaborn, and Folium
- Parquet outputs for compact feature tables

## Modeling Notes

The project is built to support later demand modeling while avoiding common leakage mistakes:

- Current-week returns should not be used as predictors for the same week demand target.
- Geospatial cluster-level statistics should be computed only on the training split before being applied to validation or test data.
- Raw high-cardinality identifiers such as `Cliente_ID` should be handled carefully; chain membership and target-encoded features are safer modeling inputs.

## Reproduce From Scratch

```bash
# 1. Create and activate your environment, then install dependencies
pip install -r requirements.txt

# 2. Place the Kaggle zip at data/raw/grupo-bimbo-inventory-demand.zip

# 3. Run the data preparation pipeline
python src/data/pipeline.py

# 4. Open the notebooks
jupyter notebook experiments/eda_bimbo.ipynb
jupyter notebook experiments/geospatial_exploration.ipynb
```

If you only want to validate the local feature code without online geocoding:

```bash
python src/data/pipeline.py --skip-unzip --skip-online-geocoding
```

## Status

Completed:

- Raw archive extraction pipeline
- Product regex feature extraction
- Client chain keyword extraction
- Agency town cleaning and geocoding prep
- Geospatial feature generation
- EDA and geospatial visualization notebooks

Next modeling steps:

- Build leakage-safe lag features by week.
- Add train/validation split by `Semana`.
- Train a baseline demand model and compare feature groups.
- Evaluate RMSLE and inspect residuals by product, channel, and geography.

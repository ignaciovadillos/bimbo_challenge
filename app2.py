import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR     = PROJECT_ROOT / "data" / "raw" / "csv"
INTERIM_DIR  = PROJECT_ROOT / "data" / "interim"
MODELS_DIR   = PROJECT_ROOT / "models"

st.set_page_config(page_title="Bimbo Demand Forecaster", layout="wide")
st.title("Grupo Bimbo — Demand Forecasting Dashboard")
st.markdown("---")

# ── Data ─────────────────────────────────────────────────────────────────────

@st.cache_data
def load_data():
    train = pd.read_csv(
        DATA_DIR / "train.csv",
        dtype={"Semana": "int8", "Agencia_ID": "int32", "Canal_ID": "int8",
               "Ruta_SAK": "int32", "Cliente_ID": "int32", "Producto_ID": "int32",
               "Venta_uni_hoy": "int32", "Dev_uni_proxima": "int32",
               "Demanda_uni_equil": "int32"},
        usecols=["Semana", "Agencia_ID", "Canal_ID", "Ruta_SAK",
                 "Cliente_ID", "Producto_ID", "Venta_uni_hoy",
                 "Dev_uni_proxima", "Demanda_uni_equil"]
    )
    towns    = pd.read_parquet(INTERIM_DIR / "agency_geolocation_features.parquet")
    products = pd.read_csv(DATA_DIR / "producto_tabla.csv")

    train = train.merge(
        towns[["Agencia_ID", "lat", "lon", "geo_cluster", "dist_to_hub_km", "lat_zone"]],
        on="Agencia_ID", how="left"
    )
    train = train.merge(products, on="Producto_ID", how="left")
    train["return_rate"] = train["Dev_uni_proxima"] / (train["Venta_uni_hoy"] + 1)
    return train, towns, products


@st.cache_resource
def load_model():
    model_path = MODELS_DIR / "bimbo_improved.pkl"
    if not model_path.exists():
        return None
    with open(model_path, "rb") as f:
        return pickle.load(f)


with st.spinner("Loading data — this takes ~2 min the first time..."):
    train, towns, products = load_data()

model_bundle = load_model()

# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.header("Filters")
agency_id  = st.sidebar.selectbox("Agency",  sorted(train["Agencia_ID"].unique()))
product_id = st.sidebar.selectbox("Product", sorted(train["Producto_ID"].unique()))
canal_id   = st.sidebar.selectbox("Channel", sorted(train["Canal_ID"].unique()))

st.sidebar.markdown("---")
st.sidebar.write(f"Rows loaded: {len(train):,}")
if model_bundle:
    st.sidebar.success("Model loaded ✓")
else:
    st.sidebar.warning("Model not found — run 04_improvements.ipynb first")

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_hist, tab_map, tab_pred = st.tabs(["Demand History", "Agency Map", "Week 10 Forecast"])

# ── Tab 1: Demand History ─────────────────────────────────────────────────────

with tab_hist:
    st.subheader("Demand History & Return Rate")

    mask = (
        (train["Agencia_ID"]  == agency_id) &
        (train["Producto_ID"] == product_id) &
        (train["Canal_ID"]    == canal_id)
    )
    hist = (
        train[mask]
        .groupby(train[mask]["Semana"].astype(int))
        .agg({"Demanda_uni_equil": "sum", "return_rate": "mean"})
        .reset_index()
        .rename(columns={"Semana": "Semana"})
    )

    if len(hist) == 0:
        st.warning("No data found for this combination. Try different filters.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Net demand per week**")
            st.line_chart(hist.set_index("Semana")[["Demanda_uni_equil"]],
                          use_container_width=True)

        with col2:
            st.markdown("**Return rate per week**")
            st.line_chart(hist.set_index("Semana")[["return_rate"]],
                          use_container_width=True)

        m1, m2, m3 = st.columns(3)
        m1.metric("Avg demand",     f"{hist['Demanda_uni_equil'].mean():.0f} units")
        m2.metric("Avg return rate", f"{hist['return_rate'].mean():.1%}")
        m3.metric("Weeks with data", len(hist))

# ── Tab 2: Agency Map ─────────────────────────────────────────────────────────

with tab_map:
    st.subheader("Agency Network — colour = spatial cluster")

    import folium
    from streamlit_folium import st_folium

    towns_clean = towns.dropna(subset=["lat", "lon"])

    m = folium.Map(location=[23.6, -102.5], zoom_start=5, tiles="CartoDB positron")

    cluster_colours = {
        0: "red", 1: "blue", 2: "green", 3: "purple",
        4: "orange", 5: "darkred", 6: "cadetblue", 7: "darkgreen"
    }

    for _, row in towns_clean.iterrows():
        colour = cluster_colours.get(int(row.get("geo_cluster", 0)), "gray")
        is_selected = row["Agencia_ID"] == agency_id
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=10 if is_selected else 5,
            color="black" if is_selected else colour,
            fill=True,
            fill_color=colour,
            fill_opacity=1.0 if is_selected else 0.6,
            popup=folium.Popup(
                f"<b>Agency {row['Agencia_ID']}</b><br>"
                f"{row.get('Town', '')}<br>"
                f"Cluster: {int(row.get('geo_cluster', 0))}<br>"
                f"Dist to hub: {row.get('dist_to_hub_km', 0):.0f} km",
                max_width=200
            )
        ).add_to(m)

    st_folium(m, use_container_width=True, height=500)
    st.markdown("**Selected agency** is shown with a black border and larger circle.")

# ── Tab 3: Week 10 Forecast ───────────────────────────────────────────────────

with tab_pred:
    st.subheader("Week 10 Forecast")

    if model_bundle is None:
        st.error("No model found. Run 04_improvements.ipynb first.")
        st.stop()

    model       = model_bundle["model"]
    features    = model_bundle["features"]
    fill_values = model_bundle["fill_values"]

    combo = train[
        (train["Agencia_ID"]  == agency_id) &
        (train["Producto_ID"] == product_id) &
        (train["Canal_ID"]    == canal_id)
    ].sort_values("Semana")

    if len(combo) == 0:
        st.warning("No historical data for this combination — cannot build lag features.")
        st.stop()

    demand_by_week = combo.groupby(combo["Semana"].astype(int))["Demanda_uni_equil"].sum()

    lag1 = demand_by_week.get(9, demand_by_week.iloc[-1])
    lag2 = demand_by_week.get(8, demand_by_week.iloc[-2] if len(demand_by_week) >= 2 else lag1)
    lag3 = demand_by_week.get(7, demand_by_week.iloc[-3] if len(demand_by_week) >= 3 else lag2)

    roll2 = np.mean([lag1, lag2])
    roll3 = np.mean([lag1, lag2, lag3])

    hist_rr  = combo["Dev_uni_proxima"].sum() / (combo["Venta_uni_hoy"].sum() + 1)
    canal_rr = (
        train[train["Canal_ID"] == canal_id]["Dev_uni_proxima"].sum() /
        (train[train["Canal_ID"] == canal_id]["Venta_uni_hoy"].sum() + 1)
    )

    geo_row      = towns[towns["Agencia_ID"] == agency_id]
    geo_cluster  = int(geo_row["geo_cluster"].values[0])      if len(geo_row) else 0
    dist_to_hub  = float(geo_row["dist_to_hub_km"].values[0]) if len(geo_row) else 0.0
    zone_map     = {"south": 0, "central": 1, "north": 2}
    lat_zone_enc = zone_map.get(str(geo_row["lat_zone"].values[0]).lower(), 1) if len(geo_row) else 1

    def extract_weight_g(name):
        m = re.search(r"(\d+(?:\.\d+)?)\s*[Gg]", str(name))
        return float(m.group(1)) if m else np.nan

    def extract_units(name):
        m = re.search(r"(\d+)\s*[Pp]z", str(name))
        return float(m.group(1)) if m else 1.0

    prod_row        = products[products["Producto_ID"] == product_id]
    prod_name       = prod_row["NombreProducto"].values[0] if len(prod_row) else ""
    weight_g        = extract_weight_g(prod_name)
    units_per_pack  = extract_units(prod_name)
    weight_per_unit = weight_g / units_per_pack if not np.isnan(weight_g) else np.nan

    pc         = train[(train["Producto_ID"] == product_id) & (train["Canal_ID"] == canal_id)]["Demanda_uni_equil"]
    pc_median  = pc.median() if len(pc) > 0 else np.nan
    pc_std     = pc.std()    if len(pc) > 0 else np.nan

    ruta    = combo["Ruta_SAK"].mode().iloc[0]  if len(combo) > 0 else 0
    cliente = combo["Cliente_ID"].mode().iloc[0] if len(combo) > 0 else 0

    feature_row = {
        "Semana":            10,
        "Agencia_ID":        agency_id,
        "Canal_ID":          canal_id,
        "Ruta_SAK":          ruta,
        "Cliente_ID":        cliente,
        "Producto_ID":       product_id,
        "demand_lag1":       lag1,
        "demand_lag2":       lag2,
        "demand_lag3":       lag3,
        "demand_roll2":      roll2,
        "demand_roll3":      roll3,
        "hist_return_rate":  hist_rr,
        "canal_return_rate": canal_rr,
        "geo_cluster":       geo_cluster,
        "dist_to_hub_km":    dist_to_hub,
        "lat_zone_enc":      lat_zone_enc,
        "weight_g":          weight_g,
        "units_per_pack":    units_per_pack,
        "weight_per_unit_g": weight_per_unit,
        "pc_median_demand":  pc_median,
        "pc_std_demand":     pc_std,
    }

    X_pred = pd.DataFrame([feature_row])[features].fillna(fill_values)
    pred   = float(np.expm1(model.predict(X_pred)).clip(0)[0])

    residual_std = combo["Demanda_uni_equil"].std() if len(combo) > 1 else pred * 0.2
    ci_low  = max(0, pred - 1.96 * residual_std)
    ci_high = pred + 1.96 * residual_std

    st.markdown("### Predicted demand for Week 10")
    c1, c2, c3 = st.columns(3)
    c1.metric("Point forecast", f"{pred:.1f} units")
    c2.metric("95% CI lower",   f"{ci_low:.1f} units")
    c3.metric("95% CI upper",   f"{ci_high:.1f} units")

    st.markdown("---")

    plot_df = demand_by_week.reset_index()
    plot_df.columns = ["Semana", "Actual demand"]
    forecast_row = pd.DataFrame({"Semana": [10], "Actual demand": [np.nan]})
    plot_df = pd.concat([plot_df, forecast_row], ignore_index=True).set_index("Semana")
    plot_df["Week 10 forecast"] = np.nan
    plot_df.loc[10, "Week 10 forecast"] = pred

    st.line_chart(plot_df, use_container_width=True)
    st.caption("Historical actual demand (weeks 3–9) + Week 10 point forecast")

    with st.expander("Show feature values used for this prediction"):
        st.dataframe(
            pd.DataFrame(X_pred.T.values, index=features, columns=["Value"]),
            use_container_width=True
        )
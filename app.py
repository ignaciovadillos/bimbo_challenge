import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "csv"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"

st.set_page_config(page_title="Bimbo Demand Forecaster", layout="wide")
st.title("Grupo Bimbo — Demand Forecasting Dashboard")
st.markdown("---")

@st.cache_data
def load_data():
    train = pd.read_csv(DATA_DIR / "train.csv", dtype={"Semana": "int8"}, nrows=500000)
    towns = pd.read_parquet(INTERIM_DIR / "agency_geolocation_features.parquet")
    products = pd.read_csv(DATA_DIR / "producto_tabla.csv")
    train = train.merge(towns[["Agencia_ID", "lat", "lon", "geo_cluster"]], on="Agencia_ID", how="left")
    train = train.merge(products, on="Producto_ID", how="left")
    train["return_rate"] = train["Dev_uni_proxima"] / (train["Venta_uni_hoy"] + 1)
    return train, towns, products

with st.spinner("Loading data..."):
    train, towns, products = load_data()

# Sidebar filters
st.sidebar.header("Filters")
agency_id = st.sidebar.selectbox("Agency", sorted(train["Agencia_ID"].unique()))
product_id = st.sidebar.selectbox("Product", sorted(train["Producto_ID"].unique()))
canal_id = st.sidebar.selectbox("Channel", sorted(train["Canal_ID"].unique()))

st.sidebar.markdown("---")
st.sidebar.write(f"Total rows loaded: {len(train):,}")

# Main tabs
tab_hist, tab_map = st.tabs(["Demand History", "Agency Map"])

with tab_hist:
    st.subheader("Demand History & Return Rate")
    
    mask = (
        (train["Agencia_ID"] == agency_id) &
        (train["Producto_ID"] == product_id) &
        (train["Canal_ID"] == canal_id)
    )
    hist = train[mask].groupby("Semana")[
        ["Demanda_uni_equil", "return_rate"]
    ].mean().reset_index()
    
    if len(hist) == 0:
        st.warning("No data found for this combination. Try different filters.")
    else:
        st.line_chart(hist.set_index("Semana")[["Demanda_uni_equil"]], 
                     use_container_width=True)
        st.caption("Average net demand per week")
        
        st.line_chart(hist.set_index("Semana")[["return_rate"]], 
                     use_container_width=True)
        st.caption("Average return rate per week")

with tab_map:
    st.subheader("Agency Network")
    
    import folium
    from streamlit_folium import st_folium

    towns_clean = towns.dropna(subset=["lat", "lon"])

    m = folium.Map(location=[23.6, -102.5], zoom_start=5)

    cluster_colours = {
        0: "red", 1: "blue", 2: "green", 3: "purple",
        4: "orange", 5: "darkred", 6: "cadetblue", 7: "darkgreen"
    }

    for _, row in towns_clean.iterrows():
        colour = cluster_colours.get(int(row.get("geo_cluster", 0)), "gray")
        is_selected = row["Agencia_ID"] == agency_id
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=8 if is_selected else 5,
            color="white" if is_selected else colour,
            fill=True,
            fill_color=colour,
            fill_opacity=1.0 if is_selected else 0.6,
            popup=f"Agency {row['Agencia_ID']}: {row['Town']}"
        ).add_to(m)

    st_folium(m, use_container_width=True, height=500)
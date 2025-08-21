import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import io

# ML imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Folium imports
import folium
from streamlit_folium import st_folium

# --- Streamlit Page Config ---
st.set_page_config(page_title="Land Cover Analysis", layout="wide")

# --- Custom Styling ---
st.markdown(
    """
    <style>
    .main {background-color: #f8f9fa;}
    .stButton>button {background-color: #4CAF50; color: white; border-radius: 8px;}
    .stFileUploader {border: 2px dashed #6c757d;}
    .success-box {padding:10px; background:#d4edda; border-radius:5px; color:#155724;}
    .error-box {padding:10px; background:#f8d7da; border-radius:5px; color:#721c24;}
    </style>
    """,
    unsafe_allow_html=True
)

# --- Title ---
st.title("🌍 Land Cover Analysis Dashboard")

# --- Navigation ---
page = st.radio("Navigate", ["Data Upload", "Data Download", "Visualization", "Model Training", "Classification", "Results"], horizontal=True)

# --- 1. Data Upload ---
if page == "Data Upload":
    st.header("📂 Upload Data")

    # CSV Upload
    uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("CSV loaded successfully!")
            st.write(df.head())
        except Exception as e:
            st.error(f"Error loading CSV: {e}")

    # GeoJSON Upload
    geojson_file = st.file_uploader("Upload GeoJSON AOI", type=["geojson"])
    if geojson_file is not None:
        try:
            gdf = gpd.read_file(io.BytesIO(geojson_file.read()))

            # Ensure CRS is WGS84
            if gdf.crs is None or gdf.crs.to_string() != "EPSG:4326":
                gdf = gdf.to_crs("EPSG:4326")

            st.markdown('<div class="success-box">GeoJSON loaded successfully!</div>', unsafe_allow_html=True)
            st.write(gdf.head())

            # --- Display AOI on interactive Folium map ---
            center = [gdf.geometry.centroid.y.mean(), gdf.geometry.centroid.x.mean()]
            m = folium.Map(location=center, zoom_start=12, tiles="CartoDB positron")
            folium.GeoJson(gdf).add_to(m)

            st_folium(m, width=700, height=500)

        except Exception as e:
            st.markdown(f'<div class="error-box">Error loading GeoJSON: {e}</div>', unsafe_allow_html=True)

# --- 2. Data Download ---
elif page == "Data Download":
    st.header("⬇️ Download Processed Data")
    st.info("Feature coming soon: Export processed datasets, shapefiles, and models.")

# --- 3. Visualization ---
elif page == "Visualization":
    st.header("📊 Data Visualization")
    st.info("Upload data first to generate visualizations.")

# --- 4. Model Training ---
elif page == "Model Training":
    st.header("🤖 Model Training")
    st.info("Upload training dataset to build ML models.")

# --- 5. Classification ---
elif page == "Classification":
    st.header("🗺️ Land Cover Classification")
    st.info("Upload AOI and satellite imagery for classification.")

# --- 6. Results ---
elif page == "Results":
    st.header("📋 Results and Reports")
    st.info("View classification accuracy, confusion matrices, and export results.")

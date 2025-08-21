import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import io
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# --- Custom Dark Mode CSS ---
st.markdown("""
<style>
    /* App background */
    .stApp {
        background-color: #121212;
        color: #f1f1f1;
    }
    /* Headings */
    h1, h2, h3, h4 {
        color: #ffffff !important;
    }
    /* Buttons */
    .stButton>button {
        background-color: #1f6feb;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 8px 20px;
    }
    .stButton>button:hover {
        background-color: #1158c7;
    }
    /* Info boxes */
    .success-box {
        background-color: #1a3322;
        border-left: 4px solid #28a745;
        padding: 0.7rem;
        border-radius: 5px;
    }
    .error-box {
        background-color: #331b1b;
        border-left: 4px solid #dc3545;
        padding: 0.7rem;
        border-radius: 5px;
    }
    /* Hide default sidebar */
    section[data-testid="stSidebar"] {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)

# --- Top Navigation (using radio, no extra install needed) ---
cols = st.columns(6)
pages = ["Data Upload", "Data Download", "Visualization", "Model Training", "Classification", "Results"]
icons = ["⬆️", "⬇️", "📊", "🤖", "🗺️", "📋"]

page = None
for i, col in enumerate(cols):
    if col.button(f"{icons[i]} {pages[i]}"):
        page = pages[i]
if page is None:
    page = pages[0]  # Default page

# --- Page Content ---
if page == "Data Upload":
    st.header("⬆️ Data Upload")
    uploaded_file = st.file_uploader("Upload your CSV data", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success("Data uploaded successfully!")
        st.write(df.head())

    # GeoJSON upload with FIX
    geojson_file = st.file_uploader("Upload GeoJSON AOI", type=["geojson"])
    if geojson_file is not None:
        try:
            gdf = gpd.read_file(io.BytesIO(geojson_file.read()))

            # Ensure CRS is WGS84 for mapping
            if gdf.crs is None or gdf.crs.to_string() != "EPSG:4326":
                gdf = gdf.to_crs("EPSG:4326")

            st.markdown('<div class="success-box">GeoJSON loaded successfully!</div>', unsafe_allow_html=True)
            st.write(gdf.head())

            # Plot AOI
            st.map(gdf)
        except Exception as e:
            st.markdown(f'<div class="error-box">Error loading GeoJSON: {e}</div>', unsafe_allow_html=True)

elif page == "Data Download":
    st.header("⬇️ Data Download")
    st.info("This page will provide processed data for download.")
    # Example: Download button
    sample_data = pd.DataFrame({"col1": [1,2,3], "col2":[4,5,6]})
    csv = sample_data.to_csv(index=False).encode("utf-8")
    st.download_button("Download Sample Data", csv, "data.csv", "text/csv")

elif page == "Visualization":
    st.header("📊 Visualization")
    st.info("Visualize land cover classes and statistics here.")

    # Example visualization
    sample_data = pd.DataFrame({
        "class": ["Forest", "Water", "Urban"],
        "pixels": [1200, 800, 600]
    })
    fig, ax = plt.subplots()
    ax.bar(sample_data["class"], sample_data["pixels"])
    st.pyplot(fig)

elif page == "Model Training":
    st.header("🤖 Model Training")
    st.info("Train a land cover classification model here.")

    # Example training with dummy data
    X, y = np.random.rand(100, 5), np.random.randint(0, 2, 100)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    st.success(f"Model trained with accuracy: {clf.score(X_test, y_test):.2f}")

elif page == "Classification":
    st.header("🗺️ Classification")
    st.info("Apply trained model for classification on satellite images.")
    st.warning("Demo only – classification pipeline to be connected.")

elif page == "Results":
    st.header("📋 Results")
    st.info("View classification results and statistics here.")
    st.success("Demo complete!")

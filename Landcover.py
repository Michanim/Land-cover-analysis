import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import json
from datetime import datetime, date
import zipfile
import os
import tempfile
import plotly.express as px  # Added Plotly Express import

# ML imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.cluster import KMeans

# Folium imports
import folium
from streamlit_folium import st_folium

# Earth Engine imports
try:
    import ee
    EE_AVAILABLE = True
except ImportError:
    EE_AVAILABLE = False
    st.error("Google Earth Engine not installed. Please install with: pip install earthengine-api")

# Raster processing imports
try:
    import rasterio
    from rasterio.features import shapes
    from rasterio.transform import from_bounds
    RASTER_AVAILABLE = True
except ImportError:
    RASTER_AVAILABLE = False
    st.warning("Rasterio not available. Some features may be limited.")

# --- Initialize session state ---
if 'df' not in st.session_state:
    st.session_state.df = None
if 'gdf' not in st.session_state:
    st.session_state.gdf = None
if 'ee_image' not in st.session_state:
    st.session_state.ee_image = None
if 'trained_model' not in st.session_state:
    st.session_state.trained_model = None
if 'feature_data' not in st.session_state:
    st.session_state.feature_data = None
if 'ee_authenticated' not in st.session_state:
    st.session_state.ee_authenticated = False
if 'classified_data' not in st.session_state:
    st.session_state.classified_data = None

# --- Earth Engine Authentication ---
def authenticate_ee():
    """Authenticate Google Earth Engine with different methods"""
    try:
        ee.Initialize()
        st.session_state.ee_authenticated = True
        return True, "Initialized with existing credentials"
    except Exception as e:
        st.session_state.ee_authenticated = False
        return False, f"Authentication error: {str(e)}"

def authenticate_with_json_key(json_key_content):
    """Authenticate using JSON service account key"""
    try:
        # Parse JSON content if it's a string
        if isinstance(json_key_content, str):
            credentials_dict = json.loads(json_key_content)
        else:
            credentials_dict = json_key_content

        # Create credentials object
        from google.oauth2 import service_account
        credentials = service_account.Credentials.from_service_account_info(
            credentials_dict,
            scopes=['https://www.googleapis.com/auth/earthengine']
        )

        # Initialize Earth Engine with credentials
        ee.Initialize(credentials)
        st.session_state.ee_authenticated = True
        return True, "Successfully authenticated with JSON key"

    except Exception as e:
        st.session_state.ee_authenticated = False
        return False, f"Failed to authenticate with JSON key: {str(e)}"

def authenticate_with_token():
    """Authenticate using interactive token method"""
    try:
        ee.Authenticate()
        ee.Initialize()
        st.session_state.ee_authenticated = True
        return True, "Successfully authenticated with token"
    except Exception as e:
        st.session_state.ee_authenticated = False
        return False, f"Token authentication failed: {str(e)}"

# --- Streamlit Page Config ---
st.set_page_config(page_title="Land Cover Analysis", layout="wide")

# --- Custom Styling ---
st.markdown(
    """
    <style>
    .main {background-color: #f8f9fa;}
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 20px;
        font-weight: bold;
    }
    .stFileUploader {border: 2px dashed #6c757d; border-radius: 8px; padding: 20px;}
    .success-box {
        padding: 15px;
        background: #d4edda;
        border-radius: 8px;
        color: #155724;
        margin: 10px 0;
        border-left: 5px solid #28a745;
    }
    .error-box {
        padding: 15px;
        background: #f8d7da;
        border-radius: 8px;
        color: #721c24;
        margin: 10px 0;
        border-left: 5px solid #dc3545;
    }
    .info-box {
        padding: 15px;
        background: #d1ecf1;
        border-radius: 8px;
        color: #0c5460;
        margin: 10px 0;
        border-left: 5px solid #0c5460;
    }
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .json-key-box {
        background: #f8f9fa;
        border: 1px dashed #dee2e6;
        border-radius: 8px;
        padding: 15px;
        margin: 15px 0;
    }
    .code-block {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 5px;
        padding: 10px;
        font-family: monospace;
        white-space: pre-wrap;
        word-break: break-all;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Helper Functions ---
def calculate_ndvi(image):
    """Calculate NDVI from Sentinel-2 bands"""
    if EE_AVAILABLE:
        ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
        return image.addBands(ndvi)
    return None

def calculate_ndwi(image):
    """Calculate NDWI for water detection"""
    if EE_AVAILABLE:
        ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI')
        return image.addBands(ndwi)
    return None

def mask_clouds(image):
    """Mask clouds in Sentinel-2 imagery"""
    if EE_AVAILABLE:
        qa = image.select('QA60')
        cloud_bit_mask = 1 << 10
        cirrus_bit_mask = 1 << 11
        mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))
        return image.updateMask(mask).divide(10000)
    return None

def extract_features_from_image(image, geometry):
    """Extract spectral features from image within geometry"""
    if not EE_AVAILABLE:
        return None

    try:
        # Add spectral indices
        image = calculate_ndvi(image)
        image = calculate_ndwi(image)

        # Sample the image
        sample = image.sample(
            region=geometry,
            scale=10,
            numPixels=1000,
            geometries=True
        )

        return sample
    except Exception as e:
        st.error(f"Error extracting features: {e}")
        return None

# --- Title ---
st.title("🌍 Advanced Land Cover Analysis Dashboard")
st.markdown("*Powered by Google Earth Engine and Machine Learning*")

# --- Navigation ---
page = st.radio("Navigate", [
    "🏠 Home", "📂 Data Upload", "🛰️ Satellite Data", "📊 Visualization",
    "🤖 Model Training", "🗺️ Classification", "📋 Results", "⬇️ Downloads"
], horizontal=True)

# --- Home Page ---
if page == "🏠 Home":
    st.header("Welcome to Land Cover Analysis Dashboard")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="metric-card">
        <h3>🛰️ Satellite Data</h3>
        <p>Download Sentinel-2 imagery from Google Earth Engine with custom date ranges and AOI</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
        <h3>🤖 ML Classification</h3>
        <p>Train machine learning models to classify land cover types: Forest, Water, Urban, Agriculture</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card">
        <h3>📊 Analytics</h3>
        <p>Visualize results with interactive maps, charts, and comprehensive reports</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Getting Started")
    st.markdown("""
    1. **Upload Data**: Upload your AOI (GeoJSON) and training data (CSV)
    2. **Authenticate**: Set up Google Earth Engine authentication (JSON key recommended)
    3. **Download Satellite Data**: Get Sentinel-2 imagery for your area of interest
    4. **Train Model**: Use your training data to build classification models
    5. **Classify**: Apply the model to classify land cover types
    6. **Analyze Results**: View accuracy metrics and export results
    """)

# --- 1. Data Upload ---
elif page == "📂 Data Upload":
    st.header("📂 Upload Data")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Training Dataset (CSV)")
        uploaded_file = st.file_uploader("Upload your training dataset", type=["csv"], key="csv_upload")

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.df = df
                st.markdown('<div class="success-box">✅ CSV loaded successfully!</div>', unsafe_allow_html=True)

                st.write("**Dataset Overview:**")
                st.write(f"Shape: {df.shape}")
                st.write(df.head())

                # Show column info
                st.write("**Columns:**", list(df.columns))

                # Detect potential label columns
                potential_labels = [col for col in df.columns if 'class' in col.lower() or 'label' in col.lower() or 'type' in col.lower()]
                if potential_labels:
                    st.info(f"Potential label columns detected: {potential_labels}")

            except Exception as e:
                st.markdown(f'<div class="error-box">❌ Error loading CSV: {e}</div>', unsafe_allow_html=True)

    with col2:
        st.subheader("Area of Interest (GeoJSON)")
        geojson_file = st.file_uploader("Upload your AOI boundary", type=["geojson"], key="geojson_upload")

        if geojson_file is not None:
            try:
                gdf = gpd.read_file(io.BytesIO(geojson_file.read()))

                # Ensure CRS is WGS84
                if gdf.crs is None or gdf.crs.to_string() != "EPSG:4326":
                    gdf = gdf.to_crs("EPSG:4326")

                st.session_state.gdf = gdf
                st.markdown('<div class="success-box">✅ GeoJSON loaded successfully!</div>', unsafe_allow_html=True)

                st.write("**AOI Overview:**")
                st.write(f"Features: {len(gdf)}")
                st.write(f"Area: {gdf.geometry.area.sum():.4f} degrees²")

                # Display AOI on map
                center = [gdf.geometry.centroid.y.mean(), gdf.geometry.centroid.x.mean()]
                m = folium.Map(location=center, zoom_start=12, tiles="OpenStreetMap")

                folium.GeoJson(
                    gdf,
                    style_function=lambda feature: {
                        'fillColor': 'lightblue',
                        'color': 'blue',
                        'weight': 2,
                        'fillOpacity': 0.3,
                    }
                ).add_to(m)

                st_folium(m, width=700, height=400)

            except Exception as e:
                st.markdown(f'<div class="error-box">❌ Error loading GeoJSON: {e}</div>', unsafe_allow_html=True)

# --- 2. Satellite Data Download ---
elif page == "🛰️ Satellite Data":
    st.header("🛰️ Sentinel-2 Data Download")

    if not EE_AVAILABLE:
        st.error("Google Earth Engine is not available. Please install earthengine-api.")
        st.stop()

    # Earth Engine Authentication Section
    st.subheader("🔐 Earth Engine Authentication")

    # Check current authentication status
    if st.session_state.ee_authenticated:
        st.success("✅ Google Earth Engine is authenticated and ready to use!")
    else:
        st.warning("⚠️ Google Earth Engine is not authenticated.")

        # Authentication options
        auth_method = st.radio(
            "Choose authentication method:",
            ["JSON Service Account Key", "Interactive Token", "Manual Terminal"],
            horizontal=True
        )

        if auth_method == "JSON Service Account Key":
            st.markdown("""
            <div class="info-box">
            <p>Upload your Google Earth Engine service account JSON key file.</p>
            <p>You can obtain this from the <a href="https://console.cloud.google.com/" target="_blank">Google Cloud Console</a>.</p>
            </div>
            """, unsafe_allow_html=True)

            # Enhanced JSON key upload section
            st.markdown("**Option 1: Upload JSON Key File**")
            json_key_file = st.file_uploader(
                "Upload JSON Key File",
                type=["json"],
                help="Upload your Earth Engine service account JSON key file",
                key="json_file_uploader"
            )

            st.markdown("**Option 2: Paste JSON Key Content**")
            json_key_text = st.text_area(
                "JSON Key Content",
                placeholder='{\n  "type": "service_account",\n  "project_id": "your-project",\n  "private_key": "...",\n  ...\n}',
                height=200,
                help="Paste the entire content of your JSON key file here",
                key="json_text_area"
            )

            # Validate JSON format
            json_valid = True
            if json_key_text.strip():
                try:
                    json.loads(json_key_text.strip())
                except json.JSONDecodeError:
                    json_valid = False
                    st.error("❌ Invalid JSON format. Please check your JSON key content.")

            if st.button("🔑 Authenticate with JSON Key"):
                json_content = None

                if json_key_file is not None:
                    try:
                        json_content = json.load(json_key_file)
                    except Exception as e:
                        st.error(f"❌ Error reading JSON file: {e}")
                        st.stop()

                elif json_key_text.strip() and json_valid:
                    try:
                        json_content = json.loads(json_key_text.strip())
                    except Exception as e:
                        st.error(f"❌ Error parsing JSON: {e}")
                        st.stop()
                else:
                    st.error("Please provide JSON key content either by file upload or text input")
                    st.stop()

                if json_content:
                    # Display JSON preview for confirmation
                    st.markdown("**JSON Key Preview (first few lines):**")
                    st.markdown(f"""
                    <div class="json-key-box">
                    <div class="code-block">
                    {json.dumps(json_content, indent=2)[:500]}...
                    </div>
                    </div>
                    """, unsafe_allow_html=True)

                    if st.checkbox("I confirm this is my valid Earth Engine service account JSON key"):
                        with st.spinner("🔄 Authenticating..."):
                            success, message = authenticate_with_json_key(json_content)

                        if success:
                            st.success(f"✅ {message}")
                            st.rerun()
                        else:
                            st.error(f"❌ {message}")
                    else:
                        st.warning("Please confirm the JSON key before proceeding.")

        elif auth_method == "Interactive Token":
            st.markdown("""
            <div class="info-box">
            <p>This will open a browser window for authentication.</p>
            <p>Note: This method may not work in some deployment environments.</p>
            </div>
            """, unsafe_allow_html=True)

            if st.button("🌐 Authenticate with Token"):
                with st.spinner("🔄 Opening authentication window..."):
                    success, message = authenticate_with_token()

                if success:
                    st.success(f"✅ {message}")
                    st.rerun()
                else:
                    st.error(f"❌ {message}")

        elif auth_method == "Manual Terminal":
            st.markdown("""
            <div class="info-box">
            <p>Run the following command in your terminal:</p>
            <div class="code-block">earthengine authenticate</div>
            <p>Then restart this application.</p>
            </div>
            """, unsafe_allow_html=True)

            if st.button("🔄 Check Authentication"):
                success, message = authenticate_ee()
                if success:
                    st.success(f"✅ {message}")
                    st.rerun()
                else:
                    st.error("❌ Authentication still not working. Please try again.")

    # Only show download options if authenticated
    if not st.session_state.ee_authenticated:
        st.stop()

    if st.session_state.gdf is None:
        st.warning("⚠️ Please upload an AOI (GeoJSON) first in the Data Upload section")
        st.stop()

    st.markdown("---")
    st.subheader("📡 Download Parameters")

    col1, col2 = st.columns(2)

    with col1:
        start_date = st.date_input(
            "Start Date",
            value=date(2023, 1, 1),
            help="Start date for image collection"
        )

        cloud_cover = st.slider(
            "Maximum Cloud Cover (%)",
            min_value=0,
            max_value=100,
            value=20,
            help="Filter images by cloud cover percentage"
        )

    with col2:
        end_date = st.date_input(
            "End Date",
            value=date(2023, 12, 31),
            help="End date for image collection"
        )

        resolution = st.selectbox(
            "Spatial Resolution (m)",
            options=[10, 20, 60],
            index=0,
            help="Pixel resolution for download"
        )

    if st.button("🛰️ Download Sentinel-2 Data"):
        if not EE_AVAILABLE:
            st.error("Earth Engine not available")
            st.stop()

        try:
            with st.spinner("🔄 Processing satellite data..."):
                # Convert GeoDataFrame to Earth Engine geometry
                geom_json = json.loads(st.session_state.gdf.to_json())
                ee_geom = ee.Geometry(geom_json['features'][0]['geometry'])

                # Create image collection
                collection = ee.ImageCollection('COPERNICUS/S2_SR') \
                    .filterDate(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')) \
                    .filterBounds(ee_geom) \
                    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_cover))

                # Check if collection is empty
                size = collection.size()
                if size.getInfo() == 0:
                    st.error("No images found for the specified criteria. Try adjusting the date range or increasing the cloud cover threshold.")
                    st.stop()

                st.success(f"✅ Found {size.getInfo()} images")

                # Create median composite
                image = collection.map(mask_clouds).median().clip(ee_geom)

                # Add spectral indices
                image = calculate_ndvi(image)
                image = calculate_ndwi(image)

                # Store in session state
                st.session_state.ee_image = image

                # Extract features for training
                features = extract_features_from_image(image, ee_geom)
                if features:
                    # Convert to DataFrame
                    feature_info = features.getInfo()
                    if feature_info and 'features' in feature_info:
                        feature_data = []
                        for feature in feature_info['features']:
                            props = feature['properties']
                            # Add geometry info
                            if 'geometry' in feature and feature['geometry']['type'] == 'Point':
                                coords = feature['geometry']['coordinates']
                                props['longitude'] = coords[0]
                                props['latitude'] = coords[1]
                            feature_data.append(props)

                        feature_df = pd.DataFrame(feature_data)
                        st.session_state.feature_data = feature_df

                        st.success("✅ Features extracted successfully!")
                        st.write("**Extracted Features Preview:**")
                        st.write(feature_df.head())

                # Display image preview with improved parameters
                st.subheader("📷 Image Preview")

                try:
                    # Get image bounds for visualization
                    bounds = ee_geom.bounds().getInfo()['coordinates'][0]

                    # Create visualization parameters with adjusted values
                    vis_params = {
                        'bands': ['B4', 'B3', 'B2'],  # RGB
                        'min': 0.05,  # Increased from 0 to 0.05
                        'max': 0.3,   # Reduced from 3000 to 0.3 (normalized values)
                        'gamma': 1.4
                    }

                    # Get image URL for display with higher resolution
                    url = image.select(['B4', 'B3', 'B2']).getThumbURL({
                        'dimensions': 800,  # Increased from 512
                        'region': ee_geom,
                        'format': 'png',
                        **vis_params
                    })

                    st.image(url, caption="Sentinel-2 RGB Composite", use_container_width=True)

                    # Add debug information
                    with st.expander("🔍 Debug Information"):
                        st.write("**Image Statistics:**")
                        stats = image.reduceRegion(
                            reducer=ee.Reducer.minMax(),
                            geometry=ee_geom,
                            scale=10,
                            maxPixels=1e9
                        ).getInfo()

                        if stats:
                            st.write(f"Band Min/Max Values:")
                            for band in ['B2', 'B3', 'B4', 'B8', 'B11', 'B12']:
                                if band in stats:
                                    st.write(f"{band}: {stats[band]['min']:.4f} - {stats[band]['max']:.4f}")
                        else:
                            st.write("Could not retrieve image statistics")

                        # Check if image is empty
                        is_empty = image.reduceRegion(
                            reducer=ee.Reducer.anyNonZero(),
                            geometry=ee_geom,
                            scale=10,
                            maxPixels=1e9
                        ).getInfo()

                        st.write(f"Image contains valid pixels: {bool(is_empty.get('B4', False))}")

                        if not bool(is_empty.get('B4', False)):
                            st.warning("""
                            The image appears to be empty. This could be due to:
                            1. All pixels being masked out by cloud detection
                            2. No valid images available for the selected date range
                            3. The area of interest not being covered by Sentinel-2 data

                            Try these solutions:
                            - Increase the cloud cover threshold
                            - Adjust the date range to include more dates
                            - Check if your area of interest is valid
                            """)

                except Exception as preview_error:
                    st.error(f"Error generating image preview: {preview_error}")
                    st.info("""
                    Tips for fixing black image preview:
                    1. Try adjusting the date range to get images with less cloud cover
                    2. Check if your area of interest is valid
                    3. The image might be completely covered by clouds after masking
                    4. Try increasing the cloud cover threshold
                    """)

        except Exception as e:
            st.error(f"Error downloading satellite data: {e}")

# --- 3. Visualization ---
elif page == "📊 Visualization":
    st.header("📊 Data Visualization")

    if st.session_state.feature_data is not None:
        df = st.session_state.feature_data

        st.subheader("🔍 Feature Analysis")

        # Basic statistics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Pixels", len(df))

        with col2:
            if 'NDVI' in df.columns:
                st.metric("Avg NDVI", f"{df['NDVI'].mean():.3f}")

        with col3:
            if 'NDWI' in df.columns:
                st.metric("Avg NDWI", f"{df['NDWI'].mean():.3f}")

        with col4:
            if 'B2' in df.columns:
                st.metric("Avg Blue", f"{df['B2'].mean():.0f}")

        # Visualizations
        col1, col2 = st.columns(2)

        with col1:
            if 'NDVI' in df.columns:
                st.subheader("NDVI Distribution")
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.hist(df['NDVI'].dropna(), bins=50, alpha=0.7, color='green')
                ax.set_xlabel('NDVI')
                ax.set_ylabel('Frequency')
                ax.set_title('NDVI Distribution')
                st.pyplot(fig)

        with col2:
            if 'NDWI' in df.columns:
                st.subheader("NDWI Distribution")
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.hist(df['NDWI'].dropna(), bins=50, alpha=0.7, color='blue')
                ax.set_xlabel('NDWI')
                ax.set_ylabel('Frequency')
                ax.set_title('NDWI Distribution')
                st.pyplot(fig)

        # Scatter plot
        if 'NDVI' in df.columns and 'NDWI' in df.columns:
            st.subheader("NDVI vs NDWI Scatter Plot")
            fig, ax = plt.subplots(figsize=(10, 6))
            scatter = ax.scatter(df['NDVI'], df['NDWI'], alpha=0.5, c=df.index, cmap='viridis')
            ax.set_xlabel('NDVI')
            ax.set_ylabel('NDWI')
            ax.set_title('NDVI vs NDWI')
            plt.colorbar(scatter)
            st.pyplot(fig)

        # Correlation matrix
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            st.subheader("Feature Correlation Matrix")
            fig, ax = plt.subplots(figsize=(12, 8))
            correlation_matrix = df[numeric_cols].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
            st.pyplot(fig)

    else:
        st.info("📥 Please download satellite data first to generate visualizations.")

# --- 4. Model Training ---
elif page == "🤖 Model Training":
    st.header("🤖 Model Training")

    if st.session_state.df is None and st.session_state.feature_data is None:
        st.warning("⚠️ Please upload training data or download satellite data first.")
        st.stop()

    # Choose data source
    data_source = st.radio("Select training data source:",
                          ["Uploaded CSV", "Extracted Satellite Features"],
                          horizontal=True)

    if data_source == "Uploaded CSV" and st.session_state.df is not None:
        df = st.session_state.df.copy()
    elif data_source == "Extracted Satellite Features" and st.session_state.feature_data is not None:
        df = st.session_state.feature_data.copy()
        # Create synthetic labels based on spectral indices for demonstration
        if 'NDVI' in df.columns and 'NDWI' in df.columns:
            def classify_pixel(row):
                ndvi = row.get('NDVI', 0)
                ndwi = row.get('NDWI', 0)

                if ndwi > 0.3:
                    return 'Water'
                elif ndvi > 0.6:
                    return 'Forest'
                elif ndvi > 0.3:
                    return 'Vegetation'
                elif ndvi < 0.1:
                    return 'Urban'
                else:
                    return 'Bare_Soil'

            df['land_cover'] = df.apply(classify_pixel, axis=1)
            st.info("🏷️ Synthetic labels created based on spectral indices")
    else:
        st.error("No suitable data available for training")
        st.stop()

    st.subheader("📋 Training Configuration")

    col1, col2 = st.columns(2)

    with col1:
        # Select target column
        potential_targets = [col for col in df.columns if df[col].dtype == 'object' or 'class' in col.lower() or 'label' in col.lower()]
        target_col = st.selectbox("Select target column:", potential_targets)

        if target_col:
            st.write(f"**Classes found:** {df[target_col].unique()}")
            st.write(f"**Class distribution:**")
            st.write(df[target_col].value_counts())

    with col2:
        # Select feature columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = st.multiselect("Select feature columns:", numeric_cols, default=numeric_cols[:5])

        test_size = st.slider("Test set size (%)", 10, 40, 20) / 100
        random_state = st.number_input("Random state", value=42, min_value=0)

    if st.button("🚀 Train Model"):
        if not feature_cols or not target_col:
            st.error("Please select feature and target columns")
            st.stop()

        try:
            with st.spinner("🔄 Training model..."):
                # Prepare data
                X = df[feature_cols].fillna(0)  # Fill NaN values
                y = df[target_col]

                # Encode labels if necessary
                le = LabelEncoder()
                y_encoded = le.fit_transform(y)

                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
                )

                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                # Train Random Forest
                rf_model = RandomForestClassifier(
                    n_estimators=100,
                    random_state=random_state,
                    max_depth=10,
                    min_samples_split=5
                )
                rf_model.fit(X_train_scaled, y_train)

                # Make predictions
                y_pred = rf_model.predict(X_test_scaled)

                # Calculate accuracy
                accuracy = accuracy_score(y_test, y_pred)

                # Store model and preprocessors
                st.session_state.trained_model = {
                    'model': rf_model,
                    'scaler': scaler,
                    'label_encoder': le,
                    'feature_cols': feature_cols,
                    'accuracy': accuracy
                }

                st.success(f"✅ Model trained successfully! Accuracy: {accuracy:.3f}")

                # Display results
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("📊 Classification Report")
                    report = classification_report(y_test, y_pred,
                                                 target_names=le.classes_,
                                                 output_dict=True)
                    report_df = pd.DataFrame(report).transpose()
                    st.write(report_df.round(3))

                with col2:
                    st.subheader("🎯 Confusion Matrix")
                    cm = confusion_matrix(y_test, y_pred)
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d',
                              xticklabels=le.classes_,
                              yticklabels=le.classes_,
                              cmap='Blues', ax=ax)
                    ax.set_ylabel('True Label')
                    ax.set_xlabel('Predicted Label')
                    st.pyplot(fig)

                # Feature importance
                st.subheader("📈 Feature Importance")
                importance_df = pd.DataFrame({
                    'Feature': feature_cols,
                    'Importance': rf_model.feature_importances_
                }).sort_values('Importance', ascending=False)

                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(data=importance_df, x='Importance', y='Feature', ax=ax)
                ax.set_title('Feature Importance')
                st.pyplot(fig)

        except Exception as e:
            st.error(f"Error training model: {e}")

# --- 5. Classification ---
elif page == "🗺️ Classification":
    st.header("🗺️ Land Cover Classification")

    if st.session_state.trained_model is None:
        st.warning("⚠️ Please train a model first in the Model Training section.")
        st.stop()

    if st.session_state.ee_image is None:
        st.warning("⚠️ Please download satellite data first.")
        st.stop()

    st.subheader("🔮 Apply Classification")

    model_info = st.session_state.trained_model
    accuracy = model_info['accuracy']

    st.info(f"🎯 Using trained model with accuracy: {accuracy:.3f}")

    if st.button("🚀 Classify Land Cover"):
        try:
            with st.spinner("🔄 Classifying land cover..."):
                # Get feature data
                if st.session_state.feature_data is not None:
                    df = st.session_state.feature_data.copy()

                    # Prepare features
                    feature_cols = model_info['feature_cols']
                    X = df[feature_cols].fillna(0)

                    # Scale features
                    X_scaled = model_info['scaler'].transform(X)

                    # Make predictions
                    predictions = model_info['model'].predict(X_scaled)
                    prediction_probs = model_info['model'].predict_proba(X_scaled)

                    # Decode labels
                    predicted_labels = model_info['label_encoder'].inverse_transform(predictions)

                    # Add predictions to dataframe
                    df['predicted_class'] = predicted_labels
                    df['prediction_confidence'] = prediction_probs.max(axis=1)

                    # Display results
                    st.success("✅ Classification completed!")

                    # Class distribution
                    col1, col2 = st.columns(2)

                    with col1:
                        st.subheader("📊 Predicted Class Distribution")
                        class_counts = pd.Series(predicted_labels).value_counts()
                        st.write(class_counts)

                        # Plot distribution
                        fig, ax = plt.subplots(figsize=(10, 6))
                        class_counts.plot(kind='bar', ax=ax, color='skyblue')
                        ax.set_title('Predicted Land Cover Distribution')
                        ax.set_xlabel('Land Cover Class')
                        ax.set_ylabel('Number of Pixels')
                        plt.xticks(rotation=45)
                        st.pyplot(fig)

                    with col2:
                        st.subheader("🎯 Prediction Confidence")
                        avg_confidence = df['prediction_confidence'].mean()
                        st.metric("Average Confidence", f"{avg_confidence:.3f}")

                        # Confidence distribution
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.hist(df['prediction_confidence'], bins=30, alpha=0.7, color='orange')
                        ax.set_xlabel('Prediction Confidence')
                        ax.set_ylabel('Frequency')
                        ax.set_title('Prediction Confidence Distribution')
                        st.pyplot(fig)

                    # Store classified data
                    st.session_state.classified_data = df

                    # Show sample results
                    st.subheader("🔍 Sample Classification Results")
                    display_cols = ['predicted_class', 'prediction_confidence'] + feature_cols[:3]
                    st.write(df[display_cols].head(10))

        except Exception as e:
            st.error(f"Error during classification: {e}")

# --- 6. Results ---
elif page == "📋 Results":
    st.header("📋 Results and Analysis")

    if 'classified_data' not in st.session_state or st.session_state.classified_data is None:
        st.warning("⚠️ Please complete the classification process first.")
        st.stop()

    df = st.session_state.classified_data
# --- 7. Downloads ---
elif page == "⬇️ Downloads":
    st.header("⬇️ Download Results")

    st.markdown("""
    <div class="info-box">
    <p>Download your processed data and results in various formats.</p>
    <p>Select the data you want to download from the options below.</p>
    </div>
    """, unsafe_allow_html=True)

    # Check what data is available for download
    available_data = []
    if st.session_state.df is not None:
        available_data.append("Training Data (CSV)")
    if st.session_state.feature_data is not None:
        available_data.append("Extracted Features (CSV)")
    if 'classified_data' in st.session_state and st.session_state.classified_data is not None:
        available_data.append("Classification Results (CSV)")
    if st.session_state.gdf is not None:
        available_data.append("Area of Interest (GeoJSON)")
    if st.session_state.trained_model is not None:
        available_data.append("Trained Model (Joblib)")

    if not available_data:
        st.warning("⚠️ No data available for download. Please process some data first.")
        st.stop()

    # Create download options
    st.subheader("📥 Available Downloads")

    download_option = st.selectbox(
        "Select data to download:",
        available_data
    )

    if download_option == "Training Data (CSV)" and st.session_state.df is not None:
        st.markdown("""
        <div class="info-box">
        <p>Download your original training data as a CSV file.</p>
        </div>
        """, unsafe_allow_html=True)

        csv = st.session_state.df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Training Data CSV",
            data=csv,
            file_name="training_data.csv",
            mime="text/csv",
        )

    elif download_option == "Extracted Features (CSV)" and st.session_state.feature_data is not None:
        st.markdown("""
        <div class="info-box">
        <p>Download the features extracted from satellite imagery as a CSV file.</p>
        </div>
        """, unsafe_allow_html=True)

        csv = st.session_state.feature_data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Extracted Features CSV",
            data=csv,
            file_name="extracted_features.csv",
            mime="text/csv",
        )

    elif download_option == "Classification Results (CSV)" and 'classified_data' in st.session_state and st.session_state.classified_data is not None:
        st.markdown("""
        <div class="info-box">
        <p>Download your classification results as a CSV file.</p>
        </div>
        """, unsafe_allow_html=True)

        csv = st.session_state.classified_data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Classification Results CSV",
            data=csv,
            file_name="classification_results.csv",
            mime="text/csv",
        )

        # Also offer GeoJSON download for spatial data
        if 'longitude' in st.session_state.classified_data.columns and 'latitude' in st.session_state.classified_data.columns:
            st.markdown("""
            <div class="info-box" style="margin-top: 20px;">
            <p>Download your classification results as a GeoJSON file for GIS applications.</p>
            </div>
            """, unsafe_allow_html=True)

            # Convert to GeoDataFrame
            gdf = gpd.GeoDataFrame(
                st.session_state.classified_data,
                geometry=gpd.points_from_xy(
                    st.session_state.classified_data.longitude,
                    st.session_state.classified_data.latitude
                ),
                crs="EPSG:4326"
            )

            # Convert to GeoJSON
            geojson_str = gdf.to_json()
            geojson_bytes = geojson_str.encode('utf-8')

            st.download_button(
                label="📥 Download Classification Results GeoJSON",
                data=geojson_bytes,
                file_name="classification_results.geojson",
                mime="application/json",
            )

    elif download_option == "Area of Interest (GeoJSON)" and st.session_state.gdf is not None:
        st.markdown("""
        <div class="info-box">
        <p>Download your original area of interest as a GeoJSON file.</p>
        </div>
        """, unsafe_allow_html=True)

        geojson_str = st.session_state.gdf.to_json()
        geojson_bytes = geojson_str.encode('utf-8')

        st.download_button(
            label="📥 Download AOI GeoJSON",
            data=geojson_bytes,
            file_name="area_of_interest.geojson",
            mime="application/json",
        )

    elif download_option == "Trained Model (Joblib)" and st.session_state.trained_model is not None:
        st.markdown("""
        <div class="info-box">
        <p>Download your trained machine learning model for later use.</p>
        <p>This file can be loaded back into the application or used in other Python scripts.</p>
        </div>
        """, unsafe_allow_html=True)

        import joblib
        import io

        # Create a buffer to hold the model
        buffer = io.BytesIO()
        joblib.dump(st.session_state.trained_model, buffer)
        buffer.seek(0)

        st.download_button(
            label="📥 Download Trained Model",
            data=buffer,
            file_name="trained_model.joblib",
            mime="application/octet-stream",
        )

    # Summary statistics
    st.subheader("📊 Classification Summary")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Pixels Classified", len(df))

    with col2:
        st.metric("Number of Classes", df['predicted_class'].nunique())

    with col3:
        avg_confidence = df['prediction_confidence'].mean()
        st.metric("Average Confidence", f"{avg_confidence:.3f}")

    with col4:
        high_confidence = (df['prediction_confidence'] > 0.8).sum()
        st.metric("High Confidence Predictions", f"{high_confidence} ({100*high_confidence/len(df):.1f}%)")

    # Detailed analysis
    st.subheader("🔍 Detailed Analysis")

    tabs = st.tabs(["Class Statistics", "Spatial Distribution", "Confidence Analysis", "Feature Analysis"])

    with tabs[0]:
        # Class statistics
        class_stats = df.groupby('predicted_class').agg({
            'prediction_confidence': ['count', 'mean', 'std'],
            'NDVI': ['mean', 'std'] if 'NDVI' in df.columns else lambda x: None,
            'NDWI': ['mean', 'std'] if 'NDWI' in df.columns else lambda x: None
        }).round(3)

        st.write(class_stats)

    with tabs[1]:
        if 'longitude' in df.columns and 'latitude' in df.columns:
            st.subheader("🗺️ Spatial Distribution Map")
            try:
                fig = px.scatter_mapbox(
                    df,
                    lat='latitude',
                    lon='longitude',
                    color='predicted_class',
                    zoom=10,
                    height=600,
                    title="Predicted Land Cover Classes"
                )
                fig.update_layout(mapbox_style="open-street-map")
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating spatial distribution map: {e}")
                st.write("Make sure you have a valid Mapbox token or use another map style.")
                # Fallback to a simpler plot
                fig, ax = plt.subplots(figsize=(10, 6))
                for land_class in df['predicted_class'].unique():
                    class_df = df[df['predicted_class'] == land_class]
                    ax.scatter(class_df['longitude'], class_df['latitude'],
                              label=land_class, alpha=0.5, s=10)
                ax.set_title('Spatial Distribution of Predicted Classes')
                ax.set_xlabel('Longitude')
                ax.set_ylabel('Latitude')
                ax.legend()
                st.pyplot(fig)

    with tabs[2]:
        st.subheader("📊 Confidence Analysis")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=df, x='predicted_class', y='prediction_confidence', ax=ax)
        ax.set_title('Prediction Confidence by Class')
        ax.set_xlabel('Land Cover Class')
        ax.set_ylabel('Prediction Confidence')
        plt.xticks(rotation=45)
        st.pyplot(fig)

    with tabs[3]:
        st.subheader("🔍 Feature Analysis by Class")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols[:5]:  # Show first 5 numeric features
            if col not in ['longitude', 'latitude', 'prediction_confidence']:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.boxplot(data=df, x='predicted_class', y=col, ax=ax)
                ax.set_title(f'Distribution of {col} by Class')
                ax.set_xlabel('Land Cover Class')
                ax.set_ylabel(col)
                plt.xticks(rotation=45)
                st.pyplot(fig)


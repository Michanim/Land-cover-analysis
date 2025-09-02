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
import plotly.express as px

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
if 'new_aoi_gdf' not in st.session_state:
    st.session_state.new_aoi_gdf = None
if 'new_classification_results' not in st.session_state:
    st.session_state.new_classification_results = None

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
        background: black;
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

def calculate_additional_indices(image):
    """Calculate additional spectral indices"""
    if EE_AVAILABLE:
        # Enhanced Vegetation Index (EVI)
        evi = image.expression(
            '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))',
            {
                'NIR': image.select('B8'),
                'RED': image.select('B4'),
                'BLUE': image.select('B2')
            }
        ).rename('EVI')
        
        # Normalized Difference Built-up Index (NDBI)
        ndbi = image.normalizedDifference(['B11', 'B8']).rename('NDBI')
        
        # Bare Soil Index (BSI)
        bsi = image.expression(
            '((SWIR1 + RED) - (NIR + BLUE)) / ((SWIR1 + RED) + (NIR + BLUE))',
            {
                'SWIR1': image.select('B11'),
                'RED': image.select('B4'),
                'NIR': image.select('B8'),
                'BLUE': image.select('B2')
            }
        ).rename('BSI')
        
        return image.addBands([evi, ndbi, bsi])
    return None

def mask_clouds_sr(image):
    """Mask clouds in Sentinel-2 Surface Reflectance imagery using SCL band"""
    if EE_AVAILABLE:
        # Get the Scene Classification Layer (SCL) band
        scl = image.select('SCL')
        
        # Create masks for different conditions
        # SCL values: 0=NODATA, 1=SATURATED_DEFECTIVE, 2=DARK_AREA_PIXELS, 3=CLOUD_SHADOWS, 
        # 4=VEGETATION, 5=NOT_VEGETATED, 6=WATER, 7=UNCLASSIFIED, 8=CLOUD_MEDIUM_PROBABILITY, 
        # 9=CLOUD_HIGH_PROBABILITY, 10=THIN_CIRRUS, 11=SNOW_ICE
        
        # Keep pixels that are vegetation (4), not vegetated (5), water (6), or unclassified (7)
        # Exclude clouds (8,9), cloud shadows (3), cirrus (10), snow/ice (11), saturated (1), nodata (0)
        clear_mask = scl.eq(4).Or(scl.eq(5)).Or(scl.eq(6)).Or(scl.eq(7)).Or(scl.eq(2))
        
        # Apply mask and scale to reflectance values (SR data is already scaled)
        return image.updateMask(clear_mask)
    return None

def extract_features_from_image(image, geometry):
    """Extract spectral features from image within geometry"""
    if not EE_AVAILABLE:
        return None

    try:
        # Add spectral indices
        image = calculate_ndvi(image)
        image = calculate_ndwi(image)
        image = calculate_additional_indices(image)

        # Sample the image with more points for better coverage
        sample = image.sample(
            region=geometry,
            scale=10,
            numPixels=5000,  # Increased from 1000
            geometries=True
        )

        return sample
    except Exception as e:
        st.error(f"Error extracting features: {e}")
        return None

def classify_new_aoi(new_aoi_gdf, trained_model, start_date, end_date, cloud_cover):
    """Classify a new AOI using the trained model"""
    if not EE_AVAILABLE:
        st.error("Earth Engine not available")
        return None
    
    try:
        with st.spinner("Processing new AOI..."):
            # Convert GeoDataFrame to Earth Engine geometry
            geom_json = json.loads(new_aoi_gdf.to_json())
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
                return None

            st.success(f"Found {size.getInfo()} images")

            # Create median composite
            image = collection.map(mask_clouds_sr).median().clip(ee_geom)

            # Add spectral indices
            image = calculate_ndvi(image)
            image = calculate_ndwi(image)
            image = calculate_additional_indices(image)

            # Extract features for classification
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
                    
                    # Prepare features for prediction
                    feature_cols = trained_model['feature_cols']
                    
                    # Check if all required features are available
                    available_cols = [col for col in feature_cols if col in feature_df.columns]
                    if len(available_cols) != len(feature_cols):
                        missing_cols = set(feature_cols) - set(available_cols)
                        st.warning(f"Missing features: {missing_cols}. Using available features only.")
                        feature_cols = available_cols
                    
                    X = feature_df[feature_cols].fillna(0)
                    
                    # Scale features
                    X_scaled = trained_model['scaler'].transform(X)
                    
                    # Make predictions
                    predictions = trained_model['model'].predict(X_scaled)
                    prediction_probs = trained_model['model'].predict_proba(X_scaled)
                    
                    # Decode labels
                    predicted_labels = trained_model['label_encoder'].inverse_transform(predictions)
                    
                    # Add predictions to dataframe
                    feature_df['predicted_class'] = predicted_labels
                    feature_df['prediction_confidence'] = prediction_probs.max(axis=1)
                    
                    return feature_df
                    
        return None
    except Exception as e:
        st.error(f"Error classifying new AOI: {e}")
        return None

# --- Title ---
st.title("Advanced Land Cover Analysis Dashboard")
st.markdown("*Powered by Google Earth Engine and Machine Learning*")

# --- Navigation ---
page = st.radio("Navigate", [
    "Home", "Data Upload", "Satellite Data", "Visualization",
    "Model Training", "Classification", "Results", "Downloads"
], horizontal=True)

# --- Home Page ---
if page == "Home":
    st.header("Welcome to Land Cover Analysis Dashboard")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="metric-card">
        <h3>Satellite Data</h3>
        <p>Download Sentinel-2 imagery from Google Earth Engine with custom date ranges and AOI</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
        <h3>ML Classification</h3>
        <p>Train machine learning models to classify land cover types: Forest, Water, Urban, Agriculture</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card">
        <h3>Analytics</h3>
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
elif page == "Data Upload":
    st.header("Upload Data")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Training Dataset (CSV)")
        uploaded_file = st.file_uploader("Upload your training dataset", type=["csv"], key="csv_upload")

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.df = df
                st.markdown('<div class="success-box">CSV loaded successfully!</div>', unsafe_allow_html=True)

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
                st.markdown(f'<div class="error-box">Error loading CSV: {e}</div>', unsafe_allow_html=True)

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
                st.markdown('<div class="success-box">GeoJSON loaded successfully!</div>', unsafe_allow_html=True)

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
                st.markdown(f'<div class="error-box">Error loading GeoJSON: {e}</div>', unsafe_allow_html=True)
                
        # New AOI upload for classification (only show if model is trained)
        if st.session_state.trained_model is not None:
            st.subheader("New AOI for Classification")
            new_aoi_file = st.file_uploader("Upload a new AOI to classify", type=["geojson"], key="new_aoi_upload")
            
            if new_aoi_file is not None:
                try:
                    new_aoi_gdf = gpd.read_file(io.BytesIO(new_aoi_file.read()))

                    # Ensure CRS is WGS84
                    if new_aoi_gdf.crs is None or new_aoi_gdf.crs.to_string() != "EPSG:4326":
                        new_aoi_gdf = new_aoi_gdf.to_crs("EPSG:4326")

                    st.session_state.new_aoi_gdf = new_aoi_gdf
                    st.markdown('<div class="success-box">New AOI loaded successfully!</div>', unsafe_allow_html=True)

                    st.write("**New AOI Overview:**")
                    st.write(f"Features: {len(new_aoi_gdf)}")
                    st.write(f"Area: {new_aoi_gdf.geometry.area.sum():.4f} degrees²")

                    # Display new AOI on map
                    center = [new_aoi_gdf.geometry.centroid.y.mean(), new_aoi_gdf.geometry.centroid.x.mean()]
                    m = folium.Map(location=center, zoom_start=12, tiles="OpenStreetMap")

                    folium.GeoJson(
                        new_aoi_gdf,
                        style_function=lambda feature: {
                            'fillColor': 'lightgreen',
                            'color': 'green',
                            'weight': 2,
                            'fillOpacity': 0.3,
                        }
                    ).add_to(m)

                    st_folium(m, width=700, height=400)
                    
                    # Classification parameters for new AOI
                    st.subheader("Classification Parameters")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        start_date = st.date_input(
                            "Start Date",
                            value=date(2023, 1, 1),
                            help="Start date for image collection",
                            key="new_aoi_start_date"
                        )

                    with col2:
                        end_date = st.date_input(
                            "End Date",
                            value=date(2023, 12, 31),
                            help="End date for image collection",
                            key="new_aoi_end_date"
                        )
                    
                    cloud_cover = st.slider(
                        "Maximum Cloud Cover (%)",
                        min_value=0,
                        max_value=100,
                        value=20,
                        help="Filter images by cloud cover percentage",
                        key="new_aoi_cloud_cover"
                    )
                    
                    if st.button("Classify New AOI", key="classify_new_aoi_btn"):
                        results = classify_new_aoi(
                            new_aoi_gdf, 
                            st.session_state.trained_model,
                            start_date,
                            end_date,
                            cloud_cover
                        )
                        
                        if results is not None:
                            st.session_state.new_classification_results = results
                            st.success("New AOI classified successfully!")
                            
                            # Show classification results
                            st.subheader("Classification Results")
                            st.write(f"Total pixels classified: {len(results)}")
                            
                            # Class distribution
                            class_counts = results['predicted_class'].value_counts()
                            st.write("**Class Distribution:**")
                            st.write(class_counts)
                            
                            # Confidence statistics
                            avg_confidence = results['prediction_confidence'].mean()
                            st.write(f"**Average Confidence:** {avg_confidence:.3f}")
                            
                            # Show sample results
                            st.write("**Sample Results:**")
                            st.write(results.head())
                            
                            # Create a simple map visualization
                            if 'longitude' in results.columns and 'latitude' in results.columns:
                                st.subheader("Spatial Distribution")
                                
                                # Create a sample for faster rendering
                                sample_size = min(1000, len(results))
                                sample_df = results.sample(sample_size, random_state=42)
                                
                                fig = px.scatter_mapbox(
                                    sample_df,
                                    lat='latitude',
                                    lon='longitude',
                                    color='predicted_class',
                                    hover_data=['prediction_confidence'],
                                    zoom=10,
                                    height=500,
                                    title="Land Cover Classification"
                                )
                                fig.update_layout(mapbox_style="open-street-map")
                                st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.markdown(f'<div class="error-box">Error loading new AOI: {e}</div>', unsafe_allow_html=True)

# --- 2. Satellite Data Download ---
elif page == "Satellite Data":
    st.header("Sentinel-2 Data Download")

    if not EE_AVAILABLE:
        st.error("Google Earth Engine is not available. Please install earthengine-api.")
        st.stop()

    # Earth Engine Authentication Section
    st.subheader("Earth Engine Authentication")

    # Check current authentication status
    if st.session_state.ee_authenticated:
        st.success("Google Earth Engine is authenticated and ready to use!")
    else:
        st.warning("Google Earth Engine is not authenticated.")

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
                    st.error("Invalid JSON format. Please check your JSON key content.")

            if st.button("Authenticate with JSON Key"):
                json_content = None

                if json_key_file is not None:
                    try:
                        json_content = json.load(json_key_file)
                    except Exception as e:
                        st.error(f"Error reading JSON file: {e}")
                        st.stop()

                elif json_key_text.strip() and json_valid:
                    try:
                        json_content = json.loads(json_key_text.strip())
                    except Exception as e:
                        st.error(f"Error parsing JSON: {e}")
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
                        with st.spinner("Authenticating..."):
                            success, message = authenticate_with_json_key(json_content)

                        if success:
                            st.success(f"{message}")
                            st.rerun()
                        else:
                            st.error(f"{message}")
                    else:
                        st.warning("Please confirm the JSON key before proceeding.")

        elif auth_method == "Interactive Token":
            st.markdown("""
            <div class="info-box">
            <p>This will open a browser window for authentication.</p>
            <p>Note: This method may not work in some deployment environments.</p>
            </div>
            """, unsafe_allow_html=True)

            if st.button("Authenticate with Token"):
                with st.spinner("Opening authentication window..."):
                    success, message = authenticate_with_token()

                if success:
                    st.success(f"{message}")
                    st.rerun()
                else:
                    st.error(f"{message}")

        elif auth_method == "Manual Terminal":
            st.markdown("""
            <div class="info-box">
            <p>Run the following command in your terminal:</p>
            <div class="code-block">earthengine authenticate</div>
            <p>Then restart this application.</p>
            </div>
            """, unsafe_allow_html=True)

            if st.button("Check Authentication"):
                success, message = authenticate_ee()
                if success:
                    st.success(f"{message}")
                    st.rerun()
                else:
                    st.error("Authentication still not working. Please try again.")

    # Only show download options if authenticated
    if not st.session_state.ee_authenticated:
        st.stop()

    if st.session_state.gdf is None:
        st.warning("Please upload an AOI (GeoJSON) first in the Data Upload section")
        st.stop()

    st.markdown("---")
    st.subheader("Download Parameters")

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
            value=30,  # Increased default from 20 to 30
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

    if st.button("Download Sentinel-2 Data"):
        if not EE_AVAILABLE:
            st.error("Earth Engine not available

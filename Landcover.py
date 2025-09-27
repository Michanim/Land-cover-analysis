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

# Raster processing imports
try:
    import rasterio
    from rasterio.features import shapes
    from rasterio.transform import from_bounds
    RASTER_AVAILABLE = True
except ImportError:
    RASTER_AVAILABLE = False

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
if 'current_step' not in st.session_state:
    st.session_state.current_step = 0

# --- Modern Professional Styling ---
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styling */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
        font-family: 'Inter', sans-serif;
    }
    
    .block-container {
        padding: 2rem 1rem;
        max-width: 1200px;
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        backdrop-filter: blur(10px);
        box-shadow: 0 25px 50px rgba(0, 0, 0, 0.1);
        margin-top: 2rem;
    }
    
    /* Sidebar Styling */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #2c3e50 0%, #34495e 100%);
        border-radius: 15px;
        padding: 1rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
    }
    
    /* Navigation Steps */
    .nav-step {
        display: flex;
        align-items: center;
        padding: 12px 20px;
        margin: 8px 0;
        border-radius: 12px;
        transition: all 0.3s ease;
        cursor: pointer;
        background: rgba(255, 255, 255, 0.1);
        color: #ffffff;
        text-decoration: none;
    }
    
    .nav-step:hover {
        background: rgba(255, 255, 255, 0.2);
        transform: translateX(5px);
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
    }
    
    .nav-step.active {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    .nav-step-number {
        background: rgba(255, 255, 255, 0.2);
        border-radius: 50%;
        width: 30px;
        height: 30px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-right: 15px;
        font-weight: 600;
        font-size: 14px;
    }
    
    /* Navigation Arrows */
    .nav-arrows {
        display: flex;
        justify-content: space-between;
        margin: 3rem 0 2rem 0;
        padding: 0 2rem;
    }
    
    .nav-arrow {
        display: flex;
        align-items: center;
        padding: 15px 25px;
        border-radius: 50px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-decoration: none;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.3);
        border: none;
        cursor: pointer;
        font-size: 16px;
    }
    
    .nav-arrow:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 35px rgba(102, 126, 234, 0.4);
    }
    
    .nav-arrow.disabled {
        background: #bdc3c7;
        cursor: not-allowed;
        box-shadow: none;
    }
    
    .nav-arrow.disabled:hover {
        transform: none;
    }
    
    /* Progress Bar */
    .progress-container {
        background: #ecf0f1;
        height: 8px;
        border-radius: 10px;
        margin: 2rem 0;
        overflow: hidden;
    }
    
    .progress-bar {
        height: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        transition: width 0.5s ease;
    }
    
    /* Cards and Containers */
    .feature-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.9) 0%, rgba(255, 255, 255, 0.7) 100%);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 25px 50px rgba(0, 0, 0, 0.15);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 20px 40px rgba(102, 126, 234, 0.3);
        margin: 1rem 0;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: scale(1.05);
    }
    
    .metric-number {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
    }
    
    .metric-label {
        font-size: 1rem;
        opacity: 0.9;
        margin-top: 0.5rem;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 50px;
        padding: 12px 30px;
        font-weight: 600;
        font-size: 16px;
        transition: all 0.3s ease;
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 15px 35px rgba(102, 126, 234, 0.4);
    }
    
    /* File Uploader */
    .stFileUploader {
        border: 2px dashed #667eea;
        border-radius: 20px;
        padding: 2rem;
        background: rgba(102, 126, 234, 0.05);
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .stFileUploader:hover {
        border-color: #764ba2;
        background: rgba(118, 75, 162, 0.1);
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', sans-serif;
        color: #2c3e50;
        font-weight: 700;
    }
    
    h1 {
        font-size: 3rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        font-size: 1.2rem;
        color: #7f8c8d;
        margin-bottom: 3rem;
        text-align: center;
    }
    
    /* Status Messages */
    .success-message {
        background: linear-gradient(135deg, #00b894 0%, #00a085 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 10px 25px rgba(0, 184, 148, 0.3);
    }
    
    .error-message {
        background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 10px 25px rgba(231, 76, 60, 0.3);
    }
    
    .warning-message {
        background: linear-gradient(135deg, #f39c12 0%, #e67e22 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 10px 25px rgba(243, 156, 18, 0.3);
    }
    
    .info-message {
        background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 10px 25px rgba(52, 152, 219, 0.3);
    }
    
    /* Step Header */
    .step-header {
        text-align: center;
        margin-bottom: 3rem;
    }
    
    .step-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    
    .step-description {
        font-size: 1.1rem;
        color: #7f8c8d;
        max-width: 600px;
        margin: 0 auto;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .block-container {
            margin: 1rem;
            padding: 1rem;
        }
        
        h1 {
            font-size: 2rem;
        }
        
        .nav-arrows {
            padding: 0 1rem;
        }
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
    """Mask clouds in Sentinel-2 imagery using the SCL band."""
    if EE_AVAILABLE:
        cloud_mask = image.select('SCL').neq(3).And(
            image.select('SCL').neq(8)).And(
            image.select('SCL').neq(9)).And(
            image.select('SCL').neq(10))
        return image.updateMask(cloud_mask).divide(10000)
    return None

def authenticate_ee():
    """Authenticate Google Earth Engine"""
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
        if isinstance(json_key_content, str):
            credentials_dict = json.loads(json_key_content)
        else:
            credentials_dict = json_key_content
            
        from google.oauth2 import service_account
        credentials = service_account.Credentials.from_service_account_info(
            credentials_dict,
            scopes=['https://www.googleapis.com/auth/earthengine']
        )
        ee.Initialize(credentials)
        st.session_state.ee_authenticated = True
        return True, "Successfully authenticated with JSON key"
    except Exception as e:
        st.session_state.ee_authenticated = False
        return False, f"Failed to authenticate with JSON key: {str(e)}"

def get_step_completion_status():
    """Check which steps are completed"""
    return {
        'data_uploaded': st.session_state.df is not None or st.session_state.gdf is not None,
        'ee_authenticated': st.session_state.ee_authenticated,
        'satellite_data': st.session_state.feature_data is not None,
        'model_trained': st.session_state.trained_model is not None,
        'classification_done': st.session_state.classified_data is not None,
    }

def render_navigation_arrows(current_step, total_steps):
    """Render navigation arrows"""
    status = get_step_completion_status()
    
    # Calculate progress percentage
    progress = (current_step / (total_steps - 1)) * 100
    
    # Render progress bar
    st.markdown(f"""
    <div class="progress-container">
        <div class="progress-bar" style="width: {progress}%;"></div>
    </div>
    """, unsafe_allow_html=True)
    
    # Render navigation arrows
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if current_step > 0:
            if st.button("← Previous Step", key="prev_btn", help="Go to previous step"):
                st.session_state.current_step = current_step - 1
                st.rerun()
        else:
            st.markdown('<div class="nav-arrow disabled">← Previous Step</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"<div style='text-align: center; padding: 15px; font-weight: 600; color: #2c3e50;'>Step {current_step + 1} of {total_steps}</div>", unsafe_allow_html=True)
    
    with col3:
        # Check if current step is completed to enable next button
        can_proceed = True
        if current_step == 0:  # Home - always can proceed
            can_proceed = True
        elif current_step == 1:  # Data Upload
            can_proceed = status['data_uploaded']
        elif current_step == 2:  # Satellite Data
            can_proceed = status['satellite_data']
        elif current_step == 3:  # Visualization
            can_proceed = True  # Can view visualization if data exists
        elif current_step == 4:  # Model Training
            can_proceed = status['model_trained']
        elif current_step == 5:  # Classification
            can_proceed = status['classification_done']
        elif current_step == 6:  # Results
            can_proceed = True
        
        if current_step < total_steps - 1:
            if can_proceed:
                if st.button("Next Step →", key="next_btn", help="Go to next step"):
                    st.session_state.current_step = current_step + 1
                    st.rerun()
            else:
                st.markdown('<div class="nav-arrow disabled" title="Complete current step to proceed">Next Step →</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="nav-arrow disabled">Next Step →</div>', unsafe_allow_html=True)

# --- Page Configuration ---
st.set_page_config(
    page_title="Land Cover Analysis Platform",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Navigation Setup ---
STEPS = [
    {"title": "Welcome", "icon": "🏠", "description": "Introduction and overview"},
    {"title": "Data Upload", "icon": "📂", "description": "Upload training data and AOI"},
    {"title": "Satellite Data", "icon": "🛰️", "description": "Authenticate and download imagery"},
    {"title": "Visualization", "icon": "📊", "description": "Explore and analyze data"},
    {"title": "Model Training", "icon": "🤖", "description": "Train ML classification model"},
    {"title": "Classification", "icon": "🗺️", "description": "Apply model to classify land cover"},
    {"title": "Results", "icon": "📋", "description": "View results and analytics"},
    {"title": "Downloads", "icon": "⬇️", "description": "Export data and models"}
]

# --- Sidebar Navigation ---
st.sidebar.markdown("""
<div style="text-align: center; padding: 2rem 0;">
    <h2 style="color: #ffffff; margin-bottom: 0.5rem;">🌍 Land Cover Analysis</h2>
    <p style="color: #bdc3c7; margin-bottom: 2rem;">Advanced ML-Powered Platform</p>
</div>
""", unsafe_allow_html=True)

# Get completion status
status = get_step_completion_status()

# Render navigation steps
for i, step in enumerate(STEPS):
    # Check if step is completed
    is_completed = False
    is_active = i == st.session_state.current_step
    
    if i == 0:  # Welcome
        is_completed = True
    elif i == 1:  # Data Upload
        is_completed = status['data_uploaded']
    elif i == 2:  # Satellite Data
        is_completed = status['satellite_data']
    elif i == 3:  # Visualization
        is_completed = status['satellite_data']
    elif i == 4:  # Model Training
        is_completed = status['model_trained']
    elif i == 5:  # Classification
        is_completed = status['classification_done']
    elif i == 6:  # Results
        is_completed = status['classification_done']
    elif i == 7:  # Downloads
        is_completed = True
    
    # Create navigation item
    active_class = "active" if is_active else ""
    status_icon = "✅" if is_completed else "⏳" if i <= st.session_state.current_step else "⚪"
    
    if st.sidebar.button(f"{status_icon} {step['title']}", key=f"nav_{i}", help=step['description']):
        st.session_state.current_step = i
        st.rerun()

# --- Main Content ---
current_step = st.session_state.current_step
current_step_info = STEPS[current_step]

# Page Header
st.markdown(f"""
<div class="step-header">
    <h1>{current_step_info['icon']} {current_step_info['title']}</h1>
    <p class="subtitle">{current_step_info['description']}</p>
</div>
""", unsafe_allow_html=True)

# --- Step Content ---
if current_step == 0:  # Welcome
    st.markdown("""
    <div class="feature-card">
        <h2 style="text-align: center; margin-bottom: 2rem;">Welcome to the Future of Land Cover Analysis</h2>
        <p style="text-align: center; font-size: 1.1rem; color: #7f8c8d; margin-bottom: 3rem;">
            Harness the power of Google Earth Engine and Machine Learning to analyze and classify land cover with unprecedented accuracy and speed.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card" style="text-align: center;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">🛰️</div>
            <h3>Satellite Data Integration</h3>
            <p>Access Sentinel-2 imagery directly from Google Earth Engine with advanced cloud masking and spectral indices calculation.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card" style="text-align: center;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">🤖</div>
            <h3>Machine Learning Pipeline</h3>
            <p>Train state-of-the-art Random Forest models for accurate land cover classification with comprehensive accuracy metrics.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card" style="text-align: center;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">📊</div>
            <h3>Advanced Analytics</h3>
            <p>Interactive visualizations, spatial analysis, and professional reporting tools for comprehensive land cover insights.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Process overview
    st.markdown("""
    <div class="feature-card">
        <h3 style="text-align: center; margin-bottom: 2rem;">Analysis Workflow</h3>
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
            <div style="text-align: center; flex: 1; margin: 1rem;">
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; width: 60px; height: 60px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 0 auto 1rem; font-weight: bold; font-size: 1.2rem;">1</div>
                <h4>Upload Data</h4>
                <p>Upload your AOI and training datasets</p>
            </div>
            <div style="text-align: center; flex: 1; margin: 1rem;">
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; width: 60px; height: 60px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 0 auto 1rem; font-weight: bold; font-size: 1.2rem;">2</div>
                <h4>Fetch Imagery</h4>
                <p>Download satellite data from Earth Engine</p>
            </div>
            <div style="text-align: center; flex: 1; margin: 1rem;">
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; width: 60px; height: 60px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 0 auto 1rem; font-weight: bold; font-size: 1.2rem;">3</div>
                <h4>Train Model</h4>
                <p>Build ML classification models</p>
            </div>
            <div style="text-align: center; flex: 1; margin: 1rem;">
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; width: 60px; height: 60px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 0 auto 1rem; font-weight: bold; font-size: 1.2rem;">4</div>
                <h4>Analyze Results</h4>
                <p>View classifications and export findings</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

elif current_step == 1:  # Data Upload
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>📊 Training Dataset (CSV)</h3>
            <p>Upload your training data containing spectral features and land cover labels.</p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Choose CSV file", type=["csv"], key="csv_upload")
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.df = df
                
                st.markdown('<div class="success-message">✅ CSV loaded successfully!</div>', unsafe_allow_html=True)
                
                # Dataset metrics
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.markdown(f"""<div class="metric-card">
                        <div class="metric-number">{df.shape[0]}</div>
                        <div class="metric-label">Rows</div>
                    </div>""", unsafe_allow_html=True)
                with col_b:
                    st.markdown(f"""<div class="metric-card">
                        <div class="metric-number">{df.shape[1]}</div>
                        <div class="metric-label">Columns</div>
                    </div>""", unsafe_allow_html=True)
                with col_c:
                    st.markdown(f"""<div class="metric-card">
                        <div class="metric-number">{df.memory_usage(deep=True).sum() // 1024}KB</div>
                        <div class="metric-label">Size</div>
                    </div>""", unsafe_allow_html=True)
                
                with st.expander("📋 Dataset Preview"):
                    st.write(df.head())
                    
            except Exception as e:
                st.markdown(f'<div class="error-message">❌ Error loading CSV: {e}</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>🗺️ Area of Interest (GeoJSON)</h3>
            <p>Upload your study area boundary as a GeoJSON file.</p>
        </div>
        """, unsafe_allow_html=True)
        
        geojson_file = st.file_uploader("Choose GeoJSON file", type=["geojson"], key="geojson_upload")
        if geojson_file is not None:
            try:
                gdf = gpd.read_file(io.BytesIO(geojson_file.read()))
                if gdf.crs is None or gdf.crs.to_string() != "EPSG:4326":
                    gdf = gdf.to_crs("EPSG:4326")
                st.session_state.gdf = gdf
                
                st.markdown('<div class="success-message">✅ GeoJSON loaded successfully!</div>', unsafe_allow_html=True)
                
                # AOI metrics
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown(f"""<div class="metric-card">
                        <div class="metric-number">{len(gdf)}</div>
                        <div class="metric-label">Features</div>
                    </div>""", unsafe_allow_html=True)
                with col_b:
                    area = gdf.geometry.area.sum()
                    st.markdown(f"""<div class="metric-card">
                        <div class="metric-number">{area:.4f}</div>
                        <div class="metric-label">Area (deg²)</div>
                    </div>""", unsafe_allow_html=True)
                
                # Interactive map
                center = [gdf.geometry.centroid.y.mean(), gdf.geometry.centroid.x.mean()]
                m = folium.Map(location=center, zoom_start=12, tiles="OpenStreetMap")
                folium.GeoJson(
                    gdf,
                    style_function=lambda feature: {
                        'fillColor': '#667eea',
                        'color': '#764ba2',
                        'weight': 3,
                        'fillOpacity': 0.3,
                    }
                ).add_to(m)
                st_folium(m, width=700, height=400)
                
            except Exception as e:
                st.markdown(f'<div class="error-message">❌ Error loading GeoJSON: {e}</div>', unsafe_allow_html=True)

elif current_step == 2:  # Satellite Data
    if not EE_AVAILABLE:
        st.markdown('<div class="error-message">❌ Google Earth Engine is not available. Please install earthengine-api.</div>', unsafe_allow_html=True)
    else:
        # Authentication Status
        if st.session_state.ee_authenticated:
            st.markdown('<div class="success-message">✅ Google Earth Engine authenticated successfully!</div>', unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="feature-card">
                <h3>🔐 Earth Engine Authentication Required</h3>
                <p>Please authenticate with Google Earth Engine to access satellite imagery.</p>
            </div>
            """, unsafe_allow_html=True)
            
            auth_method = st.radio(
                "Choose authentication method:",
                ["JSON Service Account Key", "Interactive Token", "Manual Terminal"],
                horizontal=True
            )
            
            if auth_method == "JSON Service Account Key":
                st.markdown("""
                <div class="info-message">
                <p>Upload your Google Earth Engine service account JSON key file from the Google Cloud Console.</p>
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    json_key_file = st.file_uploader("Upload JSON Key File", type=["json"], key="json_file_uploader")
                with col2:
                    json_key_text = st.text_area("Or paste JSON content", placeholder='{"type": "service_account", ...}', height=150)
                
                if st.button("🔑 Authenticate with JSON Key", type="primary"):
                    json_content = None
                    if json_key_file is not None:
                        try:
                            json_content = json.load(json_key_file)
                        except Exception as e:
                            st.markdown(f'<div class="error-message">❌ Error reading JSON file: {e}</div>', unsafe_allow_html=True)
                    elif json_key_text.strip():
                        try:
                            json_content = json.loads(json_key_text.strip())
                        except Exception as e:
                            st.markdown(f'<div class="error-message">❌ Error parsing JSON: {e}</div>', unsafe_allow_html=True)
                    
                    if json_content:
                        with st.spinner("🔄 Authenticating..."):
                            success, message = authenticate_with_json_key(json_content)
                        if success:
                            st.markdown(f'<div class="success-message">✅ {message}</div>', unsafe_allow_html=True)
                            st.rerun()
                        else:
                            st.markdown(f'<div class="error-message">❌ {message}</div>', unsafe_allow_html=True)
        
        # Only show download options if authenticated and AOI is available
        if st.session_state.ee_authenticated:
            if st.session_state.gdf is None:
                st.markdown('<div class="warning-message">⚠️ Please upload an AOI (GeoJSON) first in the previous step.</div>', unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="feature-card">
                    <h3>📡 Satellite Data Configuration</h3>
                    <p>Configure parameters for Sentinel-2 imagery download from Google Earth Engine.</p>
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    start_date = st.date_input("Start Date", value=date(2023, 1, 1))
                    cloud_cover = st.slider("Maximum Cloud Cover (%)", 0, 100, 20)
                with col2:
                    end_date = st.date_input("End Date", value=date(2023, 12, 31))
                    resolution = st.selectbox("Spatial Resolution (m)", [10, 20, 60], index=0)
                
                if st.button("🛰️ Download Sentinel-2 Data", type="primary"):
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
                            
                            common_bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12', 'SCL']
                            collection = collection.map(lambda image: image.select(common_bands))
                            
                            size = collection.size()
                            if size.getInfo() == 0:
                                st.markdown('<div class="error-message">❌ No images found. Try adjusting the date range or cloud cover threshold.</div>', unsafe_allow_html=True)
                            else:
                                st.markdown(f'<div class="success-message">✅ Found {size.getInfo()} images</div>', unsafe_allow_html=True)
                                
                                # Create median composite
                                image = collection.map(mask_clouds).median().clip(ee_geom)
                                image = calculate_ndvi(image)
                                image = calculate_ndwi(image)
                                
                                st.session_state.ee_image = image
                                
                                # Extract features
                                features = image.sample(region=ee_geom, scale=10, numPixels=1000, geometries=True)
                                feature_info = features.getInfo()
                                
                                if feature_info and 'features' in feature_info:
                                    feature_data = []
                                    for feature in feature_info['features']:
                                        props = feature['properties']
                                        if 'geometry' in feature and feature['geometry']['type'] == 'Point':
                                            coords = feature['geometry']['coordinates']
                                            props['longitude'] = coords[0]
                                            props['latitude'] = coords[1]
                                        feature_data.append(props)
                                    
                                    feature_df = pd.DataFrame(feature_data)
                                    st.session_state.feature_data = feature_df
                                    
                                    # Display success metrics
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.markdown(f"""<div class="metric-card">
                                            <div class="metric-number">{len(feature_df)}</div>
                                            <div class="metric-label">Extracted Points</div>
                                        </div>""", unsafe_allow_html=True)
                                    with col2:
                                        if 'NDVI' in feature_df.columns:
                                            avg_ndvi = feature_df['NDVI'].mean()
                                            st.markdown(f"""<div class="metric-card">
                                                <div class="metric-number">{avg_ndvi:.3f}</div>
                                                <div class="metric-label">Avg NDVI</div>
                                            </div>""", unsafe_allow_html=True)
                                    with col3:
                                        if 'NDWI' in feature_df.columns:
                                            avg_ndwi = feature_df['NDWI'].mean()
                                            st.markdown(f"""<div class="metric-card">
                                                <div class="metric-number">{avg_ndwi:.3f}</div>
                                                <div class="metric-label">Avg NDWI</div>
                                            </div>""", unsafe_allow_html=True)
                                    
                                    # Image preview
                                    st.markdown("""
                                    <div class="feature-card">
                                        <h3>📷 Image Preview</h3>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                    vis_params = {'bands': ['B4', 'B3', 'B2'], 'min': 0.05, 'max': 0.3, 'gamma': 1.4}
                                    url = image.select(['B4', 'B3', 'B2']).getThumbURL({
                                        'dimensions': 800,
                                        'region': ee_geom,
                                        'format': 'png',
                                        **vis_params
                                    })
                                    st.image(url, caption="Sentinel-2 RGB Composite", use_container_width=True)
                                    
                    except Exception as e:
                        st.markdown(f'<div class="error-message">❌ Error downloading satellite data: {e}</div>', unsafe_allow_html=True)

elif current_step == 3:  # Visualization
    if st.session_state.feature_data is not None:
        df = st.session_state.feature_data
        
        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-number">{len(df)}</div>
                <div class="metric-label">Total Pixels</div>
            </div>""", unsafe_allow_html=True)
        with col2:
            if 'NDVI' in df.columns:
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-number">{df['NDVI'].mean():.3f}</div>
                    <div class="metric-label">Avg NDVI</div>
                </div>""", unsafe_allow_html=True)
        with col3:
            if 'NDWI' in df.columns:
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-number">{df['NDWI'].mean():.3f}</div>
                    <div class="metric-label">Avg NDWI</div>
                </div>""", unsafe_allow_html=True)
        with col4:
            if 'B2' in df.columns:
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-number">{df['B2'].mean():.3f}</div>
                    <div class="metric-label">Avg Blue Band</div>
                </div>""", unsafe_allow_html=True)
        
        # Visualization tabs
        tab1, tab2, tab3 = st.tabs(["📊 Distributions", "🔗 Correlations", "🗺️ Spatial Analysis"])
        
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                if 'NDVI' in df.columns:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.hist(df['NDVI'].dropna(), bins=50, alpha=0.7, color='#667eea')
                    ax.set_xlabel('NDVI')
                    ax.set_ylabel('Frequency')
                    ax.set_title('NDVI Distribution')
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
            with col2:
                if 'NDWI' in df.columns:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.hist(df['NDWI'].dropna(), bins=50, alpha=0.7, color='#764ba2')
                    ax.set_xlabel('NDWI')
                    ax.set_ylabel('Frequency')
                    ax.set_title('NDWI Distribution')
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
        
        with tab2:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                fig, ax = plt.subplots(figsize=(12, 8))
                correlation_matrix = df[numeric_cols].corr()
                sns.heatmap(correlation_matrix, annot=True, cmap='RdYlBu_r', center=0, ax=ax)
                ax.set_title('Feature Correlation Matrix')
                st.pyplot(fig)
        
        with tab3:
            if 'longitude' in df.columns and 'latitude' in df.columns:
                if 'NDVI' in df.columns:
                    fig = px.scatter_mapbox(
                        df.sample(min(1000, len(df))),
                        lat='latitude',
                        lon='longitude',
                        color='NDVI',
                        zoom=10,
                        height=600,
                        title="NDVI Spatial Distribution",
                        color_continuous_scale='RdYlGn'
                    )
                    fig.update_layout(mapbox_style="open-street-map")
                    st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.markdown('<div class="info-message">📥 Please download satellite data first to generate visualizations.</div>', unsafe_allow_html=True)

elif current_step == 4:  # Model Training
    if st.session_state.df is None and st.session_state.feature_data is None:
        st.markdown('<div class="warning-message">⚠️ Please upload training data or download satellite data first.</div>', unsafe_allow_html=True)
    else:
        # Data source selection
        data_source = st.radio("Select training data source:", ["Uploaded CSV", "Extracted Satellite Features"], horizontal=True)
        
        if data_source == "Uploaded CSV" and st.session_state.df is not None:
            df = st.session_state.df.copy()
        elif data_source == "Extracted Satellite Features" and st.session_state.feature_data is not None:
            df = st.session_state.feature_data.copy()
            # Create synthetic labels
            if 'NDVI' in df.columns and 'NDWI' in df.columns:
                def classify_pixel(row):
                    ndvi, ndwi = row.get('NDVI', 0), row.get('NDWI', 0)
                    if ndwi > 0.3: return 'Water'
                    elif ndvi > 0.6: return 'Forest'
                    elif ndvi > 0.3: return 'Vegetation'
                    elif ndvi < 0.1: return 'Urban'
                    else: return 'Bare_Soil'
                df['land_cover'] = df.apply(classify_pixel, axis=1)
                st.markdown('<div class="info-message">🏷️ Synthetic labels created based on spectral indices</div>', unsafe_allow_html=True)
        
        # Training configuration
        st.markdown("""
        <div class="feature-card">
            <h3>🎯 Model Configuration</h3>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            potential_targets = [col for col in df.columns if df[col].dtype == 'object' or 'class' in col.lower()]
            target_col = st.selectbox("Target column:", potential_targets)
            if target_col:
                st.info(f"Classes: {list(df[target_col].unique())}")
        
        with col2:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            feature_cols = st.multiselect("Feature columns:", numeric_cols, default=numeric_cols[:5])
            test_size = st.slider("Test set size (%)", 10, 40, 20) / 100
        
        if st.button("🚀 Train Model", type="primary"):
            if not feature_cols or not target_col:
                st.markdown('<div class="error-message">❌ Please select feature and target columns</div>', unsafe_allow_html=True)
            else:
                try:
                    with st.spinner("🔄 Training model..."):
                        # Prepare data
                        X = df[feature_cols].fillna(0)
                        y = df[target_col]
                        
                        # Encode labels
                        le = LabelEncoder()
                        y_encoded = le.fit_transform(y)
                        
                        # Split data
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
                        )
                        
                        # Scale features
                        scaler = StandardScaler()
                        X_train_scaled = scaler.fit_transform(X_train)
                        X_test_scaled = scaler.transform(X_test)
                        
                        # Train model
                        rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
                        rf_model.fit(X_train_scaled, y_train)
                        
                        # Predictions
                        y_pred = rf_model.predict(X_test_scaled)
                        accuracy = accuracy_score(y_test, y_pred)
                        
                        # Store model
                        st.session_state.trained_model = {
                            'model': rf_model,
                            'scaler': scaler,
                            'label_encoder': le,
                            'feature_cols': feature_cols,
                            'accuracy': accuracy
                        }
                        
                        st.markdown(f'<div class="success-message">✅ Model trained successfully! Accuracy: {accuracy:.3f}</div>', unsafe_allow_html=True)
                        
                        # Results display
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("""
                            <div class="feature-card">
                                <h3>📊 Classification Report</h3>
                            </div>
                            """, unsafe_allow_html=True)
                            report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
                            report_df = pd.DataFrame(report).transpose()
                            st.dataframe(report_df.round(3))
                        
                        with col2:
                            st.markdown("""
                            <div class="feature-card">
                                <h3>🎯 Confusion Matrix</h3>
                            </div>
                            """, unsafe_allow_html=True)
                            cm = confusion_matrix(y_test, y_pred)
                            fig, ax = plt.subplots(figsize=(8, 6))
                            sns.heatmap(cm, annot=True, fmt='d', xticklabels=le.classes_, 
                                       yticklabels=le.classes_, cmap='Blues', ax=ax)
                            ax.set_ylabel('True Label')
                            ax.set_xlabel('Predicted Label')
                            st.pyplot(fig)
                        
                        # Feature importance
                        st.markdown("""
                        <div class="feature-card">
                            <h3>📈 Feature Importance</h3>
                        </div>
                        """, unsafe_allow_html=True)
                        importance_df = pd.DataFrame({
                            'Feature': feature_cols,
                            'Importance': rf_model.feature_importances_
                        }).sort_values('Importance', ascending=False)
                        
                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.barplot(data=importance_df, x='Importance', y='Feature', ax=ax, palette='viridis')
                        ax.set_title('Feature Importance')
                        st.pyplot(fig)
                        
                except Exception as e:
                    st.markdown(f'<div class="error-message">❌ Error training model: {e}</div>', unsafe_allow_html=True)

elif current_step == 5:  # Classification
    if st.session_state.trained_model is None:
        st.markdown('<div class="warning-message">⚠️ Please train a model first.</div>', unsafe_allow_html=True)
    elif st.session_state.feature_data is None:
        st.markdown('<div class="warning-message">⚠️ Please download satellite data first.</div>', unsafe_allow_html=True)
    else:
        model_info = st.session_state.trained_model
        accuracy = model_info['accuracy']
        
        st.markdown(f"""
        <div class="feature-card">
            <h3>🔮 Land Cover Classification</h3>
            <p>Apply trained model with accuracy: <strong>{accuracy:.3f}</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚀 Classify Land Cover", type="primary"):
            try:
                with st.spinner("🔄 Classifying land cover..."):
                    df = st.session_state.feature_data.copy()
                    
                    # Prepare features
                    feature_cols = model_info['feature_cols']
                    X = df[feature_cols].fillna(0)
                    
                    # Scale and predict
                    X_scaled = model_info['scaler'].transform(X)
                    predictions = model_info['model'].predict(X_scaled)
                    prediction_probs = model_info['model'].predict_proba(X_scaled)
                    
                    # Decode labels
                    predicted_labels = model_info['label_encoder'].inverse_transform(predictions)
                    
                    # Add predictions
                    df['predicted_class'] = predicted_labels
                    df['prediction_confidence'] = prediction_probs.max(axis=1)
                    
                    st.session_state.classified_data = df
                    
                    st.markdown('<div class="success-message">✅ Classification completed!</div>', unsafe_allow_html=True)
                    
                    # Results metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        total_pixels = len(df)
                        st.markdown(f"""<div class="metric-card">
                            <div class="metric-number">{total_pixels}</div>
                            <div class="metric-label">Pixels Classified</div>
                        </div>""", unsafe_allow_html=True)
                    with col2:
                        num_classes = df['predicted_class'].nunique()
                        st.markdown(f"""<div class="metric-card">
                            <div class="metric-number">{num_classes}</div>
                            <div class="metric-label">Land Cover Classes</div>
                        </div>""", unsafe_allow_html=True)
                    with col3:
                        avg_confidence = df['prediction_confidence'].mean()
                        st.markdown(f"""<div class="metric-card">
                            <div class="metric-number">{avg_confidence:.3f}</div>
                            <div class="metric-label">Avg Confidence</div>
                        </div>""", unsafe_allow_html=True)
                    
                    # Class distribution
                    st.markdown("""
                    <div class="feature-card">
                        <h3>📊 Land Cover Distribution</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    class_counts = df['predicted_class'].value_counts()
                    fig, ax = plt.subplots(figsize=(10, 6))
                    colors = plt.cm.Set3(np.linspace(0, 1, len(class_counts)))
                    class_counts.plot(kind='bar', ax=ax, color=colors)
                    ax.set_title('Predicted Land Cover Distribution')
                    ax.set_xlabel('Land Cover Class')
                    ax.set_ylabel('Number of Pixels')
                    plt.xticks(rotation=45)
                    st.pyplot(fig)
                    
            except Exception as e:
                st.markdown(f'<div class="error-message">❌ Error during classification: {e}</div>', unsafe_allow_html=True)

elif current_step == 6:  # Results
    if st.session_state.classified_data is None:
        st.markdown('<div class="warning-message">⚠️ Please complete the classification process first.</div>', unsafe_allow_html=True)
    else:
        df = st.session_state.classified_data
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-number">{len(df)}</div>
                <div class="metric-label">Total Pixels</div>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-number">{df['predicted_class'].nunique()}</div>
                <div class="metric-label">Classes Found</div>
            </div>""", unsafe_allow_html=True)
        with col3:
            avg_confidence = df['prediction_confidence'].mean()
            st.markdown(f"""<div class="metric-card">
                <div class="metric-number">{avg_confidence:.3f}</div>
                <div class="metric-label">Avg Confidence</div>
            </div>""", unsafe_allow_html=True)
        with col4:
            high_confidence = (df['prediction_confidence'] > 0.8).sum()
            pct = 100 * high_confidence / len(df)
            st.markdown(f"""<div class="metric-card">
                <div class="metric-number">{pct:.1f}%</div>
                <div class="metric-label">High Confidence</div>
            </div>""", unsafe_allow_html=True)
        
        # Detailed analysis tabs
        tab1, tab2, tab3 = st.tabs(["📊 Statistics", "🗺️ Spatial View", "📈 Analysis"])
        
        with tab1:
            class_stats = df.groupby('predicted_class').agg({
                'prediction_confidence': ['count', 'mean', 'std'],
                'NDVI': ['mean', 'std'] if 'NDVI' in df.columns else lambda x: None,
                'NDWI': ['mean', 'std'] if 'NDWI' in df.columns else lambda x: None
            }).round(3)
            st.dataframe(class_stats)
        
        with tab2:
            if 'longitude' in df.columns and 'latitude' in df.columns:
                fig = px.scatter_mapbox(
                    df.sample(min(1000, len(df))),
                    lat='latitude',
                    lon='longitude',
                    color='predicted_class',
                    zoom=10,
                    height=600,
                    title="Land Cover Classification Results"
                )
                fig.update_layout(mapbox_style="open-street-map")
                st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(data=df, x='predicted_class', y='prediction_confidence', ax=ax)
            ax.set_title('Prediction Confidence by Class')
            ax.set_xlabel('Land Cover Class')
            ax.set_ylabel('Prediction Confidence')
            plt.xticks(rotation=45)
            st.pyplot(fig)

elif current_step == 7:  # Downloads
    st.markdown("""
    <div class="feature-card">
        <h3>⬇️ Export Your Results</h3>
        <p>Download processed data, trained models, and analysis results in various formats.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check available data
    available_data = []
    if st.session_state.df is not None:
        available_data.append("Training Data (CSV)")
    if st.session_state.feature_data is not None:
        available_data.append("Extracted Features (CSV)")
    if st.session_state.classified_data is not None:
        available_data.append("Classification Results (CSV)")
    if st.session_state.gdf is not None:
        available_data.append("Area of Interest (GeoJSON)")
    if st.session_state.trained_model is not None:
        available_data.append("Trained Model (Joblib)")
    
    if not available_data:
        st.markdown('<div class="warning-message">⚠️ No data available for download. Please process some data first.</div>', unsafe_allow_html=True)
    else:
        download_option = st.selectbox("Select data to download:", available_data)
        
        if download_option == "Classification Results (CSV)" and st.session_state.classified_data is not None:
            csv = st.session_state.classified_data.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Classification Results",
                data=csv,
                file_name="classification_results.csv",
                mime="text/csv",
                type="primary"
            )
        
        # Add other download options as needed...

# Navigation arrows
render_navigation_arrows(current_step, len(STEPS))

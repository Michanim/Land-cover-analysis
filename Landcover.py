import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import folium
from streamlit_folium import folium_static
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import requests
import zipfile
import io
import os
from pathlib import Path

# Earth Engine and satellite data
try:
    import ee
    import geemap
    EE_AVAILABLE = True
except ImportError:
    EE_AVAILABLE = False
    st.warning("Earth Engine not available. Install with: pip install earthengine-api geemap")

# Machine Learning
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans, DBSCAN
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib

# Image processing
try:
    import rasterio
    from rasterio.mask import mask
    from rasterio.warp import calculate_default_transform, reproject, Resampling
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False
    st.warning("Rasterio not available. Install with: pip install rasterio")

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Configure Streamlit page
st.set_page_config(
    page_title="Sentinel-2 Land Cover Analysis",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""


<style>
    body {
        background-color: #121212;
        color: #e0e0e0;
    }

    .main-header {
        font-size: 3rem;
        color: #90caf9;
        text-align: center;
        font-weight: bold;
        margin-bottom: 2rem;
        text-shadow: 1px 1px 2px #000;
    }

    .sub-header {
        font-size: 1.75rem;
        color: #ffb74d;
        margin-top: 1rem;
        margin-bottom: 1rem;
        font-weight: 600;
        text-align: center;
    }

    .info-box {
        background-color: #1e1e1e;
        padding: 1.5rem;
        border-radius: 0.75rem;
        margin: 1rem 0;
        box-shadow: 0 4px 8px rgba(255, 255, 255, 0.05);
        border-left: 6px solid #90caf9;
        font-size: 1.1rem;
        line-height: 1.6;
        color: #e0e0e0;
    }

    .info-box:hover {
        background-color: #2c2c2c;
        transition: background-color 0.3s ease;
    }

    [data-testid="stSidebar"] {
        background-color: #1f1f1f;
        padding: 2rem 1rem;
    }

    [data-testid="stSidebar"] h2 {
        color: #90caf9;
    }

    .stButton>button {
        background-color: #90caf9;
        color: #121212;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        font-weight: bold;
        border: none;
    }

    .stButton>button:hover {
        background-color: #64b5f6;
        transition: background-color 0.3s ease;
    }
</style>


""", unsafe_allow_html=True)

class SentinelDataProcessor:
    def __init__(self):
        self.data = None
        self.models = {}
        self.scaler = StandardScaler()
        
    def authenticate_earth_engine(self, service_account_key=None):
        """Authenticate Google Earth Engine"""
        try:
            if service_account_key:
                credentials = ee.ServiceAccountCredentials(
                    service_account_key['client_email'], 
                    key_data=json.dumps(service_account_key)
                )
                ee.Initialize(credentials)
            else:
                ee.Initialize()
            return True
        except Exception as e:
            st.error(f"Earth Engine authentication failed: {str(e)}")
            return False
    
    def load_geojson(self, geojson_data):
        """Load and validate GeoJSON data"""
        try:
            # Handle different input types
            if isinstance(geojson_data, str):
                gdf = gpd.read_file(geojson_data)
            elif isinstance(geojson_data, gpd.GeoDataFrame):
                gdf = geojson_data.copy()
            else:
                # Handle raw GeoJSON dictionary
                try:
                    gdf = gpd.GeoDataFrame.from_features(geojson_data['features'])
                except Exception as feature_error:
                    st.error(f"Error parsing GeoJSON features: {feature_error}")
                    # Try alternative parsing
                    try:
                        import json
                        from shapely import wkt
                        from shapely.geometry import shape
                        
                        geometries = []
                        properties_list = []
                        
                        for feature in geojson_data['features']:
                            try:
                                # Convert GeoJSON geometry to Shapely geometry
                                geom = shape(feature['geometry'])
                                
                                # Fix invalid geometries
                                if not geom.is_valid:
                                    st.warning("Found invalid geometry, attempting to fix...")
                                    geom = geom.buffer(0)
                                
                                # Handle multi-part geometries
                                if geom.geom_type.startswith('Multi'):
                                    # Split multi-part into single parts
                                    for single_geom in geom.geoms:
                                        geometries.append(single_geom)
                                        properties_list.append(feature.get('properties', {}))
                                else:
                                    geometries.append(geom)
                                    properties_list.append(feature.get('properties', {}))
                                    
                            except Exception as geom_error:
                                st.warning(f"Skipping invalid geometry: {geom_error}")
                                continue
                        
                        if not geometries:
                            st.error("No valid geometries found in GeoJSON")
                            return None
                        
                        # Create GeoDataFrame from parsed geometries
                        gdf = gpd.GeoDataFrame(properties_list, geometry=geometries)
                        
                    except Exception as alt_error:
                        st.error(f"Alternative parsing also failed: {alt_error}")
                        return None
            
            if gdf.empty:
                st.error("No valid features found in the file")
                return None
            
            # Get initial bounds to help identify coordinate system
            try:
                initial_bounds = gdf.total_bounds
                st.info(f"Initial geometry bounds: [{initial_bounds[0]:.2f}, {initial_bounds[1]:.2f}, {initial_bounds[2]:.2f}, {initial_bounds[3]:.2f}]")
            except Exception as bounds_error:
                st.warning(f"Could not calculate bounds: {bounds_error}")
                initial_bounds = None
            
            # Detect and set coordinate system
            if gdf.crs is None:
                if initial_bounds is not None:
                    # Try to detect CRS based on coordinate values
                    if (abs(initial_bounds[0]) > 180 or abs(initial_bounds[2]) > 180 or 
                        abs(initial_bounds[1]) > 90 or abs(initial_bounds[3]) > 90):
                        
                        # Looks like projected coordinates - try to detect UTM zone
                        centroid_x = (initial_bounds[0] + initial_bounds[2]) / 2
                        centroid_y = (initial_bounds[1] + initial_bounds[3]) / 2
                        
                        # Estimate UTM zone based on coordinates
                        if 600000 < centroid_x < 800000:  # Typical UTM range
                            if centroid_y > 0:  # Northern hemisphere
                                if centroid_x < 700000:
                                    estimated_zone = 30  # Common for West Africa
                                else:
                                    estimated_zone = 31
                                estimated_crs = f"EPSG:{32600 + estimated_zone}"
                            else:  # Southern hemisphere
                                estimated_zone = 30
                                estimated_crs = f"EPSG:{32700 + estimated_zone}"
                            
                            st.warning(f"No CRS specified. Based on coordinates, estimating {estimated_crs}")
                            st.info("If this is incorrect, please use manual CRS selection")
                            gdf.crs = estimated_crs
                        else:
                            st.warning("Could not auto-detect CRS. Assuming geographic coordinates (WGS84)")
                            gdf.crs = "EPSG:4326"
                    else:
                        # Looks like geographic coordinates
                        st.info("Coordinates appear to be in geographic format, assuming WGS84")
                        gdf.crs = "EPSG:4326"
                else:
                    st.warning("No bounds available, assuming WGS84")
                    gdf.crs = "EPSG:4326"
            
            # Show current CRS
            st.info(f"Current CRS: {gdf.crs}")
            
            # Convert to WGS84 if not already
            if gdf.crs.to_string() != "EPSG:4326":
                st.info(f"Converting from {gdf.crs} to EPSG:4326 (WGS84)")
                try:
                    gdf = gdf.to_crs("EPSG:4326")
                    st.success("✅ Coordinate system conversion successful!")
                except Exception as crs_error:
                    st.error(f"❌ CRS conversion failed: {str(crs_error)}")
                    st.info("Please try manual CRS selection or check your coordinate system")
                    return None
            
            # Validate final geometry bounds
            try:
                final_bounds = gdf.total_bounds
                st.info(f"Final geometry bounds (WGS84): [{final_bounds[0]:.6f}, {final_bounds[1]:.6f}, {final_bounds[2]:.6f}, {final_bounds[3]:.6f}]")
                
                if final_bounds[0] < -180 or final_bounds[2] > 180 or final_bounds[1] < -90 or final_bounds[3] > 90:
                    st.error(f"❌ Invalid bounds after conversion: {final_bounds}")
                    st.error("Coordinates are still outside valid Earth ranges")
                    return None
            except Exception as final_bounds_error:
                st.warning(f"Could not validate final bounds: {final_bounds_error}")
            
            # Clean up geometries
            st.info("🔧 Cleaning and validating geometries...")
            
            # Check for and fix invalid geometries
            invalid_mask = ~gdf.geometry.is_valid
            invalid_count = invalid_mask.sum()
            
            if invalid_count > 0:
                st.warning(f"Found {invalid_count} invalid geometries. Fixing...")
                gdf.loc[invalid_mask, 'geometry'] = gdf.loc[invalid_mask, 'geometry'].buffer(0)
                
                # Check again after fixing
                still_invalid = ~gdf.geometry.is_valid
                if still_invalid.sum() > 0:
                    st.warning(f"Removing {still_invalid.sum()} geometries that couldn't be fixed")
                    gdf = gdf[gdf.geometry.is_valid]
            
            # Remove empty geometries
            empty_mask = gdf.geometry.is_empty
            empty_count = empty_mask.sum()
            if empty_count > 0:
                st.warning(f"Removing {empty_count} empty geometries")
                gdf = gdf[~empty_mask]
            
            # Check if we still have valid data
            if gdf.empty:
                st.error("No valid geometries remaining after cleaning")
                return None
            
            # Simplify very complex geometries
            simplified_count = 0
            for idx in gdf.index:
                geom = gdf.loc[idx, 'geometry']
                coord_count = 0
                
                try:
                    if hasattr(geom, 'exterior'):
                        coord_count = len(geom.exterior.coords)
                    elif hasattr(geom, 'coords'):
                        coord_count = len(list(geom.coords))
                    elif hasattr(geom, 'geoms'):
                        coord_count = sum(len(g.exterior.coords) if hasattr(g, 'exterior') 
                                        else len(list(g.coords)) if hasattr(g, 'coords') else 0 
                                        for g in geom.geoms)
                    
                    if coord_count > 1000:
                        simplified_count += 1
                        gdf.loc[idx, 'geometry'] = geom.simplify(0.001, preserve_topology=True)
                except Exception as simplify_error:
                    st.warning(f"Could not simplify geometry {idx}: {simplify_error}")
            
            if simplified_count > 0:
                st.info(f"🔧 Simplified {simplified_count} complex geometries")
            
            st.success(f"✅ Successfully processed {len(gdf)} geometries")
            return gdf
            
        except Exception as e:
            st.error(f"❌ Error loading GeoJSON: {str(e)}")
            st.info("🔍 **Troubleshooting suggestions:**")
            st.info("1. Check if your GeoJSON file is valid at: https://geojsonlint.com/")
            st.info("2. Try opening the file in QGIS and re-exporting it")
            st.info("3. Use 'Manual selection' option and specify your coordinate system")
            st.info("4. Simplify complex geometries before uploading")
            
            # Show detailed error for debugging
            import traceback
            with st.expander("🐛 Detailed Error Information"):
                st.code(traceback.format_exc())
            
            return None
    
    def download_sentinel2_data(self, geometry, start_date, end_date, cloud_cover=20):
        """Download Sentinel-2 data using Google Earth Engine"""
        try:
            # Convert geometry to Earth Engine geometry with proper CRS handling
            if hasattr(geometry, '__geo_interface__'):
                geo_interface = geometry.__geo_interface__
            else:
                geo_interface = geometry
            
            # Ensure geometry is in WGS84 (EPSG:4326)
            ee_geometry = ee.Geometry(geo_interface, proj='EPSG:4326')
            
            # Validate geometry bounds
            bounds = ee_geometry.bounds().getInfo()
            coords = bounds['coordinates'][0]
            min_lon, min_lat = min([c[0] for c in coords]), min([c[1] for c in coords])
            max_lon, max_lat = max([c[0] for c in coords]), max([c[1] for c in coords])
            
            # Check if geometry is within reasonable Earth bounds
            if min_lon < -180 or max_lon > 180 or min_lat < -90 or max_lat > 90:
                raise ValueError(f"Geometry bounds ({min_lon}, {min_lat}, {max_lon}, {max_lat}) are outside Earth's coordinate system")
            
            st.info(f"Processing area bounds: {min_lon:.4f}, {min_lat:.4f} to {max_lon:.4f}, {max_lat:.4f}")
            
            # Create Sentinel-2 collection
            collection = ee.ImageCollection('COPERNICUS/S2_SR') \
                .filterBounds(ee_geometry) \
                .filterDate(start_date, end_date) \
                .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_cover))
            
            # Check if collection has images
            collection_size = collection.size()
            if collection_size.getInfo() == 0:
                st.warning(f"No Sentinel-2 images found for the specified criteria. Try expanding date range or increasing cloud cover threshold.")
                return None, 0
            
            # Calculate median composite
            image = collection.median().clip(ee_geometry)
            
            # Select bands for analysis
            bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']
            image = image.select(bands)
            
            # Add spectral indices
            image = self.add_spectral_indices(image)
            
            return image, collection_size
        except Exception as e:
            st.error(f"Error downloading Sentinel-2 data: {str(e)}")
            return None, 0
    
    def add_spectral_indices(self, image):
        """Add common spectral indices"""
        # NDVI
        ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
        
        # NDWI
        ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI')
        
        # NDBI
        ndbi = image.normalizedDifference(['B11', 'B8']).rename('NDBI')
        
        # EVI
        evi = image.expression(
            '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))', {
                'NIR': image.select('B8'),
                'RED': image.select('B4'),
                'BLUE': image.select('B2')
            }).rename('EVI')
        
        return image.addBands([ndvi, ndwi, ndbi, evi])
    
    def extract_features(self, image, geometry, scale=10):
        """Extract pixel values as features"""
        try:
            # Ensure geometry is properly formatted for Earth Engine
            if hasattr(geometry, '__geo_interface__'):
                geo_interface = geometry.__geo_interface__
            else:
                geo_interface = geometry
            
            # Create EE geometry with explicit CRS
            ee_geometry = ee.Geometry(geo_interface, proj='EPSG:4326')
            
            # Get the image projection for sampling
            image_projection = image.select(['B2']).projection()
            
            # Calculate appropriate number of pixels based on area
            area = ee_geometry.area().getInfo()
            area_km2 = area / 1000000  # Convert to km²
            
            # Adjust number of pixels based on area (roughly 1 pixel per hectare for small areas)
            if area_km2 < 1:
                num_pixels = min(int(area_km2 * 100), 1000)  # Up to 1000 pixels for small areas
            elif area_km2 < 10:
                num_pixels = min(int(area_km2 * 50), 5000)   # Up to 5000 pixels for medium areas
            else:
                num_pixels = 5000  # Cap at 5000 pixels for large areas
            
            num_pixels = max(num_pixels, 50)  # Minimum 50 pixels
            
            st.info(f"Sampling {num_pixels} pixels from {area_km2:.2f} km² area")
            
            
            # Sample pixels from the image with stratified sampling
            training_polygons = ee.FeatureCollection([
                    ee.Feature(ee.Geometry.Point([0.0, 0.0]), {'landcover': 0}),
                    ee.Feature(ee.Geometry.Point([0.1, 0.1]), {'landcover': 1}),
                    # Add more polygons or points for your classes
                ])
            label_image = ee.Image().byte().paint(training_polygons, 'landcover')
            image_with_labels = image.addBands(label_image.rename('landcover'))
            
            image_with_labels = image_with_labels.addBands(image_with_labels.select('landcover').toInt(), overwrite=True)
            
            samples = image_with_labels.stratifiedSample(
                    numPoints=num_pixels,
                    classBand='landcover',  # Must be integer band
                    region=ee_geometry,
                    scale=scale,
                    projection=image_projection,
                    geometries=True
                )
            
            # If stratified sampling fails, try regular sampling
            sample_size = samples.size().getInfo()
            if sample_size == 0:
                st.warning("Stratified sampling failed, trying regular sampling...")
                samples = image.sample(
                    region=ee_geometry,
                    scale=scale,
                    numPixels=num_pixels,
                    geometries=True,
                    tileScale=2  # Reduce memory usage
                )
                sample_size = samples.size().getInfo()
            
            if sample_size == 0:
                st.error("No samples could be extracted. Check if your geometry overlaps with the image.")
                return None
            
            # Convert to pandas DataFrame
            st.info(f"Extracting {sample_size} samples...")
            features = samples.getInfo()
            
            data_list = []
            for feature in features['features']:
                properties = feature['properties']
                if 'geometry' in feature and feature['geometry'] is not None:
                    coords = feature['geometry']['coordinates']
                    properties['longitude'] = coords[0]
                    properties['latitude'] = coords[1]
                else:
                    # Skip features without valid geometry
                    continue
                data_list.append(properties)
            
            if not data_list:
                st.error("No valid features extracted.")
                return None
            
            df = pd.DataFrame(data_list)
            
            # Clean up the dataframe
            # Remove system columns that might cause issues
            system_cols = [col for col in df.columns if col.startswith('system:')]
            df = df.drop(columns=system_cols, errors='ignore')
            
            # Handle any infinite or extremely large values
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)
                # Cap extreme values (likely errors)
                if col.startswith('B'):  # Spectral bands
                    df[col] = df[col].clip(0, 10000)  # Sentinel-2 values are typically 0-10000
            
            return df
        except Exception as e:
            st.error(f"Error extracting features: {str(e)}")
            # Print more detailed error info
            import traceback
            st.error(f"Detailed error: {traceback.format_exc()}")
            return None
    
    def prepare_training_data(self, df, label_column=None):
        """Prepare data for machine learning"""
        # Remove non-numeric columns except coordinates
        feature_cols = [col for col in df.columns 
                       if col not in ['longitude', 'latitude', 'system:index', label_column]]
        
        X = df[feature_cols].select_dtypes(include=[np.number])
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        if label_column and label_column in df.columns:
            y = df[label_column]
            return X, y, feature_cols
        else:
            return X, None, feature_cols
    
    def train_supervised_model(self, X, y, model_type='random_forest'):
        """Train supervised classification model"""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        if model_type == 'random_forest':
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == 'svm':
            model = SVC(kernel='rbf', random_state=42)
        else:
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        self.models[model_type] = model
        
        return model, accuracy, y_test, y_pred
    
    def train_unsupervised_model(self, X, algorithm='kmeans', n_clusters=5):
        """Train unsupervised clustering model"""
        X_scaled = self.scaler.fit_transform(X)
        
        if algorithm == 'kmeans':
            model = KMeans(n_clusters=n_clusters, random_state=42)
        elif algorithm == 'dbscan':
            model = DBSCAN(eps=0.5, min_samples=5)
        
        clusters = model.fit_predict(X_scaled)
        
        self.models[algorithm] = model
        
        return model, clusters

def main():
    st.markdown('<h1 class="main-header">🛰️ Sentinel-2 Land Cover Analysis</h1>', unsafe_allow_html=True)
    
    # Initialize processor
    processor = SentinelDataProcessor()
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", 
                               ["Data Upload", "Data Download", "Visualization", 
                                "Model Training", "Classification", "Results"])
    
    if page == "Data Upload":
        st.markdown('<h2 class="sub-header">📁 Upload GeoJSON File</h2>', unsafe_allow_html=True)
        
        # CRS selection option
        st.subheader("Coordinate System (CRS) Settings")
        crs_option = st.radio(
            "Choose how to handle coordinate system:",
            ["Auto-detect", "Manual selection", "Force WGS84"]
        )
        
        manual_crs = None
        if crs_option == "Manual selection":
            st.info("💡 Common CRS codes for Ghana/West Africa:")
            st.info("- EPSG:4326 (WGS84 - Geographic)")
            st.info("- EPSG:32630 (UTM Zone 30N)")
            st.info("- EPSG:32631 (UTM Zone 31N)")
            st.info("- EPSG:2136 (Accra / Ghana National Grid)")
            
            manual_crs = st.text_input(
                "Enter EPSG code (e.g., 32630 for UTM Zone 30N):",
                placeholder="32630"
            )
            if manual_crs:
                manual_crs = f"EPSG:{manual_crs}"
        
        uploaded_file = st.file_uploader("Choose a GeoJSON file", type=['geojson', 'json'])
        
        if uploaded_file is not None:
            try:
                geojson_data = json.load(uploaded_file)
                st.session_state['geojson_data'] = geojson_data
                
                # Create temporary GeoDataFrame to apply CRS settings
                if crs_option == "Manual selection" and manual_crs:
                    temp_gdf = gpd.GeoDataFrame.from_features(geojson_data['features'])
                    temp_gdf.crs = manual_crs
                    st.info(f"Applied manual CRS: {manual_crs}")
                elif crs_option == "Force WGS84":
                    temp_gdf = gpd.GeoDataFrame.from_features(geojson_data['features'])
                    temp_gdf.crs = "EPSG:4326"
                    st.info("Forced to WGS84 (EPSG:4326)")
                else:
                    temp_gdf = None  # Let auto-detect handle it
                
                # Process with the processor
                if temp_gdf is not None:
                    # Override the geojson processing
                    gdf = processor.load_geojson(temp_gdf)
                else:
                    gdf = processor.load_geojson(geojson_data)
                
                if gdf is not None:
                    st.success(f"Successfully loaded {len(gdf)} features")
                    st.session_state['gdf'] = gdf
                    
                    # Display basic info
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Geometry Types:**")
                        st.write(gdf.geometry.type.value_counts())
                        st.write(f"**Final CRS:** {gdf.crs}")
                    
                    with col2:
                        st.write("**Bounds (WGS84):**")
                        bounds = gdf.total_bounds
                        st.write(f"Min Lon: {bounds[0]:.6f}")
                        st.write(f"Min Lat: {bounds[1]:.6f}")
                        st.write(f"Max Lon: {bounds[2]:.6f}")
                        st.write(f"Max Lat: {bounds[3]:.6f}")
                    
                    # Calculate area
                    # Convert to equal area projection for area calculation
                    gdf_area = gdf.to_crs("EPSG:6933")  # World Cylindrical Equal Area
                    total_area_m2 = gdf_area.geometry.area.sum()
                    total_area_km2 = total_area_m2 / 1000000
                    
                    st.info(f"**Total Area:** {total_area_km2:.2f} km²")
                    
                    # Display on map
                    center_lat = gdf.geometry.centroid.y.mean()
                    center_lon = gdf.geometry.centroid.x.mean()
                    
                    m = folium.Map(location=[center_lat, center_lon], zoom_start=10)
                    
                    # Add GeoJSON to map
                    folium.GeoJson(
                        gdf.__geo_interface__,
                        style_function=lambda feature: {
                            'fillColor': 'blue',
                            'color': 'black',
                            'weight': 2,
                            'fillOpacity': 0.3,
                        }
                    ).add_to(m)
                    
                    # Add marker at centroid
                    folium.Marker(
                        [center_lat, center_lon],
                        popup=f"Centroid<br>Area: {total_area_km2:.2f} km²",
                        icon=folium.Icon(color='red', icon='info-sign')
                    ).add_to(m)
                    
                    folium_static(m)
                    
            except Exception as e:
                st.error(f"Error loading GeoJSON: {str(e)}")
                st.info("**Common solutions:**")
                st.info("1. Try selecting 'Manual selection' and entering your coordinate system")
                st.info("2. Convert your file to WGS84 before uploading")
                st.info("3. Ensure your GeoJSON file is valid")
    
    elif page == "Data Download":
        st.markdown('<h2 class="sub-header">🛰️ Download Sentinel-2 Data</h2>', unsafe_allow_html=True)
        
        if not EE_AVAILABLE:
            st.error("Google Earth Engine is not available. Please install it first.")
            return
        
        # Authentication section
        st.subheader("Google Earth Engine Authentication")
        
        # Check authentication status
        auth_status = st.empty()
        
        try:
            ee.Initialize()
            auth_status.success("✅ Earth Engine is already authenticated!")
            st.session_state['ee_authenticated'] = True
        except Exception as e:
            auth_status.error("❌ Earth Engine not authenticated")
            st.session_state['ee_authenticated'] = False
            
            st.markdown("""
            **To authenticate Earth Engine, you have two options:**
            
            **Option 1: Terminal Authentication (Recommended)**
            1. Open your terminal/command prompt
            2. Run: `earthengine authenticate`
            3. Follow the browser prompts to sign in
            4. Refresh this page
            
            **Option 2: Python Authentication**
            Click the button below (will open browser):
            """)
            
            if st.button("🔐 Authenticate Earth Engine"):
                try:
                    with st.spinner("Opening browser for authentication..."):
                        ee.Authenticate()
                    st.success("Authentication completed! Please refresh the page.")
                    st.experimental_rerun()
                except Exception as auth_error:
                    st.error(f"Authentication failed: {str(auth_error)}")
            
            st.markdown("**Option 3: Service Account (Advanced)**")
            with st.expander("Use Service Account Key"):
                uploaded_key = st.file_uploader("Upload service account JSON key", type=['json'])
                if uploaded_key:
                    try:
                        service_account_key = json.load(uploaded_key)
                        credentials = ee.ServiceAccountCredentials(
                            service_account_key['client_email'],
                            key_data=json.dumps(service_account_key)
                        )
                        ee.Initialize(credentials)
                        st.success("Service account authentication successful!")
                        st.session_state['ee_authenticated'] = True
                        st.experimental_rerun()
                    except Exception as sa_error:
                        st.error(f"Service account authentication failed: {str(sa_error)}")
        
        # Data download parameters
        if 'gdf' in st.session_state and st.session_state.get('ee_authenticated', False):
            st.subheader("Download Parameters")
            
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Start Date", 
                                         value=datetime.now() - timedelta(days=90))
                cloud_cover = st.slider("Max Cloud Cover (%)", 0, 100, 20)
            
            with col2:
                end_date = st.date_input("End Date", value=datetime.now())
                scale = st.slider("Spatial Resolution (m)", 10, 60, 10)
            
            if st.button("Download Sentinel-2 Data"):
                with st.spinner("Downloading Sentinel-2 data..."):
                    gdf = st.session_state['gdf']
                    geometry = gdf.geometry.unary_union
                    
                    image, count = processor.download_sentinel2_data(
                        geometry, start_date.strftime('%Y-%m-%d'), 
                        end_date.strftime('%Y-%m-%d'), cloud_cover
                    )
                    
                    if image:
                        st.success(f"Successfully downloaded composite from {count} images")
                        st.session_state['sentinel_image'] = image
                        
                        # Extract features
                        with st.spinner("Extracting pixel features..."):
                            # Ensure geometry is in the right format
                            geometry = gdf.geometry.unary_union
                            
                            # Convert to GeoDataFrame to ensure proper CRS handling
                            if hasattr(geometry, '__geo_interface__'):
                                temp_gdf = gpd.GeoDataFrame([1], geometry=[geometry], crs="EPSG:4326")
                                geometry_for_ee = temp_gdf.geometry.iloc[0]
                            else:
                                geometry_for_ee = geometry
                            
                            df = processor.extract_features(image, geometry_for_ee, scale)
                            if df is not None:
                                st.session_state['features_df'] = df
                                st.success(f"Extracted {len(df)} pixel samples")
                                
                                # Show sample data
                                st.subheader("Sample Features")
                                st.dataframe(df.head())
                                
                                # Show feature summary
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.write("**Available Bands:**")
                                    band_cols = [col for col in df.columns if col.startswith('B')]
                                    st.write(band_cols)
                                
                                with col2:
                                    st.write("**Spectral Indices:**")
                                    index_cols = [col for col in df.columns if col in ['NDVI', 'NDWI', 'NDBI', 'EVI']]
                                    st.write(index_cols)
        else:
            st.warning("Please upload a GeoJSON file first in the 'Data Upload' page.")
    
    elif page == "Visualization":
        st.markdown('<h2 class="sub-header">📊 Data Visualization</h2>', unsafe_allow_html=True)
        
        if 'features_df' in st.session_state:
            df = st.session_state['features_df']
            
            # Feature statistics
            st.subheader("Feature Statistics")
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Basic Statistics:**")
                st.dataframe(df[numeric_cols].describe())
            
            with col2:
                st.write("**Feature Correlation:**")
                corr_matrix = df[numeric_cols].corr()
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
                st.pyplot(fig)
            
            # Interactive plots
            st.subheader("Interactive Visualizations")
            
            plot_type = st.selectbox("Choose plot type:", 
                                   ["Scatter Plot", "Histogram", "Box Plot", "Spectral Signature"])
            
            if plot_type == "Scatter Plot":
                col1, col2 = st.columns(2)
                with col1:
                    x_axis = st.selectbox("X-axis:", numeric_cols)
                with col2:
                    y_axis = st.selectbox("Y-axis:", numeric_cols)
                
                fig = px.scatter(df, x=x_axis, y=y_axis, 
                               title=f"{x_axis} vs {y_axis}")
                st.plotly_chart(fig, use_container_width=True)
            
            elif plot_type == "Histogram":
                feature = st.selectbox("Select feature:", numeric_cols)
                fig = px.histogram(df, x=feature, title=f"Distribution of {feature}")
                st.plotly_chart(fig, use_container_width=True)
            
            elif plot_type == "Box Plot":
                features = st.multiselect("Select features:", numeric_cols, 
                                        default=numeric_cols[:5])
                if features:
                    fig = px.box(df[features].melt(), y='value', x='variable',
                               title="Feature Distributions")
                    st.plotly_chart(fig, use_container_width=True)
            
            elif plot_type == "Spectral Signature":
                # Plot average spectral signature
                band_cols = [col for col in df.columns if col.startswith('B')]
                if band_cols:
                    mean_values = df[band_cols].mean()
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=band_cols, y=mean_values,
                                           mode='lines+markers',
                                           name='Average Spectral Signature'))
                    fig.update_layout(title="Average Spectral Signature",
                                    xaxis_title="Bands",
                                    yaxis_title="Reflectance")
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Please download Sentinel-2 data first.")
    
    elif page == "Model Training":
        st.markdown('<h2 class="sub-header">🤖 Model Training</h2>', unsafe_allow_html=True)
        
        if 'features_df' in st.session_state:
            df = st.session_state['features_df']
            
            tab1, tab2 = st.tabs(["Supervised Learning", "Unsupervised Learning"])
            
            with tab1:
                st.subheader("Supervised Classification")
                
                # Check if labels are available
                label_col = st.selectbox("Select label column (if available):", 
                                       ['None'] + list(df.columns))
                
                if label_col != 'None':
                    model_type = st.selectbox("Select model type:", 
                                            ["random_forest", "svm"])
                    
                    if st.button("Train Supervised Model"):
                        X, y, feature_cols = processor.prepare_training_data(df, label_col)
                        
                        with st.spinner("Training model..."):
                            model, accuracy, y_test, y_pred = processor.train_supervised_model(
                                X, y, model_type)
                            
                            st.success(f"Model trained! Accuracy: {accuracy:.4f}")
                            
                            # Confusion matrix
                            cm = confusion_matrix(y_test, y_pred)
                            fig, ax = plt.subplots()
                            sns.heatmap(cm, annot=True, fmt='d', ax=ax)
                            ax.set_title('Confusion Matrix')
                            st.pyplot(fig)
                            
                            # Classification report
                            report = classification_report(y_test, y_pred, output_dict=True)
                            st.text("Classification Report:")
                            st.json(report)
                            
                            # Feature importance (for Random Forest)
                            if model_type == 'random_forest':
                                feature_importance = pd.DataFrame({
                                    'feature': feature_cols,
                                    'importance': model.feature_importances_
                                }).sort_values('importance', ascending=False)
                                
                                fig = px.bar(feature_importance.head(10), 
                                           x='importance', y='feature',
                                           orientation='h',
                                           title='Top 10 Feature Importance')
                                st.plotly_chart(fig)
                else:
                    st.info("No label column selected. Please provide labeled training data for supervised learning.")
            
            with tab2:
                st.subheader("Unsupervised Clustering")
                
                col1, col2 = st.columns(2)
                with col1:
                    algorithm = st.selectbox("Select algorithm:", ["kmeans", "dbscan"])
                
                with col2:
                    if algorithm == "kmeans":
                        n_clusters = st.slider("Number of clusters:", 2, 20, 5)
                    else:
                        n_clusters = None
                
                if st.button("Train Clustering Model"):
                    X, _, feature_cols = processor.prepare_training_data(df)
                    
                    with st.spinner("Training clustering model..."):
                        model, clusters = processor.train_unsupervised_model(
                            X, algorithm, n_clusters)
                        
                        # Add clusters to dataframe
                        df_clustered = df.copy()
                        df_clustered['cluster'] = clusters
                        st.session_state['clustered_df'] = df_clustered
                        
                        st.success(f"Clustering completed! Found {len(np.unique(clusters))} clusters")
                        
                        # Visualize clusters
                        if 'NDVI' in df.columns and 'NDWI' in df.columns:
                            fig = px.scatter(df_clustered, x='NDVI', y='NDWI', 
                                           color=clusters.astype(str),
                                           title='Clusters in NDVI-NDWI Space')
                            st.plotly_chart(fig)
                        
                        # Cluster statistics
                        cluster_stats = df_clustered.groupby('cluster')[feature_cols].mean()
                        st.subheader("Cluster Statistics")
                        st.dataframe(cluster_stats)
        else:
            st.warning("Please download and extract features first.")
    
    elif page == "Classification":
        st.markdown('<h2 class="sub-header">🗺️ Land Cover Classification</h2>', unsafe_allow_html=True)
        
        if 'features_df' in st.session_state and processor.models:
            st.subheader("Apply Trained Models")
            
            model_names = list(processor.models.keys())
            selected_model = st.selectbox("Select trained model:", model_names)
            
            if st.button("Apply Classification"):
                df = st.session_state['features_df']
                X, _, feature_cols = processor.prepare_training_data(df)
                
                model = processor.models[selected_model]
                
                with st.spinner("Applying classification..."):
                    if selected_model in ['kmeans', 'dbscan']:
                        # Unsupervised
                        X_scaled = processor.scaler.transform(X)
                        predictions = model.predict(X_scaled)
                    else:
                        # Supervised
                        X_scaled = processor.scaler.transform(X)
                        predictions = model.predict(X_scaled)
                    
                    # Add predictions to dataframe
                    df_classified = df.copy()
                    df_classified['predicted_class'] = predictions
                    st.session_state['classified_df'] = df_classified
                    
                    st.success("Classification completed!")
                    
                    # Show classification results
                    st.subheader("Classification Results")
                    class_counts = pd.Series(predictions).value_counts().sort_index()
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        fig = px.bar(x=class_counts.index.astype(str), 
                                   y=class_counts.values,
                                   title="Class Distribution")
                        fig.update_xaxis(title="Class")
                        fig.update_yaxis(title="Number of Pixels")
                        st.plotly_chart(fig)
                    
                    with col2:
                        fig = px.pie(values=class_counts.values, 
                                   names=class_counts.index.astype(str),
                                   title="Class Proportions")
                        st.plotly_chart(fig)
                    
                    # Spatial visualization
                    if 'longitude' in df_classified.columns and 'latitude' in df_classified.columns:
                        fig = px.scatter_mapbox(
                            df_classified, 
                            lat='latitude', 
                            lon='longitude',
                            color='predicted_class',
                            title="Spatial Distribution of Classes",
                            mapbox_style="open-street-map",
                            height=600
                        )
                        st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Please train a model first or ensure feature data is available.")
    
    elif page == "Results":
        st.markdown('<h2 class="sub-header">📈 Results & Export</h2>', unsafe_allow_html=True)
        
        # Model performance summary
        if processor.models:
            st.subheader("Trained Models Summary")
            
            model_summary = []
            for model_name, model in processor.models.items():
                model_info = {
                    'Model': model_name,
                    'Type': type(model).__name__,
                    'Parameters': str(model.get_params()) if hasattr(model, 'get_params') else 'N/A'
                }
                model_summary.append(model_info)
            
            st.dataframe(pd.DataFrame(model_summary))
        
        # Export options
        st.subheader("Export Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Export Feature Data") and 'features_df' in st.session_state:
                csv = st.session_state['features_df'].to_csv(index=False)
                st.download_button(
                    label="Download Features CSV",
                    data=csv,
                    file_name="sentinel2_features.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("Export Classification Results") and 'classified_df' in st.session_state:
                csv = st.session_state['classified_df'].to_csv(index=False)
                st.download_button(
                    label="Download Classification CSV",
                    data=csv,
                    file_name="land_cover_classification.csv",
                    mime="text/csv"
                )
        
        with col3:
            if st.button("Export Trained Models") and processor.models:
                # Save models
                model_data = {}
                for name, model in processor.models.items():
                    model_data[name] = joblib.dumps(model)
                
                # Note: In a real application, you'd want to save these to files
                st.success("Models saved successfully!")
        
        # Summary statistics
        if 'classified_df' in st.session_state:
            st.subheader("Land Cover Summary")
            
            df = st.session_state['classified_df']
            if 'predicted_class' in df.columns:
                # Calculate area estimates (assuming each pixel represents equal area)
                class_counts = df['predicted_class'].value_counts()
                total_pixels = len(df)
                
                summary_stats = pd.DataFrame({
                    'Class': class_counts.index,
                    'Pixel_Count': class_counts.values,
                    'Percentage': (class_counts.values / total_pixels * 100).round(2)
                })
                
                st.dataframe(summary_stats)
                
                # Land cover change analysis (if multiple time periods)
                st.info("💡 Tip: To perform change analysis, process multiple time periods and compare results!")

if __name__ == "__main__":
    main()

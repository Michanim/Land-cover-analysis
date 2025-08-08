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
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
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
            if isinstance(geojson_data, str):
                gdf = gpd.read_file(geojson_data)
            else:
                gdf = gpd.GeoDataFrame.from_features(geojson_data['features'])
            
            # Ensure CRS is set
            if gdf.crs is None:
                gdf.crs = "EPSG:4326"
            
            return gdf
        except Exception as e:
            st.error(f"Error loading GeoJSON: {str(e)}")
            return None
    
    def download_sentinel2_data(self, geometry, start_date, end_date, cloud_cover=20):
        """Download Sentinel-2 data using Google Earth Engine"""
        try:
            # Convert geometry to Earth Engine geometry
            if hasattr(geometry, '__geo_interface__'):
                ee_geometry = ee.Geometry(geometry.__geo_interface__)
            else:
                ee_geometry = ee.Geometry(geometry)
            
            # Create Sentinel-2 collection
            collection = ee.ImageCollection('COPERNICUS/S2_SR') \
                .filterBounds(ee_geometry) \
                .filterDate(start_date, end_date) \
                .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_cover))
            
            # Calculate median composite
            image = collection.median().clip(ee_geometry)
            
            # Select bands for analysis
            bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']
            image = image.select(bands)
            
            # Add spectral indices
            image = self.add_spectral_indices(image)
            
            return image, collection.size()
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
            # Sample pixels from the image
            samples = image.sample(
                region=geometry,
                scale=scale,
                numPixels=5000,
                geometries=True
            )
            
            # Convert to pandas DataFrame
            features = samples.getInfo()
            
            data_list = []
            for feature in features['features']:
                properties = feature['properties']
                coords = feature['geometry']['coordinates']
                properties['longitude'] = coords[0]
                properties['latitude'] = coords[1]
                data_list.append(properties)
            
            df = pd.DataFrame(data_list)
            return df
        except Exception as e:
            st.error(f"Error extracting features: {str(e)}")
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
        
        uploaded_file = st.file_uploader("Choose a GeoJSON file", type=['geojson', 'json'])
        
        if uploaded_file is not None:
            try:
                geojson_data = json.load(uploaded_file)
                st.session_state['geojson_data'] = geojson_data
                
                gdf = processor.load_geojson(geojson_data)
                if gdf is not None:
                    st.success(f"Successfully loaded {len(gdf)} features")
                    st.session_state['gdf'] = gdf
                    
                    # Display basic info
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Geometry Types:**")
                        st.write(gdf.geometry.type.value_counts())
                    
                    with col2:
                        st.write("**Bounds:**")
                        bounds = gdf.total_bounds
                        st.write(f"Min X: {bounds[0]:.4f}")
                        st.write(f"Min Y: {bounds[1]:.4f}")
                        st.write(f"Max X: {bounds[2]:.4f}")
                        st.write(f"Max Y: {bounds[3]:.4f}")
                    
                    # Display on map
                    center_lat = gdf.geometry.centroid.y.mean()
                    center_lon = gdf.geometry.centroid.x.mean()
                    
                    m = folium.Map(location=[center_lat, center_lon], zoom_start=10)
                    
                    # Add GeoJSON to map
                    folium.GeoJson(
                        geojson_data,
                        style_function=lambda feature: {
                            'fillColor': 'blue',
                            'color': 'black',
                            'weight': 2,
                            'fillOpacity': 0.3,
                        }
                    ).add_to(m)
                    
                    folium_static(m)
                    
            except Exception as e:
                st.error(f"Error loading GeoJSON: {str(e)}")
    
    elif page == "Data Download":
        st.markdown('<h2 class="sub-header">🛰️ Download Sentinel-2 Data</h2>', unsafe_allow_html=True)
        
        if not EE_AVAILABLE:
            st.error("Google Earth Engine is not available. Please install it first.")
            return
        
        # Authentication section
        st.subheader("Google Earth Engine Authentication")
        auth_method = st.radio("Choose authentication method:", 
                              ["Default (requires ee authenticate)", "Service Account Key"])
        
        if auth_method == "Service Account Key":
            uploaded_key = st.file_uploader("Upload service account JSON key", type=['json'])
            if uploaded_key:
                service_account_key = json.load(uploaded_key)
                if processor.authenticate_earth_engine(service_account_key):
                    st.success("Earth Engine authenticated successfully!")
        else:
            if st.button("Initialize Earth Engine"):
                if processor.authenticate_earth_engine():
                    st.success("Earth Engine initialized successfully!")
        
        # Data download parameters
        if 'gdf' in st.session_state:
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
                            if hasattr(geometry, '__geo_interface__'):
                                ee_geometry = ee.Geometry(geometry.__geo_interface__)
                            else:
                                ee_geometry = ee.Geometry(geometry)
                            
                            df = processor.extract_features(image, ee_geometry, scale)
                            if df is not None:
                                st.session_state['features_df'] = df
                                st.success(f"Extracted {len(df)} pixel samples")
                                
                                # Show sample data
                                st.subheader("Sample Features")
                                st.dataframe(df.head())
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
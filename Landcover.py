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
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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

# --- Initialize session state ---
if 'df' not in st.session_state:
    st.session_state.df = None
if 'gdf' not in st.session_state:
    st.session_state.gdf = None
if 'ee_images' not in st.session_state:
    st.session_state.ee_images = {}  # Store multiple years
if 'feature_data' not in st.session_state:
    st.session_state.feature_data = {}  # Store features by year
if 'ee_authenticated' not in st.session_state:
    st.session_state.ee_authenticated = False
if 'trained_models' not in st.session_state:
    st.session_state.trained_models = {}  # Store models by year
if 'classified_data' not in st.session_state:
    st.session_state.classified_data = {}  # Store classifications by year
if 'comparison_results' not in st.session_state:
    st.session_state.comparison_results = None
if 'current_step' not in st.session_state:
    st.session_state.current_step = 0

# ... (keep your existing CSS styling) ...

# --- Enhanced Helper Functions ---
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

def calculate_ndbi(image):
    """Calculate NDBI for built-up area detection"""
    if EE_AVAILABLE:
        ndbi = image.normalizedDifference(['B11', 'B8']).rename('NDBI')
        return image.addBands(ndbi)
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

def get_seasonal_collection(year, geometry, cloud_cover=20):
    """Get seasonal composite for a specific year"""
    if not EE_AVAILABLE:
        return None
    
    # Define seasonal ranges
    seasons = {
        'spring': (f'{year}-03-01', f'{year}-05-31'),
        'summer': (f'{year}-06-01', f'{year}-08-31'),
        'autumn': (f'{year}-09-01', f'{year}-11-30'),
        'winter': (f'{year}-12-01', f'{year}-12-31')  # Adjust for southern hemisphere if needed
    }
    
    seasonal_composites = {}
    
    for season, (start_date, end_date) in seasons.items():
        collection = ee.ImageCollection('COPERNICUS/S2_SR') \
            .filterDate(start_date, end_date) \
            .filterBounds(geometry) \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_cover))
        
        if collection.size().getInfo() > 0:
            # Create median composite for the season
            composite = collection.map(mask_clouds).median()
            composite = calculate_ndvi(composite)
            composite = calculate_ndwi(composite)
            composite = calculate_ndbi(composite)
            seasonal_composites[season] = composite.clip(geometry)
    
    return seasonal_composites

def extract_features_by_year(year, image, geometry, num_pixels=1000):
    """Extract features for a specific year"""
    try:
        features = image.sample(region=geometry, scale=10, numPixels=num_pixels, geometries=True)
        feature_info = features.getInfo()
        
        if feature_info and 'features' in feature_info:
            feature_data = []
            for feature in feature_info['features']:
                props = feature['properties']
                if 'geometry' in feature and feature['geometry']['type'] == 'Point':
                    coords = feature['geometry']['coordinates']
                    props['longitude'] = coords[0]
                    props['latitude'] = coords[1]
                props['year'] = year  # Add year identifier
                feature_data.append(props)
            
            return pd.DataFrame(feature_data)
    except Exception as e:
        st.error(f"Error extracting features for {year}: {str(e)}")
    return None

def compare_land_cover_changes(year1_data, year2_data, year1, year2):
    """Compare land cover changes between two years"""
    if year1_data is None or year2_data is None:
        return None
    
    # Ensure both datasets have the same coordinate system for comparison
    if 'longitude' in year1_data.columns and 'latitude' in year1_data.columns:
        # Create spatial joins for change detection
        from scipy.spatial import cKDTree
        
        coords_year1 = year1_data[['longitude', 'latitude']].values
        coords_year2 = year2_data[['longitude', 'latitude']].values
        
        tree = cKDTree(coords_year2)
        distances, indices = tree.query(coords_year1, k=1)
        
        # Create change analysis
        changes = []
        for i, (idx, dist) in enumerate(zip(indices, distances)):
            if dist < 0.001:  # Threshold for same location (approx 100m)
                class1 = year1_data.iloc[i]['predicted_class']
                class2 = year2_data.iloc[idx]['predicted_class']
                if class1 != class2:
                    changes.append({
                        'longitude': year1_data.iloc[i]['longitude'],
                        'latitude': year1_data.iloc[i]['latitude'],
                        f'class_{year1}': class1,
                        f'class_{year2}': class2,
                        'change_type': f'{class1}→{class2}',
                        'confidence_1': year1_data.iloc[i].get('prediction_confidence', 0),
                        'confidence_2': year2_data.iloc[idx].get('prediction_confidence', 0)
                    })
        
        return pd.DataFrame(changes)
    return None

# ... (keep your existing authenticate_ee, authenticate_with_json_key, get_step_completion_status, render_navigation_arrows functions) ...

# --- Enhanced Step 2: Satellite Data (Multi-year support) ---
elif current_step == 2:  # Satellite Data
    if not EE_AVAILABLE:
        st.markdown('<div class="error-message">❌ Google Earth Engine is not available. Please install earthengine-api.</div>', unsafe_allow_html=True)
    else:
        # Authentication Status
        if st.session_state.ee_authenticated:
            st.markdown('<div class="success-message">✅ Google Earth Engine authenticated successfully!</div>', unsafe_allow_html=True)
        else:
            # ... (keep your existing authentication code) ...
            pass
        
        # Only show download options if authenticated and AOI is available
        if st.session_state.ee_authenticated:
            if st.session_state.gdf is None:
                st.markdown('<div class="warning-message">⚠️ Please upload an AOI (GeoJSON) first in the previous step.</div>', unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="feature-card">
                    <h3>📡 Multi-Year Satellite Data Configuration</h3>
                    <p>Download Sentinel-2 imagery for multiple years to enable temporal analysis.</p>
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    start_year = st.number_input("Start Year", min_value=2015, max_value=2023, value=2020)
                    end_year = st.number_input("End Year", min_value=2015, max_value=2023, value=2023)
                    years = list(range(start_year, end_year + 1))
                    st.write(f"Selected years: {years}")
                
                with col2:
                    season = st.selectbox("Season", ["spring", "summer", "autumn", "winter", "annual"])
                    cloud_cover = st.slider("Maximum Cloud Cover (%)", 0, 100, 20)
                
                with col3:
                    resolution = st.selectbox("Spatial Resolution (m)", [10, 20, 60], index=0)
                    num_pixels = st.slider("Sample Points per Year", 500, 5000, 1000, step=500)
                
                if st.button("🛰️ Download Multi-Year Data", type="primary"):
                    try:
                        with st.spinner("🔄 Processing multi-year satellite data..."):
                            # Convert GeoDataFrame to Earth Engine geometry
                            geom_json = json.loads(st.session_state.gdf.to_json())
                            ee_geom = ee.Geometry(geom_json['features'][0]['geometry'])
                            
                            # Initialize storage
                            st.session_state.ee_images = {}
                            st.session_state.feature_data = {}
                            
                            progress_bar = st.progress(0)
                            total_years = len(years)
                            
                            for i, year in enumerate(years):
                                st.info(f"Processing {year}...")
                                
                                if season == "annual":
                                    # Annual composite
                                    collection = ee.ImageCollection('COPERNICUS/S2_SR') \
                                        .filterDate(f'{year}-01-01', f'{year}-12-31') \
                                        .filterBounds(ee_geom) \
                                        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_cover))
                                else:
                                    # Seasonal composite
                                    seasonal_data = get_seasonal_collection(year, ee_geom, cloud_cover)
                                    if seasonal_data and season in seasonal_data:
                                        collection = ee.ImageCollection([seasonal_data[season]])
                                    else:
                                        st.warning(f"No data found for {year} {season}")
                                        continue
                                
                                if collection.size().getInfo() == 0:
                                    st.warning(f"No images found for {year}. Skipping.")
                                    continue
                                
                                # Create composite
                                image = collection.map(mask_clouds).median().clip(ee_geom)
                                image = calculate_ndvi(image)
                                image = calculate_ndwi(image)
                                image = calculate_ndbi(image)
                                
                                # Store image
                                st.session_state.ee_images[year] = image
                                
                                # Extract features
                                feature_df = extract_features_by_year(year, image, ee_geom, num_pixels)
                                if feature_df is not None:
                                    st.session_state.feature_data[year] = feature_df
                                    st.success(f"✅ {year}: {len(feature_df)} points extracted")
                                
                                progress_bar.progress((i + 1) / total_years)
                            
                            # Display summary
                            if st.session_state.feature_data:
                                st.markdown('<div class="success-message">✅ Multi-year data download completed!</div>', unsafe_allow_html=True)
                                
                                # Summary statistics
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    total_points = sum(len(df) for df in st.session_state.feature_data.values())
                                    st.markdown(f"""<div class="metric-card">
                                        <div class="metric-number">{total_points}</div>
                                        <div class="metric-label">Total Points</div>
                                    </div>""", unsafe_allow_html=True)
                                with col2:
                                    st.markdown(f"""<div class="metric-card">
                                        <div class="metric-number">{len(st.session_state.feature_data)}</div>
                                        <div class="metric-label">Years Processed</div>
                                    </div>""", unsafe_allow_html=True)
                                with col3:
                                    if any('NDVI' in df.columns for df in st.session_state.feature_data.values()):
                                        avg_ndvi = np.mean([df['NDVI'].mean() for df in st.session_state.feature_data.values() if 'NDVI' in df.columns])
                                        st.markdown(f"""<div class="metric-card">
                                            <div class="metric-number">{avg_ndvi:.3f}</div>
                                            <div class="metric-label">Avg NDVI</div>
                                        </div>""", unsafe_allow_html=True)
                                
                                # Temporal NDVI trend
                                if len(st.session_state.feature_data) > 1:
                                    years_sorted = sorted(st.session_state.feature_data.keys())
                                    ndvi_means = []
                                    for year in years_sorted:
                                        df = st.session_state.feature_data[year]
                                        if 'NDVI' in df.columns:
                                            ndvi_means.append(df['NDVI'].mean())
                                    
                                    if ndvi_means:
                                        fig, ax = plt.subplots(figsize=(10, 6))
                                        ax.plot(years_sorted, ndvi_means, marker='o', linewidth=2, markersize=8)
                                        ax.set_xlabel('Year')
                                        ax.set_ylabel('Mean NDVI')
                                        ax.set_title('Temporal NDVI Trend')
                                        ax.grid(True, alpha=0.3)
                                        st.pyplot(fig)
                            
                    except Exception as e:
                        st.markdown(f'<div class="error-message">❌ Error downloading satellite data: {e}</div>', unsafe_allow_html=True)

# --- Enhanced Step 3: Visualization (Multi-year comparisons) ---
elif current_step == 3:  # Visualization
    if not st.session_state.feature_data:
        st.markdown('<div class="info-message">📥 Please download multi-year satellite data first.</div>', unsafe_allow_html=True)
    else:
        years = sorted(st.session_state.feature_data.keys())
        
        st.markdown(f"""
        <div class="feature-card">
            <h3>📊 Multi-Year Data Analysis</h3>
            <p>Analyzing data from {len(years)} years: {', '.join(map(str, years))}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Year selection for detailed analysis
        selected_years = st.multiselect("Select years for comparison:", years, default=years[:2])
        
        if len(selected_years) >= 2:
            # Comparative analysis
            tab1, tab2, tab3, tab4 = st.tabs(["📈 Temporal Trends", "🔗 Year Comparisons", "🗺️ Spatial Changes", "📊 Statistics"])
            
            with tab1:
                # Temporal trends for key indices
                indices = ['NDVI', 'NDWI', 'NDBI'] if any(idx in st.session_state.feature_data[years[0]].columns 
                                                         for idx in ['NDVI', 'NDWI', 'NDBI']) else []
                
                if indices:
                    fig = make_subplots(rows=len(indices), cols=1, subplot_titles=[f'{idx} Trend' for idx in indices])
                    
                    for i, idx in enumerate(indices):
                        years_avail = []
                        values = []
                        for year in years:
                            df = st.session_state.feature_data[year]
                            if idx in df.columns:
                                years_avail.append(year)
                                values.append(df[idx].mean())
                        
                        if years_avail:
                            fig.add_trace(
                                go.Scatter(x=years_avail, y=values, mode='lines+markers', name=idx),
                                row=i+1, col=1
                            )
                    
                    fig.update_layout(height=300*len(indices), title_text="Temporal Trends of Spectral Indices")
                    st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                # Scatter comparisons between years
                if len(selected_years) == 2:
                    year1, year2 = selected_years
                    df1 = st.session_state.feature_data[year1]
                    df2 = st.session_state.feature_data[year2]
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        index = st.selectbox("Select index for comparison:", ['NDVI', 'NDWI', 'NDBI'])
                        if index in df1.columns and index in df2.columns:
                            fig, ax = plt.subplots(figsize=(8, 6))
                            ax.scatter(df1[index].sample(min(500, len(df1))), 
                                      df2[index].sample(min(500, len(df2))), alpha=0.6)
                            ax.plot([0, 1], [0, 1], 'r--', alpha=0.8)
                            ax.set_xlabel(f'{year1} {index}')
                            ax.set_ylabel(f'{year2} {index}')
                            ax.set_title(f'{index} Comparison: {year1} vs {year2}')
                            st.pyplot(fig)
                    
                    with col2:
                        # Distribution comparison
                        if index in df1.columns and index in df2.columns:
                            fig, ax = plt.subplots(figsize=(8, 6))
                            ax.hist(df1[index].dropna(), bins=50, alpha=0.7, label=str(year1))
                            ax.hist(df2[index].dropna(), bins=50, alpha=0.7, label=str(year2))
                            ax.set_xlabel(index)
                            ax.set_ylabel('Frequency')
                            ax.legend()
                            ax.set_title(f'{index} Distribution Comparison')
                            st.pyplot(fig)
            
            with tab3:
                if len(selected_years) == 2 and 'longitude' in st.session_state.feature_data[years[0]].columns:
                    year1, year2 = selected_years
                    df1 = st.session_state.feature_data[year1].sample(min(500, len(st.session_state.feature_data[year1])))
                    df2 = st.session_state.feature_data[year2].sample(min(500, len(st.session_state.feature_data[year2])))
                    
                    fig = go.Figure()
                    
                    # Add traces for both years
                    fig.add_trace(go.Scattermapbox(
                        lat=df1['latitude'],
                        lon=df1['longitude'],
                        mode='markers',
                        marker=dict(size=8, color='blue'),
                        name=str(year1),
                        text=df1.get('NDVI', np.zeros(len(df1)))
                    ))
                    
                    fig.add_trace(go.Scattermapbox(
                        lat=df2['latitude'],
                        lon=df2['longitude'],
                        mode='markers',
                        marker=dict(size=8, color='red'),
                        name=str(year2),
                        text=df2.get('NDVI', np.zeros(len(df2)))
                    ))
                    
                    fig.update_layout(
                        mapbox_style="open-street-map",
                        mapbox=dict(zoom=10),
                        margin={"r":0,"t":0,"l":0,"b":0},
                        height=600
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            with tab4:
                # Statistical summary table
                summary_data = []
                for year in selected_years:
                    df = st.session_state.feature_data[year]
                    summary = {'Year': year, 'Samples': len(df)}
                    for idx in ['NDVI', 'NDWI', 'NDBI']:
                        if idx in df.columns:
                            summary[f'{idx}_mean'] = df[idx].mean()
                            summary[f'{idx}_std'] = df[idx].std()
                    summary_data.append(summary)
                
                st.dataframe(pd.DataFrame(summary_data).round(4))

# --- Enhanced Step 4: Model Training (Multi-year support) ---
elif current_step == 4:  # Model Training
    if not st.session_state.feature_data:
        st.markdown('<div class="warning-message">⚠️ Please download multi-year satellite data first.</div>', unsafe_allow_html=True)
    else:
        years = sorted(st.session_state.feature_data.keys())
        
        st.markdown("""
        <div class="feature-card">
            <h3>🎯 Multi-Year Model Training</h3>
            <p>Train models for individual years or combined datasets.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Training configuration
        col1, col2 = st.columns(2)
        
        with col1:
            training_strategy = st.radio(
                "Training Strategy:",
                ["Individual Years", "Combined Dataset", "Transfer Learning"],
                help="Individual: Train separate models for each year. Combined: Train one model on all data. Transfer: Train on one year, apply to others."
            )
            
            if training_strategy == "Individual Years":
                selected_years = st.multiselect("Select years to train:", years, default=years[0:1])
            elif training_strategy == "Combined Dataset":
                selected_years = years
            else:  # Transfer Learning
                source_year = st.selectbox("Source year (training):", years)
                target_years = st.multiselect("Target years (application):", [y for y in years if y != source_year])
                selected_years = [source_year]
        
        with col2:
            # Feature selection
            available_features = []
            if years:
                available_features = [col for col in st.session_state.feature_data[years[0]].columns 
                                    if col not in ['longitude', 'latitude', 'year'] and pd.api.types.is_numeric_dtype(st.session_state.feature_data[years[0]][col])]
            
            feature_cols = st.multiselect("Feature columns:", available_features, default=available_features[:5])
            test_size = st.slider("Test set size (%)", 10, 40, 20) / 100
        
        if st.button("🚀 Train Models", type="primary"):
            if not feature_cols or not selected_years:
                st.markdown('<div class="error-message">❌ Please select features and years</div>', unsafe_allow_html=True)
            else:
                try:
                    st.session_state.trained_models = {}
                    
                    if training_strategy == "Combined Dataset":
                        # Combine all years data
                        combined_data = []
                        for year in selected_years:
                            df = st.session_state.feature_data[year].copy()
                            df['year'] = year
                            combined_data.append(df)
                        
                        combined_df = pd.concat(combined_data, ignore_index=True)
                        # Train single model on combined data
                        # ... (your existing training code) ...
                        
                    else:
                        # Train individual models
                        for year in selected_years:
                            with st.spinner(f"🔄 Training model for {year}..."):
                                df = st.session_state.feature_data[year].copy()
                                
                                # Create synthetic labels (enhanced for multi-year)
                                if 'NDVI' in df.columns and 'NDWI' in df.columns:
                                    def classify_pixel(row):
                                        ndvi, ndwi, ndbi = row.get('NDVI', 0), row.get('NDWI', 0), row.get('NDBI', 0)
                                        if ndwi > 0.3: return 'Water'
                                        elif ndvi > 0.6: return 'Forest'
                                        elif ndvi > 0.3: return 'Vegetation'
                                        elif ndbi > 0.1: return 'Urban'
                                        elif ndvi < 0.1: return 'Bare_Soil'
                                        else: return 'Mixed'
                                    df['land_cover'] = df.apply(classify_pixel, axis=1)
                                
                                # Train model
                                X = df[feature_cols].fillna(0)
                                y = df['land_cover']
                                
                                le = LabelEncoder()
                                y_encoded = le.fit_transform(y)
                                
                                X_train, X_test, y_train, y_test = train_test_split(
                                    X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
                                )
                                
                                scaler = StandardScaler()
                                X_train_scaled = scaler.fit_transform(X_train)
                                X_test_scaled = scaler.transform(X_test)
                                
                                rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
                                rf_model.fit(X_train_scaled, y_train)
                                
                                y_pred = rf_model.predict(X_test_scaled)
                                accuracy = accuracy_score(y_test, y_pred)
                                
                                st.session_state.trained_models[year] = {
                                    'model': rf_model,
                                    'scaler': scaler,
                                    'label_encoder': le,
                                    'feature_cols': feature_cols,
                                    'accuracy': accuracy
                                }
                    
                    # Display results
                    if st.session_state.trained_models:
                        st.markdown('<div class="success-message">✅ Models trained successfully!</div>', unsafe_allow_html=True)
                        
                        # Model performance comparison
                        performance_data = []
                        for year, model_info in st.session_state.trained_models.items():
                            performance_data.append({
                                'Year': year,
                                'Accuracy': model_info['accuracy'],
                                'Features': len(model_info['feature_cols'])
                            })
                        
                        perf_df = pd.DataFrame(performance_data)
                        st.dataframe(perf_df.round(3))
                        
                except Exception as e:
                    st.markdown(f'<div class="error-message">❌ Error training models: {e}</div>', unsafe_allow_html=True)

# --- Enhanced Step 5: Classification (Multi-year support) ---
elif current_step == 5:  # Classification
    if not st.session_state.trained_models:
        st.markdown('<div class="warning-message">⚠️ Please train models first.</div>', unsafe_allow_html=True)
    else:
        years = sorted(st.session_state.trained_models.keys())
        
        st.markdown("""
        <div class="feature-card">
            <h3>🔮 Multi-Year Land Cover Classification</h3>
            <p>Apply trained models to classify land cover across different years.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Year selection for classification
        classification_years = st.multiselect("Select years to classify:", years, default=years)
        
        if st.button("🚀 Classify Land Cover", type="primary"):
            st.session_state.classified_data = {}
            
            for year in classification_years:
                try:
                    with st.spinner(f"🔄 Classifying {year}..."):
                        model_info = st.session_state.trained_models[year]
                        feature_data = st.session_state.feature_data[year]
                        
                        X = feature_data[model_info['feature_cols']].fillna(0)
                        X_scaled = model_info['scaler'].transform(X)
                        predictions = model_info['model'].predict(X_scaled)
                        prediction_probs = model_info['model'].predict_proba(X_scaled)
                        
                        predicted_labels = model_info['label_encoder'].inverse_transform(predictions)
                        
                        classified_df = feature_data.copy()
                        classified_df['predicted_class'] = predicted_labels
                        classified_df['prediction_confidence'] = prediction_probs.max(axis=1)
                        classified_df['year'] = year
                        
                        st.session_state.classified_data[year] = classified_df
                        
                        st.success(f"✅ {year}: {len(classified_df)} pixels classified")
                
                except Exception as e:
                    st.error(f"Error classifying {year}: {str(e)}")
            
            if st.session_state.classified_data:
                st.markdown('<div class="success-message">✅ Multi-year classification completed!</div>', unsafe_allow_html=True)

# --- Enhanced Step 6: Results (Change detection) ---
elif current_step == 6:  # Results
    if not st.session_state.classified_data:
        st.markdown('<div class="warning-message">⚠️ Please complete the classification process first.</div>', unsafe_allow_html=True)
    else:
        years = sorted(st.session_state.classified_data.keys())
        
        st.markdown("""
        <div class="feature-card">
            <h3>📊 Multi-Year Results & Change Detection</h3>
            <p>Analyze land cover changes across different years.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Year selection for comparison
        if len(years) >= 2:
            col1, col2 = st.columns(2)
            with col1:
                year1 = st.selectbox("Base Year:", years, index=0)
            with col2:
                year2 = st.selectbox("Comparison Year:", [y for y in years if y != year1], 
                                   index=min(1, len(years)-1))
            
            if st.button("🔄 Analyze Changes", type="primary"):
                st.session_state.comparison_results = compare_land_cover_changes(
                    st.session_state.classified_data[year1],
                    st.session_state.classified_data[year2],
                    year1, year2
                )
        
        # Display results
        tab1, tab2, tab3 = st.tabs(["📈 Annual Results", "🔄 Change Analysis", "📊 Summary Statistics"])
        
        with tab1:
            # Individual year results
            selected_year = st.selectbox("Select year to view:", years)
            if selected_year in st.session_state.classified_data:
                df = st.session_state.classified_data[selected_year]
                
                # Display year-specific results
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"""<div class="metric-card">
                        <div class="metric-number">{len(df)}</div>
                        <div class="metric-label">Pixels ({selected_year})</div>
                    </div>""", unsafe_allow_html=True)
                with col2:
                    st.markdown(f"""<div class="metric-card">
                        <div class="metric-number">{df['predicted_class'].nunique()}</div>
                        <div class="metric-label">Classes</div>
                    </div>""", unsafe_allow_html=True)
                with col3:
                    avg_conf = df['prediction_confidence'].mean()
                    st.markdown(f"""<div class="metric-card">
                        <div class="metric-number">{avg_conf:.3f}</div>
                        <div class="metric-label">Avg Confidence</div>
                    </div>""", unsafe_allow_html=True)
                
                # Class distribution for selected year
                class_counts = df['predicted_class'].value_counts()
                fig, ax = plt.subplots(figsize=(10, 6))
                class_counts.plot(kind='bar', ax=ax, color=plt.cm.Set3(np.linspace(0, 1, len(class_counts))))
                ax.set_title(f'Land Cover Distribution - {selected_year}')
                ax.set_xlabel('Land Cover Class')
                ax.set_ylabel('Number of Pixels')
                plt.xticks(rotation=45)
                st.pyplot(fig)
        
        with tab2:
            # Change detection results
            if st.session_state.comparison_results is not None:
                changes = st.session_state.comparison_results
                
                st.markdown(f"""
                <div class="feature-card">
                    <h3>🔄 Land Cover Changes: {year1} → {year2}</h3>
                    <p>Detected {len(changes)} significant changes</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Change statistics
                change_counts = changes['change_type'].value_counts()
                fig, ax = plt.subplots(figsize=(12, 6))
                change_counts.plot(kind='bar', ax=ax, color=plt.cm.RdYlBu_r(np.linspace(0, 1, len(change_counts))))
                ax.set_title('Land Cover Change Types')
                ax.set_xlabel('Change Type')
                ax.set_ylabel('Number of Changes')
                plt.xticks(rotation=45)
                st.pyplot(fig)
                
                # Spatial visualization of changes
                if 'longitude' in changes.columns:
                    fig = px.scatter_mapbox(
                        changes,
                        lat='latitude',
                        lon='longitude',
                        color='change_type',
                        zoom=10,
                        height=600,
                        title=f"Land Cover Changes: {year1} to {year2}"
                    )
                    fig.update_layout(mapbox_style="open-street-map")
                    st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            # Multi-year summary
            summary_data = []
            for year in years:
                df = st.session_state.classified_data[year]
                class_dist = df['predicted_class'].value_counts(normalize=True)
                summary = {'Year': year, 'Total_Pixels': len(df)}
                for class_name in class_dist.index:
                    summary[class_name] = class_dist[class_name] * 100
                summary_data.append(summary)
            
            summary_df = pd.DataFrame(summary_data).round(2)
            st.dataframe(summary_df)

# ... (rest of your code remains similar) ...

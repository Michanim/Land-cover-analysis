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
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans, DBSCAN
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from streamlit_option_menu import option_menu  # ✅ NEW for top navigation

# Earth Engine
try:
    import ee
    import geemap
    EE_AVAILABLE = True
except ImportError:
    EE_AVAILABLE = False
    st.warning("Google Earth Engine not available. Install with: pip install earthengine-api geemap")

# --- App Config ---
st.set_page_config(
    page_title="Sentinel-2 Land Cover Analysis (Nigeria)",
    page_icon="🌍",
    layout="wide"
)

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

    /* Inputs */
    .stSelectbox, .stRadio, .stSlider, .stTextInput {
        background-color: #1e1e1e !important;
        color: white !important;
        border-radius: 5px;
    }

    /* Info, warning, success, error boxes */
    .info-box {
        background-color: #1e1e1e;
        color: #e1e1e1;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #1f6feb;
    }
    .success-box {
        background-color: #1a3322;
        border-left: 4px solid #28a745;
    }
    .warning-box {
        background-color: #332b00;
        border-left: 4px solid #ffc107;
    }
    .error-box {
        background-color: #331b1b;
        border-left: 4px solid #dc3545;
    }

    /* Hide sidebar */
    section[data-testid="stSidebar"] {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown("""
# 🌍 Sentinel-2 Land Cover Analysis (Nigeria)
**Upload a GeoJSON, download satellite data, and classify land cover automatically.**
""")

# --- SentinelDataProcessor Class ---
class SentinelDataProcessor:
    def __init__(self):
        self.data = None
        self.models = {}
        self.scaler = StandardScaler()
        self.model_path = "models"
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

    def authenticate_earth_engine(self, service_account_key=None):
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

    def load_geojson(self, geojson_data, crs_option="auto", manual_crs=None):
        try:
            if isinstance(geojson_data, str):
                gdf = gpd.read_file(geojson_data)
            elif isinstance(geojson_data, gpd.GeoDataFrame):
                gdf = geojson_data.copy()
            else:
                try:
                    gdf = gpd.GeoDataFrame.from_features(geojson_data['features'])
                except Exception as feature_error:
                    st.error(f"Error parsing GeoJSON features: {feature_error}")
                    return None

            if gdf.empty:
                st.error("No valid features found in the file")
                return None

            # --- CRS handling for Nigeria ---
            if gdf.crs is None:
                if crs_option == "force_utm32n":
                    st.info("No CRS specified. Forcing UTM Zone 32N (EPSG:32632).")
                    gdf.set_crs("EPSG:32632", allow_override=True)
                else:
                    st.warning("No CRS specified. Assuming WGS84 (EPSG:4326).")
                    gdf.set_crs("EPSG:4326", allow_override=True)

            if crs_option == "preserve":
                st.info(f"Preserving original CRS: {gdf.crs}")
            elif crs_option == "force_wgs84":
                gdf = gdf.to_crs("EPSG:4326")
            elif crs_option == "force_utm32n":
                gdf = gdf.to_crs("EPSG:32632")
            elif crs_option == "manual" and manual_crs:
                gdf = gdf.to_crs(manual_crs)
            else:
                if gdf.crs.to_string() != "EPSG:4326":
                    gdf = gdf.to_crs("EPSG:4326")

            if (~gdf.geometry.is_valid).any():
                st.warning("Found invalid geometries. Fixing...")
                gdf = gdf[gdf.geometry.is_valid]

            st.success(f"Successfully loaded {len(gdf)} features")
            return gdf
        except Exception as e:
            st.error(f"Error loading GeoJSON: {str(e)}")
            return None

    def download_sentinel2_data(self, geometry, start_date, end_date, cloud_cover=20):
        try:
            ee_geometry = ee.Geometry(geometry.__geo_interface__, proj='EPSG:4326')
            collection = ee.ImageCollection('COPERNICUS/S2_SR') \
                .filterBounds(ee_geometry) \
                .filterDate(start_date, end_date) \
                .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_cover))

            if collection.size().getInfo() == 0:
                st.warning("No Sentinel-2 images found.")
                return None, 0

            image = collection.median().clip(ee_geometry)
            bands = ['B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12']
            image = image.select(bands)
            image = self.add_spectral_indices(image)
            return image, collection.size().getInfo()
        except Exception as e:
            st.error(f"Error downloading Sentinel-2 data: {str(e)}")
            return None, 0

    def add_spectral_indices(self, image):
        ndvi = image.normalizedDifference(['B8','B4']).rename('NDVI')
        ndwi = image.normalizedDifference(['B3','B8']).rename('NDWI')
        ndbi = image.normalizedDifference(['B11','B8']).rename('NDBI')
        evi = image.expression(
            '2.5*((NIR-RED)/(NIR+6*RED-7.5*BLUE+1))',{
                'NIR': image.select('B8'),
                'RED': image.select('B4'),
                'BLUE': image.select('B2')
            }).rename('EVI')
        return image.addBands([ndvi,ndwi,ndbi,evi])

    def extract_features(self, image, geometry, scale=10):
        try:
            ee_geometry = ee.Geometry(geometry.__geo_interface__, proj='EPSG:4326')
            samples = image.sample(region=ee_geometry, scale=scale, numPixels=5000, geometries=True)
            features = samples.getInfo()['features']
            df = pd.DataFrame([f['properties'] for f in features])
            df['longitude'] = [f['geometry']['coordinates'][0] for f in features]
            df['latitude'] = [f['geometry']['coordinates'][1] for f in features]
            return df
        except Exception as e:
            st.error(f"Error extracting features: {str(e)}")
            return None

    def prepare_training_data(self, df, label_column=None):
        feature_cols = [col for col in df.columns if col not in ['longitude','latitude',label_column]]
        X = df[feature_cols].select_dtypes(include=[np.number]).fillna(0)
        if label_column:
            y = df[label_column]
            return X, y, feature_cols
        return X, None, feature_cols

    def train_supervised_model(self, X, y, model_type='random_forest'):
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        if model_type=='random_forest':
            model = RandomForestClassifier(n_estimators=100,random_state=42)
        else:
            model = SVC(kernel='rbf',random_state=42)
        model.fit(X_train_scaled,y_train)
        y_pred = model.predict(X_test_scaled)
        acc = accuracy_score(y_test,y_pred)
        self.models[model_type] = model
        return model,acc,y_test,y_pred

    def train_unsupervised_model(self,X,algorithm='kmeans',n_clusters=5):
        X_scaled = self.scaler.fit_transform(X)
        if algorithm=='kmeans':
            model = KMeans(n_clusters=n_clusters,random_state=42)
        else:
            model = DBSCAN(eps=0.5,min_samples=5)
        clusters = model.fit_predict(X_scaled)
        self.models[algorithm] = model
        return model,clusters

    def save_model(self,model_name):
        try:
            joblib.dump(self.models[model_name],f"{self.model_path}/{model_name}.pkl")
            st.success(f"Model saved as {model_name}.pkl")
        except Exception as e:
            st.error(f"Error saving model: {e}")

    def load_model(self,model_name):
        try:
            model = joblib.load(f"{self.model_path}/{model_name}.pkl")
            self.models[model_name]=model
            st.success(f"Model {model_name} loaded successfully!")
        except Exception as e:
            st.warning(f"Could not load model {model_name}: {e}")

# --- Main App ---
def main():
    processor = SentinelDataProcessor()
    for model_name in ['random_forest','kmeans']:
        processor.load_model(model_name)

    # --- Top Navigation ---
    selected = option_menu(
        menu_title=None,
        options=["Data Upload","Data Download","Visualization","Model Training","Classification","Results"],
        icons=["upload","download","bar-chart","robot","map","clipboard-data"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
        styles={
            "container": {"padding":"0!important","background-color":"#1e1e1e"},
            "icon": {"color":"white","font-size":"18px"},
            "nav-link": {"color":"white","font-size":"16px","text-align":"center","margin":"0px","--hover-color":"#333"},
            "nav-link-selected": {"background-color":"#1f6feb"},
        }
    )
    page = selected

    # --- Pages ---
    if page=="Data Upload":
        st.header("📁 Upload GeoJSON (Nigeria)")
        crs_option = st.radio("CRS Handling:",["Auto-detect (convert to WGS84)","Preserve original CRS","Force WGS84","Force UTM Zone 32N (Nigeria)","Manual CRS"])
        manual_crs=None
        if crs_option=="Manual CRS":
            manual_crs=st.text_input("Enter EPSG code (e.g., 32632):",placeholder="32632")

        uploaded_file = st.file_uploader("Choose a GeoJSON file",type=['geojson','json'])
        if uploaded_file:
            try:
                geojson_data=json.load(uploaded_file)
                gdf=processor.load_geojson(geojson_data,crs_option.lower(),f"EPSG:{manual_crs}" if manual_crs else None)
                if gdf is not None:
                    st.session_state['gdf']=gdf
                    st.success("GeoJSON loaded successfully!")
                    m=folium.Map(location=[gdf.geometry.centroid.y.mean(),gdf.geometry.centroid.x.mean()],zoom_start=10)
                    folium.GeoJson(gdf).add_to(m)
                    folium_static(m)
            except Exception as e:
                st.error(f"Error: {str(e)}")

    elif page=="Data Download":
        st.header("🛰️ Download Sentinel-2 Data (Nigeria)")
        if not EE_AVAILABLE:
            st.error("Google Earth Engine not available.")
        elif 'gdf' in st.session_state:
            col1,col2=st.columns(2)
            with col1:
                start_date=st.date_input("Start Date",datetime.now()-timedelta(days=90))
                cloud_cover=st.slider("Max Cloud Cover (%)",0,100,20)
            with col2:
                end_date=st.date_input("End Date",datetime.now())
                scale=st.slider("Spatial Resolution (m)",10,60,10)

            if st.button("Download Data"):
                with st.spinner("Downloading..."):
                    image,count=processor.download_sentinel2_data(
                        st.session_state['gdf'].geometry.unary_union,
                        start_date.strftime('%Y-%m-%d'),
                        end_date.strftime('%Y-%m-%d'),
                        cloud_cover
                    )
                    if image:
                        st.session_state['sentinel_image']=image
                        st.success(f"Downloaded {count} images!")
                        df=processor.extract_features(image,st.session_state['gdf'].geometry.unary_union)
                        if df is not None:
                            st.session_state['features_df']=df
                            st.success(f"Extracted {len(df)} features!")
        else:
            st.warning("Please upload a GeoJSON first.")

    elif page=="Visualization":
        st.header("📊 Data Visualization")
        if 'features_df' in st.session_state:
            df=st.session_state['features_df']
            numeric_cols=df.select_dtypes(include=[np.number]).columns
            plot_type=st.selectbox("Plot Type",["Scatter","Histogram","Box Plot","Spectral Signature"])
            if plot_type=="Scatter":
                x_axis=st.selectbox("X-axis",numeric_cols)
                y_axis=st.selectbox("Y-axis",numeric_cols)
                fig=px.scatter(df,x=x_axis,y=y_axis)
                st.plotly_chart(fig)
            elif plot_type=="Histogram":
                feature=st.selectbox("Feature",numeric_cols)
                fig=px.histogram(df,x=feature)
                st.plotly_chart(fig)
            elif plot_type=="Box Plot":
                features=st.multiselect("Features",numeric_cols,default=numeric_cols[:3])
                fig=px.box(df[features].melt(),y='value',x='variable')
                st.plotly_chart(fig)
            elif plot_type=="Spectral Signature":
                band_cols=[col for col in df.columns if col.startswith('B')]
                fig=go.Figure()
                fig.add_trace(go.Scatter(x=band_cols,y=df[band_cols].mean(),mode='lines+markers'))
                st.plotly_chart(fig)
        else:
            st.warning("Please download data first.")

    elif page=="Model Training":
        st.header("🤖 Train ML Model")
        if 'features_df' in st.session_state:
            df=st.session_state['features_df']
            tab1,tab2=st.tabs(["Supervised","Unsupervised"])
            with tab1:
                label_col=st.selectbox("Label Column",['None']+list(df.columns))
                if label_col!='None':
                    model_type=st.selectbox("Model",["Random Forest","SVM"])
                    if st.button("Train Model"):
                        X,y,_=processor.prepare_training_data(df,label_col)
                        model,acc,_,_=processor.train_supervised_model(X,y,model_type.lower().replace(" ","_"))
                        st.success(f"Model trained! Accuracy: {acc:.2f}")
                        if st.button("Save Model"):
                            processor.save_model(model_type.lower().replace(" ","_"))
            with tab2:
                algorithm=st.selectbox("Algorithm",["K-Means","DBSCAN"])
                n_clusters=st.slider("Clusters",2,10,5) if algorithm=="K-Means" else None
                if st.button("Cluster Data"):
                    X,_,_=processor.prepare_training_data(df)
                    model,clusters=processor.train_unsupervised_model(X,algorithm.lower(),n_clusters)
                    df['cluster']=clusters
                    st.session_state['clustered_df']=df
                    st.success("Clustering complete!")
                    if st.button("Save Model"):
                        processor.save_model(algorithm.lower())
        else:
            st.warning("Please download data first.")

    elif page=="Classification":
        st.header("🗺️ Land Cover Classification")
        if 'features_df' in st.session_state and processor.models:
            model_name=st.selectbox("Select Model",list(processor.models.keys()))
            if st.button("Classify"):
                df=st.session_state['features_df']
                X,_,_=processor.prepare_training_data(df)
                X_scaled=processor.scaler.transform(X)
                predictions=processor.models[model_name].predict(X_scaled)
                df['predicted_class']=predictions
                st.session_state['classified_df']=df
                st.success("Classification complete!")
                fig=px.scatter_mapbox(df,lat='latitude',lon='longitude',color='predicted_class',
                                      mapbox_style="open-street-map",zoom=10)
                st.plotly_chart(fig)
        else:
            st.warning("Please train a model first.")

    elif page=="Results":
        st.header("📈 Results & Export")
        if 'classified_df' in st.session_state:
            df=st.session_state['classified_df']
            st.dataframe(df.head())
            if st.button("Export CSV"):
                csv=df.to_csv(index=False)
                st.download_button("Download",csv,"classification.csv","text/csv")
        else:
            st.warning("No results to export.")

# --- Run App ---
if __name__=="__main__":
    main()


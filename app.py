import os
from pathlib import Path
import streamlit as st
import ee
import json
import geopandas as gpd
import tempfile
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
from typing import Dict, Optional, Any, List, Union, Tuple

# Peta
import folium
from streamlit_folium import st_folium
from folium.plugins import Draw

# =========================
# 0) PAGE CONFIG
# =========================
Coord = Union[float, int]
Pair = List[Coord]
LAT_MIN, LAT_MAX = -11.5, 6.5
LON_MIN, LON_MAX = 94.0, 141.5
st.set_page_config(page_title="GEE Land Cover & Vegetation Analysis", layout="wide", initial_sidebar_state="expanded")

# Custom CSS untuk layout yang lebih baik
st.markdown("""
<style>
    .main > div {
        padding-left: 1rem;
        padding-right: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
    div[data-testid="column"] {
        padding: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# =========================
# 1) AUTH / INITIALIZATION
# =========================
SERVICE_ACCOUNT = "geomoka@endless-bounty-416008.iam.gserviceaccount.com"
KEY_FILE = os.getenv("GEE_KEY_FILE", "endless-bounty-416008-a6cce2f8b208.json")

def init_ee():
    """Initialize Google Earth Engine"""
    if Path(KEY_FILE).exists():
        try:
            credentials = ee.ServiceAccountCredentials(SERVICE_ACCOUNT, KEY_FILE)
            ee.Initialize(credentials)
            st.success("✔ Earth Engine initialized with local Service Account credentials.")
            return
        except Exception as e:
            st.error(f"Gagal inisialisasi GEE dari file lokal: {e}")

    if "gcp_service_account" in st.secrets:
        try:
            from google.oauth2 import service_account
            sa_info = st.secrets["gcp_service_account"]
            creds = service_account.Credentials.from_service_account_info(sa_info)
            ee.Initialize(creds)
            st.success("✔ Earth Engine initialized via Streamlit secrets.")
            return
        except Exception as e:
            st.error(f"Gagal inisialisasi GEE dari secrets: {e}")

    st.error("Tidak menemukan kredensial GEE yang valid.")
    st.stop()

# =========================
# 1B) INDONESIA ADMIN API
# =========================
API_BASE_URL = "https://api.sp3stab.id/api/en"

@st.cache_data(ttl=3600)
def fetch_region_dropdown(endpoint: str, parent_code: Optional[str] = None) -> Dict:
    """Fetch region list from API for dropdown"""
    try:
        params = {"is_for_dropdown": 1}
        if parent_code:
            params["parent_code"] = parent_code
        
        response = requests.get(f"{API_BASE_URL}/{endpoint}", params=params, timeout=10)
        response.raise_for_status()
        
        # API returns direct dictionary for dropdown
        # Example: {"ACEH": "11", "BALI": "51", ...}
        data = response.json()
        
        # Check if it's the expected format
        if isinstance(data, dict):
            return data
        
        return {}
    except Exception as e:
        st.error(f"Error fetching {endpoint}: {e}")
        return {}

def looks_like_latlon(pair: Pair) -> bool:
    """Kembalikan True jika pasangan tampak [lat, lon] (kebalik) untuk Indonesia."""
    if not isinstance(pair, (list, tuple)) or len(pair) < 2:
        return False
    a, b = pair[0], pair[1]
    # Pastikan numeric
    try:
        a = float(a); b = float(b)
    except Exception:
        return False
    return (LAT_MIN <= a <= LAT_MAX) and (LON_MIN <= b <= LON_MAX)

def swap_xy_in_coords(obj: Any) -> Any:
    """
    Rekursif menukar [lat, lon] -> [lon, lat] pada struktur koordinat GeoJSON.
    Tidak akan menukar kalau tidak perlu (berdasarkan sampel deteksi).
    """
    if isinstance(obj, (list, tuple)):
        if len(obj) >= 2 and all(isinstance(x, (int, float, str)) for x in obj[:2]):
            # Ini kandidat titik [*, *]
            # Deteksi apakah [lat, lon]; jika ya, tukar
            if looks_like_latlon(obj):
                a, b = float(obj[0]), float(obj[1])
                rest = [float(x) for x in obj[2:]] if len(obj) > 2 else []
                return [b, a, *rest]  # [lon, lat, (z, m, ...)]
            else:
                # Tidak tampak [lat, lon] (mungkin sudah [lon, lat]); biarkan
                return [float(x) if isinstance(x, str) else x for x in obj]
        else:
            # Turun rekursif (rings, lines, multiparts)
            return [swap_xy_in_coords(x) for x in obj]
    elif isinstance(obj, dict):
        # Pegang kasus tak lazim, umumnya koordinat berupa list
        return {k: swap_xy_in_coords(v) if k != "type" else v for k, v in obj.items()}
    else:
        return obj

def swap_geometry(feature: Dict) -> Dict:
    geom = feature.get("geometry")
    if not geom or "type" not in geom or "coordinates" not in geom:
        return feature

    gtype = geom["type"]
    coords = geom["coordinates"]

    # Kita hanya perlu swap di koordinat; tipe apapun diproses rekursif
    new_coords = swap_xy_in_coords(coords)
    feature["geometry"] = {"type": gtype, "coordinates": new_coords}
    return feature

def swap_featurecollection(fc: Dict) -> Dict:
    if not isinstance(fc, dict) or fc.get("type") != "FeatureCollection":
        return fc
    feats = fc.get("features", [])
    fc["features"] = [swap_geometry(f) for f in feats]
    return fc

@st.cache_data(ttl=3600)
def fetch_region_geometry(endpoint: str, code: str) -> Optional[Dict]:
    """Fetch region geometry from API, lalu betulkan urutan lat-lon -> lon-lat bila perlu."""
    try:
        response = requests.get(f"{API_BASE_URL}/{endpoint}", params={"code": code}, timeout=15)
        response.raise_for_status()
        result = response.json()

        if result.get("meta", {}).get("code") == 200:
            region_data = result.get("data", {}).get("region")
            if region_data and "features" in region_data:
                # 🔧 langkah pembalikan koordinat (kondisional via heuristik)
                region_data = swap_featurecollection(region_data)
                return region_data

        return None
    except Exception as e:
        st.error(f"Error fetching geometry for {code}: {e}")
        return None

def geojson_to_ee_geometry(geojson_data: Dict) -> Tuple[Optional[ee.Geometry], Optional[gpd.GeoDataFrame]]:
    """Convert GeoJSON from API to EE Geometry and GeoDataFrame"""
    try:
        if not geojson_data or "features" not in geojson_data:
            return None, None
        
        # Convert to GeoDataFrame
        gdf = gpd.GeoDataFrame.from_features(geojson_data["features"])
        
        if gdf.empty:
            return None, None
        
        # Ensure WGS84
        if gdf.crs is None:
            gdf.set_crs("EPSG:4326", inplace=True)
        elif gdf.crs != "EPSG:4326":
            gdf = gdf.to_crs("EPSG:4326")
        
        # Convert to EE Geometry
        feature = geojson_data["features"][0]
        ee_geom = ee.Geometry(feature["geometry"])
        
        return ee_geom, gdf
        
    except Exception as e:
        st.error(f"Error converting geometry: {e}")
        return None, None

# =========================
# 2) HELPER: add ee layer
# =========================
def add_ee_layer(f_map, ee_object, vis_params, name):
    try:
        map_id = ee_object.getMapId(vis_params)
        folium.raster_layers.TileLayer(
            tiles=map_id["tile_fetcher"].url_format,
            attr='Map © Google Earth Engine',
            name=name,
            overlay=True,
            control=True,
        ).add_to(f_map)
    except Exception as e:
        st.error(f"Gagal menambahkan layer '{name}': {e}")

folium.Map.add_ee_layer = add_ee_layer

# =========================
# 3) HELPER: Convert uploaded file to GEE geometry
# =========================
def file_to_ee_geometry(uploaded_file):
    """Convert uploaded GeoJSON/Shapefile to Earth Engine geometry"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        if uploaded_file.name.endswith('.geojson') or uploaded_file.name.endswith('.json'):
            gdf = gpd.read_file(tmp_path)
        elif uploaded_file.name.endswith('.zip'):
            gdf = gpd.read_file(f"zip://{tmp_path}")
        else:
            gdf = gpd.read_file(tmp_path)
        
        if gdf.crs and gdf.crs != 'EPSG:4326':
            gdf = gdf.to_crs('EPSG:4326')
        
        union_geom = gdf.geometry.unary_union
        geojson = json.loads(gpd.GeoSeries([union_geom]).to_json())
        ee_geom = ee.Geometry(geojson['features'][0]['geometry'])
        
        os.unlink(tmp_path)
        return ee_geom, gdf
        
    except Exception as e:
        st.error(f"Error processing file: {e}")
        if 'tmp_path' in locals():
            os.unlink(tmp_path)
        return None, None

# =========================
# 4) VEGETATION INDICES
# =========================
VEGETATION_INDICES = {
    "NDVI": {
        "name": "Normalized Difference Vegetation Index",
        "formula": "(NIR - RED) / (NIR + RED)",
        "bands": ["B8", "B4"],
        "range": [-1, 1],
        "description": "Kesehatan vegetasi umum",
        "palette": ["#8B4513", "#FFFF00", "#90EE90", "#006400"]
    },
    "NDWI": {
        "name": "Normalized Difference Water Index", 
        "formula": "(GREEN - NIR) / (GREEN + NIR)",
        "bands": ["B3", "B8"],
        "range": [-1, 1],
        "description": "Kandungan air pada vegetasi",
        "palette": ["#8B4513", "#F5DEB3", "#87CEEB", "#0000FF"]
    },
    "MNDWI": {
        "name": "Modified NDWI",
        "formula": "(GREEN - SWIR) / (GREEN + SWIR)",
        "bands": ["B3", "B11"],
        "range": [-1, 1],
        "description": "Deteksi badan air",
        "palette": ["#FFFFE0", "#98FB98", "#4682B4", "#000080"]
    },
    "NDBI": {
        "name": "Normalized Difference Built-up Index",
        "formula": "(SWIR - NIR) / (SWIR + NIR)", 
        "bands": ["B11", "B8"],
        "range": [-1, 1],
        "description": "Area terbangun",
        "palette": ["#006400", "#90EE90", "#FFD700", "#FF0000"]
    },
    "EVI": {
        "name": "Enhanced Vegetation Index",
        "formula": "2.5 * (NIR - RED) / (NIR + 6*RED - 7.5*BLUE + 1)",
        "bands": ["B8", "B4", "B2"],
        "range": [-1, 1],
        "description": "Vegetasi dengan koreksi atmosfer",
        "palette": ["#8B4513", "#FFFF00", "#90EE90", "#006400"]
    },
    "SAVI": {
        "name": "Soil Adjusted Vegetation Index",
        "formula": "1.5 * (NIR - RED) / (NIR + RED + 0.5)",
        "bands": ["B8", "B4"],
        "range": [-1, 1],
        "description": "Vegetasi dengan koreksi tanah",
        "palette": ["#8B4513", "#FFFF00", "#90EE90", "#006400"]
    },
    "BSI": {
        "name": "Bare Soil Index",
        "formula": "(SWIR + RED - NIR - BLUE) / (SWIR + RED + NIR + BLUE)",
        "bands": ["B11", "B4", "B8", "B2"],
        "range": [-1, 1],
        "description": "Tanah terbuka",
        "palette": ["#006400", "#90EE90", "#DEB887", "#8B4513"]
    },
    "NDMI": {
        "name": "Normalized Difference Moisture Index",
        "formula": "(NIR - SWIR) / (NIR + SWIR)",
        "bands": ["B8", "B11"],
        "range": [-1, 1],
        "description": "Kelembaban vegetasi",
        "palette": ["#8B4513", "#D2691E", "#90EE90", "#006400"]
    }
}

def calculate_index(image, index_name):
    """Calculate vegetation index"""
    index_info = VEGETATION_INDICES[index_name]
    
    if index_name == "EVI":
        nir = image.select('B8')
        red = image.select('B4')
        blue = image.select('B2')
        evi = nir.subtract(red).multiply(2.5).divide(
            nir.add(red.multiply(6)).subtract(blue.multiply(7.5)).add(1)
        )
        return evi.rename(index_name)
    
    elif index_name == "SAVI":
        nir = image.select('B8')
        red = image.select('B4')
        savi = nir.subtract(red).divide(nir.add(red).add(0.5)).multiply(1.5)
        return savi.rename(index_name)
    
    elif index_name == "BSI":
        blue = image.select('B2')
        red = image.select('B4')
        nir = image.select('B8')
        swir = image.select('B11')
        bsi = swir.add(red).subtract(nir).subtract(blue).divide(
            swir.add(red).add(nir).add(blue)
        )
        return bsi.rename(index_name)
    
    else:
        # NDVI, NDWI, MNDWI, NDBI, NDMI
        bands = index_info["bands"]
        return image.normalizedDifference(bands).rename(index_name)

# =========================
# 5) LAND COVER LEGENDS
# =========================
LAND_COVER_LEGENDS = {
    "ESA_WorldCover": {
        "10": {"color": "#006400", "label": "Trees"},
        "20": {"color": "#ffbb22", "label": "Shrubland"},
        "30": {"color": "#ffff4c", "label": "Grassland"},
        "40": {"color": "#f096ff", "label": "Cropland"},
        "50": {"color": "#fa0000", "label": "Built-up"},
        "60": {"color": "#b4b4b4", "label": "Barren/sparse vegetation"},
        "70": {"color": "#f0f0f0", "label": "Snow and ice"},
        "80": {"color": "#0032c8", "label": "Open water"},
        "90": {"color": "#0096a0", "label": "Herbaceous wetland"},
        "95": {"color": "#00cf75", "label": "Mangroves"},
        "100": {"color": "#fae6a0", "label": "Moss and lichen"}
    },
    "ESRI_LandCover": {
        "1": {"color": "#1A5BAB", "label": "Water"},
        "2": {"color": "#358221", "label": "Trees"},
        "3": {"color": "#A7D282", "label": "Grass"},
        "4": {"color": "#87D19E", "label": "Flooded Vegetation"},
        "5": {"color": "#FFDB5C", "label": "Crops"},
        "6": {"color": "#EECFA8", "label": "Scrub/Shrub"},
        "7": {"color": "#ED022A", "label": "Built Area"},
        "8": {"color": "#EDE9E4", "label": "Bare Ground"},
        "9": {"color": "#F2FAFF", "label": "Snow/Ice"},
        "10": {"color": "#C8C8C8", "label": "Clouds"}
    },
    "Dynamic_World": {
        "0": {"color": "#419BDF", "label": "Water"},
        "1": {"color": "#397D49", "label": "Trees"},
        "2": {"color": "#88B053", "label": "Grass"},
        "3": {"color": "#7A87C6", "label": "Flooded vegetation"},
        "4": {"color": "#E49635", "label": "Crops"},
        "5": {"color": "#DFC35A", "label": "Shrub & Scrub"},
        "6": {"color": "#C4281B", "label": "Built Area"},
        "7": {"color": "#A59B8F", "label": "Bare ground"},
        "8": {"color": "#B39FE1", "label": "Snow & Ice"}
    }
}

def add_legend_to_map(f_map, legend_dict, title):
    """Add legend to folium map with black text"""
    legend_html = f'''
    <div style="position: fixed; 
                bottom: 50px; right: 50px; width: 220px; height: auto; 
                background-color: white; z-index: 1000; 
                border: 2px solid grey; border-radius: 5px;
                font-size: 14px; box-shadow: 2px 2px 5px rgba(0,0,0,0.3);">
        <p style="margin: 10px; font-weight: bold; color: black;">{title}</p>
    '''
    
    for key, value in legend_dict.items():
        legend_html += f'''
        <p style="margin: 5px 10px; color: black;">
            <span style="background-color: {value['color']}; 
                        width: 20px; height: 12px; 
                        display: inline-block; margin-right: 8px;
                        border: 1px solid #666;">
            </span>
            {value['label']}
        </p>
        '''
    
    legend_html += '</div>'
    f_map.get_root().html.add_child(folium.Element(legend_html))

# =========================
# 6) DYNAMIC WORLD FUNCTIONS
# =========================
def create_dynamic_world_composite(aoi, start_date, end_date, mode='mode'):
    """Create Dynamic World composite"""
    dw = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1').filterDate(start_date, end_date).filterBounds(aoi)
    
    if mode == 'mode':
        classification = dw.select('label')
        return classification.reduce(ee.Reducer.mode()).clip(aoi)
    elif mode == 'probability':
        return dw.select(['water', 'trees', 'grass', 'flooded_vegetation', 
                         'crops', 'shrub_and_scrub', 'built', 'bare', 'snow_and_ice']).mean().clip(aoi)
    else:
        return create_hillshade_rgb(dw, aoi)

def create_hillshade_rgb(dw_col, aoi):
    """Create hillshade RGB visualization for Dynamic World"""
    VIS_PALETTE = [
        '419bdf', '397d49', '88b053', '7a87c6', 
        'e49635', 'dfc35a', 'c4281b', 'a59b8f', 'b39fe1'
    ]
    
    classification = dw_col.select('label').mode()
    
    prob_bands = dw_col.select([
        'water', 'trees', 'grass', 'flooded_vegetation', 
        'crops', 'shrub_and_scrub', 'built', 'bare', 'snow_and_ice'
    ]).mean()
    
    max_prob = prob_bands.reduce(ee.Reducer.max())
    hillshade = ee.Terrain.hillshade(max_prob.multiply(100)).divide(255)
    rgbImage = classification.visualize(min=0, max=8, palette=VIS_PALETTE)
    
    return rgbImage.multiply(hillshade).clip(aoi)

# =========================
# 7) UI / SIDEBAR
# =========================
st.title("🌍 Google Earth Engine • Land Cover & Vegetation Analysis")

with st.sidebar:
    st.header("⚙️ Pengaturan")
    st.caption("Auth: Service Account lokal → (cadangan) secrets.")
    
    analysis_type = st.selectbox(
        "Tipe Analisis",
        ["Vegetation Indices Analysis", "Land Cover Analysis", "Combined Analysis"]
    )
    
    year = st.slider("Tahun", 2017, 2025, 2022)
    zoom0 = st.slider("Zoom awal peta", 2, 12, 10)
    
    if analysis_type in ["Vegetation Indices Analysis", "Combined Analysis"]:
        st.subheader("Parameter Sentinel-2")
        months = st.select_slider("Rentang bulan", options=list(range(1, 13)), value=(6, 9))
        cloud_threshold = st.slider("Ambang awan (%)", 0, 100, 40)
        
        if cloud_threshold > 80:
            st.warning("⚠️ Ambang awan tinggi dapat mempengaruhi kualitas")
        
        st.subheader("Indeks Vegetasi")
        selected_indices = st.multiselect(
            "Pilih indeks vegetasi",
            list(VEGETATION_INDICES.keys()),
            default=["NDVI", "NDWI", "MNDWI", "NDBI"]
        )
    
    if analysis_type in ["Land Cover Analysis", "Combined Analysis"]:
        st.subheader("Parameter Land Cover")
        land_cover_datasets = st.multiselect(
            "Dataset Land Cover",
            ["Dynamic World", "ESA WorldCover", "ESRI Land Cover"],
            default=["Dynamic World"]
        )
        
        if "Dynamic World" in land_cover_datasets:
            dw_mode = st.selectbox(
                "Dynamic World Mode",
                ["mode", "hillshade", "probability"],
                help="mode: Most common class, hillshade: Top-1 probability with hillshade, probability: Mean probability bands"
            )

# Init EE
with st.status("🔄 Initializing Earth Engine...", expanded=False) as status:
    init_ee()
    status.update(label="✅ Earth Engine Ready", state="complete")

# =========================
# 8) AOI (Area of Interest) WITH INDONESIA ADMIN
# =========================
st.subheader("📍 Pilih Area of Interest (AOI)")

# Initialize session state for region selection
if 'selected_province' not in st.session_state:
    st.session_state.selected_province = None
if 'selected_city' not in st.session_state:
    st.session_state.selected_city = None
if 'selected_district' not in st.session_state:
    st.session_state.selected_district = None
if 'selected_village' not in st.session_state:
    st.session_state.selected_village = None

col1, col2 = st.columns([1, 3])

with col1:
    aoi_mode = st.radio(
        "Mode AOI", 
        ["Batas Wilayah Indonesia", "Upload File (GeoJSON/SHP)", "Koordinat & Buffer", "Gambar di Peta (Poligon)"], 
        index=0
    )
    
    aoi = None
    preview_gdf = None
    region_name = ""
    
    # ========== BATAS WILAYAH INDONESIA ==========
    if aoi_mode == "Batas Wilayah Indonesia":
        st.info("🇮🇩 Pilih wilayah administratif Indonesia")
        
        # Level 1: Provinsi
        with st.spinner("Loading provinsi..."):
            provinces_data = fetch_region_dropdown("province")
        
        if provinces_data:
            province_names = list(provinces_data.keys())
            selected_province_name = st.selectbox(
                "Provinsi",
                ["-- Pilih Provinsi --"] + province_names,
                key="province_select"
            )
            
            if selected_province_name != "-- Pilih Provinsi --":
                province_code = provinces_data[selected_province_name]
                st.session_state.selected_province = {
                    "name": selected_province_name,
                    "code": province_code
                }
                
                # Level 2: Kabupaten/Kota
                with st.spinner("Loading kabupaten/kota..."):
                    cities_data = fetch_region_dropdown("city", province_code)
                
                if cities_data:
                    city_names = list(cities_data.keys())
                    selected_city_name = st.selectbox(
                        "Kabupaten/Kota",
                        ["-- Pilih Kabupaten/Kota --", "-- Gunakan Provinsi --"] + city_names,
                        key="city_select"
                    )
                    
                    if selected_city_name == "-- Gunakan Provinsi --":
                        # Use province boundary
                        region_name = selected_province_name
                        with st.spinner(f"Loading geometry {region_name}..."):
                            geojson_data = fetch_region_geometry("province", province_code)
                            if geojson_data:
                                aoi, preview_gdf = geojson_to_ee_geometry(geojson_data)
                                if aoi:
                                    st.success(f"✔ Berhasil memuat: {region_name}")
                    
                    elif selected_city_name != "-- Pilih Kabupaten/Kota --":
                        city_code = cities_data[selected_city_name]
                        st.session_state.selected_city = {
                            "name": selected_city_name,
                            "code": city_code
                        }
                        
                        # Level 3: Kecamatan
                        with st.spinner("Loading kecamatan..."):
                            districts_data = fetch_region_dropdown("district", city_code)
                        
                        if districts_data:
                            district_names = list(districts_data.keys())
                            selected_district_name = st.selectbox(
                                "Kecamatan",
                                ["-- Pilih Kecamatan --", "-- Gunakan Kabupaten/Kota --"] + district_names,
                                key="district_select"
                            )
                            
                            if selected_district_name == "-- Gunakan Kabupaten/Kota --":
                                # Use city boundary
                                region_name = selected_city_name
                                with st.spinner(f"Loading geometry {region_name}..."):
                                    geojson_data = fetch_region_geometry("city", city_code)
                                    if geojson_data:
                                        aoi, preview_gdf = geojson_to_ee_geometry(geojson_data)
                                        if aoi:
                                            st.success(f"✔ Berhasil memuat: {region_name}")
                            
                            elif selected_district_name != "-- Pilih Kecamatan --":
                                district_code = districts_data[selected_district_name]
                                st.session_state.selected_district = {
                                    "name": selected_district_name,
                                    "code": district_code
                                }
                                
                                # Level 4: Kelurahan/Desa
                                with st.spinner("Loading kelurahan/desa..."):
                                    villages_data = fetch_region_dropdown("village", district_code)
                                
                                if villages_data:
                                    village_names = list(villages_data.keys())
                                    selected_village_name = st.selectbox(
                                        "Kelurahan/Desa",
                                        ["-- Pilih Kelurahan/Desa --", "-- Gunakan Kecamatan --"] + village_names,
                                        key="village_select"
                                    )
                                    
                                    if selected_village_name == "-- Gunakan Kecamatan --":
                                        # Use district boundary
                                        region_name = selected_district_name
                                        with st.spinner(f"Loading geometry {region_name}..."):
                                            geojson_data = fetch_region_geometry("district", district_code)
                                            if geojson_data:
                                                aoi, preview_gdf = geojson_to_ee_geometry(geojson_data)
                                                if aoi:
                                                    st.success(f"✔ Berhasil memuat: {region_name}")
                                    
                                    elif selected_village_name != "-- Pilih Kelurahan/Desa --":
                                        village_code = villages_data[selected_village_name]
                                        region_name = selected_village_name
                                        
                                        with st.spinner(f"Loading geometry {region_name}..."):
                                            geojson_data = fetch_region_geometry("village", village_code)
                                            if geojson_data:
                                                aoi, preview_gdf = geojson_to_ee_geometry(geojson_data)
                                                if aoi:
                                                    st.success(f"✔ Berhasil memuat: {region_name}")
    
    # ========== UPLOAD FILE ==========
    elif aoi_mode == "Upload File (GeoJSON/SHP)":
        st.info("Upload file GeoJSON atau Shapefile (zip)")
        uploaded_file = st.file_uploader(
            "Pilih file", 
            type=['geojson', 'json', 'zip', 'shp'],
            help="Untuk Shapefile, upload dalam format zip"
        )
        
        if uploaded_file is not None:
            with st.spinner("Processing file..."):
                aoi, preview_gdf = file_to_ee_geometry(uploaded_file)
                if aoi:
                    st.success(f"✔ File berhasil diproses")
                    region_name = uploaded_file.name

with col2:
    # Show attribute table
    if preview_gdf is not None:
        st.subheader(f"📊 Informasi Wilayah: {region_name}")
        
        # Display attribute table
        display_cols = [col for col in preview_gdf.columns if col != 'geometry']
        if display_cols:
            st.dataframe(
                preview_gdf[display_cols],
                use_container_width=True,
                height=200
            )
        
        # File info
        col_info1, col_info2, col_info3 = st.columns(3)
        with col_info1:
            st.metric("Jumlah Fitur", len(preview_gdf))
        with col_info2:
            st.metric("CRS", str(preview_gdf.crs))
        with col_info3:
            bounds = preview_gdf.total_bounds
            area_approx = (bounds[2] - bounds[0]) * (bounds[3] - bounds[1]) * 12100  # rough km²
            st.metric("Area (approx)", f"{area_approx:.1f} km²")
    
    elif aoi_mode == "Koordinat & Buffer":
        col_lat, col_lon, col_buffer = st.columns(3)
        with col_lat:
            lat = st.number_input("Latitude", value=-6.1754, format="%.6f")
        with col_lon:
            lon = st.number_input("Longitude", value=106.8272, format="%.6f")
        with col_buffer:
            buffer_km = st.slider("Buffer (km)", 1, 50, 10)
        
        center_geom = ee.Geometry.Point([lon, lat])
        aoi = center_geom.buffer(buffer_km * 1000).bounds()
        region_name = f"Point ({lat:.4f}, {lon:.4f})"
    
    else:  # Gambar di Peta
        st.info("💡 Gambar poligon di peta bawah, lalu klik tombol 'Gunakan AOI'")

# Map for AOI selection
st.subheader("🗺️ Peta AOI")
if aoi_mode == "Koordinat & Buffer":
    start_center = [lat, lon]
elif preview_gdf is not None:
    bounds = preview_gdf.total_bounds
    start_center = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]
else:
    start_center = [-6.1754, 106.8272]

draw_map = folium.Map(location=start_center, zoom_start=zoom0, control_scale=True)

if preview_gdf is not None:
    folium.GeoJson(
        data=preview_gdf.to_json(),
        name=f"AOI: {region_name}",
        style_function=lambda x: {
            "fillColor": "lightblue",
            "color": "blue",
            "weight": 2,
            "fillOpacity": 0.3
        },
        tooltip=region_name
    ).add_to(draw_map)

if aoi_mode == "Gambar di Peta (Poligon)":
    Draw(export=True).add_to(draw_map)

folium.LayerControl().add_to(draw_map)
m_state = st_folium(draw_map, height=400, returned_objects=["all_drawings"], use_container_width=True)

if aoi_mode == "Gambar di Peta (Poligon)":
    if st.button("Gunakan AOI dari poligon yang digambar", type="primary"):
        drawings = m_state.get("all_drawings", [])
        if drawings:
            last = drawings[-1]
            if last["geometry"]["type"].lower() in ("polygon", "multipolygon"):
                aoi = ee.Geometry(last["geometry"])
                region_name = "Custom Polygon"
                st.success("✔ AOI poligon terdeteksi.")
            else:
                st.warning("Silakan gambar **poligon**")
        else:
            st.warning("Belum ada poligon yang digambar.")

if aoi is None:
    st.warning("⚠️ Silakan pilih AOI terlebih dahulu")
    st.stop()

# =========================
# 9) ANALYSIS
# =========================
def date_range_for_year_months(y, m0, m1):
    start = f"{y}-{int(m0):02d}-01"
    end = f"{y}-12-31" if m1 == 12 else f"{y}-{int(m1)+1:02d}-01"
    return start, end

def mask_s2_clouds(img):
    qa = img.select("QA60")
    cloud = qa.bitwiseAnd(1 << 10).neq(0)
    cirrus = qa.bitwiseAnd(1 << 11).neq(0)
    mask = cloud.Or(cirrus).Not()
    return img.updateMask(mask).divide(10000)

layers = {}
vis_params = {}
index_layers = {}

# Vegetation Indices Analysis
if analysis_type in ["Vegetation Indices Analysis", "Combined Analysis"]:
    with st.status("🛰️ Processing Sentinel-2 imagery...", expanded=True) as status:
        start_date, end_date = date_range_for_year_months(year, months[0], months[1])
        
        status.write(f"📅 Date range: {start_date} to {end_date}")
        
        s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
              .filterBounds(aoi)
              .filterDate(start_date, end_date)
              .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", cloud_threshold))
              .map(mask_s2_clouds))
        
        collection_size = s2.size().getInfo()
        status.write(f"📊 Found {collection_size} Sentinel-2 images")
        
        if collection_size > 0:
            status.write("🔄 Creating median composite...")
            median_composite = s2.median().clip(aoi)
            
            layers["Sentinel-2 RGB"] = median_composite
            vis_params["Sentinel-2 RGB"] = {"bands": ["B4", "B3", "B2"], "min": 0.0, "max": 0.3}
            
            # Calculate selected indices
            status.write(f"📐 Calculating {len(selected_indices)} vegetation indices...")
            for idx_name in selected_indices:
                index_layer = calculate_index(median_composite, idx_name)
                index_layers[idx_name] = index_layer
                layers[idx_name] = index_layer
                vis_params[idx_name] = {
                    "min": VEGETATION_INDICES[idx_name]["range"][0],
                    "max": VEGETATION_INDICES[idx_name]["range"][1],
                    "palette": VEGETATION_INDICES[idx_name]["palette"]
                }
            
            status.update(label="✅ Sentinel-2 processing complete", state="complete")
        else:
            st.error(f"❌ No Sentinel-2 images found for this period. Try adjusting the date range or cloud threshold.")
            st.stop()

# Land Cover Analysis
if analysis_type in ["Land Cover Analysis", "Combined Analysis"]:
    with st.status("🗺️ Processing land cover data...", expanded=True) as status:
        if "Dynamic World" in land_cover_datasets:
            start_date = f"{year}-01-01"
            end_date = f"{year}-12-31"
            
            status.write(f"🌍 Loading Dynamic World ({dw_mode} mode)...")
            
            if dw_mode == "mode":
                dw_composite = create_dynamic_world_composite(aoi, start_date, end_date, mode='mode')
                layers["Dynamic World"] = dw_composite
                vis_params["Dynamic World"] = {
                    "min": 0,
                    "max": 8,
                    "palette": [
                        "#419BDF", "#397D49", "#88B053", "#7A87C6",
                        "#E49635", "#DFC35A", "#C4281B", "#A59B8F", "#B39FE1"
                    ]
                }
            elif dw_mode == "hillshade":
                dw_hillshade = create_dynamic_world_composite(aoi, start_date, end_date, mode='hillshade')
                layers["Dynamic World Hillshade"] = dw_hillshade
                vis_params["Dynamic World Hillshade"] = {}
            else:
                dw_prob = create_dynamic_world_composite(aoi, start_date, end_date, mode='probability')
                layers["DW Water Probability"] = dw_prob.select('water')
                vis_params["DW Water Probability"] = {"min": 0, "max": 1, "palette": ['white', 'blue']}
        
        if "ESA WorldCover" in land_cover_datasets:
            status.write("🌍 Loading ESA WorldCover...")
            esa = ee.ImageCollection("ESA/WorldCover/v100").first().clip(aoi)
            layers["ESA WorldCover"] = esa
            vis_params["ESA WorldCover"] = {"bands": ["Map"]}
        
        if "ESRI Land Cover" in land_cover_datasets:
            status.write("🌍 Loading ESRI Land Cover...")
            esri = ee.ImageCollection(
                "projects/sat-io/open-datasets/landcover/ESRI_Global-LULC_10m"
            ).mosaic().clip(aoi)
            layers["ESRI Land Cover"] = esri
            vis_params["ESRI Land Cover"] = {
                "min": 1,
                "max": 10,
                "palette": [
                    "#1A5BAB", "#358221", "#A7D282", "#87D19E", "#FFDB5C",
                    "#EECFA8", "#ED022A", "#EDE9E4", "#F2FAFF", "#C8C8C8"
                ]
            }
        
        status.update(label="✅ Land cover processing complete", state="complete")

# =========================
# 10) RESULTS MAP
# =========================
st.subheader("🗺️ Hasil Peta")

try:
    center_geom = ee.Geometry(aoi).bounds(maxError=10).centroid(maxError=10)
    lon, lat = center_geom.coordinates().getInfo()
    center_pt = [lat, lon]
except:
    center_pt = start_center

# Create tabs for different visualizations
if len(layers) > 0:
    tab_names = list(layers.keys())
    tabs = st.tabs(tab_names)
    
    for i, (layer_name, layer) in enumerate(layers.items()):
        with tabs[i]:
            col_map, col_info = st.columns([3, 1])
            
            with col_map:
                result_map = folium.Map(location=center_pt, zoom_start=zoom0, control_scale=True)
                
                # Add AOI outline
                try:
                    folium.GeoJson(
                        data=ee.FeatureCollection(ee.Feature(aoi)).getInfo(),
                        name=f"AOI: {region_name}",
                        style_function=lambda x: {"fillOpacity": 0, "weight": 2, "color": "red"},
                        tooltip=region_name
                    ).add_to(result_map)
                except:
                    pass
                
                # Add layer
                result_map.add_ee_layer(layer, vis_params[layer_name], layer_name)
                
                # Add legend for land cover layers
                if layer_name == "Dynamic World":
                    add_legend_to_map(result_map, LAND_COVER_LEGENDS["Dynamic_World"], "Dynamic World")
                elif layer_name == "ESA WorldCover":
                    add_legend_to_map(result_map, LAND_COVER_LEGENDS["ESA_WorldCover"], "ESA WorldCover")
                elif layer_name == "ESRI Land Cover":
                    add_legend_to_map(result_map, LAND_COVER_LEGENDS["ESRI_LandCover"], "ESRI Land Cover")
                
                folium.LayerControl().add_to(result_map)
                st_folium(result_map, height=600, use_container_width=True, key=f"map_{i}")
            
            with col_info:
                if layer_name in VEGETATION_INDICES:
                    st.markdown("### Info Indeks")
                    info = VEGETATION_INDICES[layer_name]
                    st.markdown(f"**{info['name']}**")
                    st.markdown(f"*Formula:* `{info['formula']}`")
                    st.markdown(f"*Deskripsi:* {info['description']}")
                    st.markdown(f"*Range:* {info['range'][0]} to {info['range'][1]}")
                
                # Show region info
                st.markdown("---")
                st.markdown("### Info Wilayah")
                st.markdown(f"**Nama:** {region_name}")
                if aoi_mode == "Batas Wilayah Indonesia":
                    if st.session_state.selected_province:
                        st.markdown(f"**Provinsi:** {st.session_state.selected_province['name']}")
                    if st.session_state.selected_city:
                        st.markdown(f"**Kab/Kota:** {st.session_state.selected_city['name']}")
                    if st.session_state.selected_district:
                        st.markdown(f"**Kecamatan:** {st.session_state.selected_district['name']}")

# =========================
# 11) STATISTICS
# =========================
st.subheader("📊 Statistik")

# Vegetation Indices Statistics
if index_layers:
    with st.status("📈 Calculating vegetation indices statistics...", expanded=True) as status:
        st.write("### Statistik Indeks Vegetasi")
        
        # Calculate statistics for all indices
        stats_data = []
        
        for idx_name, idx_layer in index_layers.items():
            try:
                status.write(f"Calculating {idx_name}...")
                stats = idx_layer.reduceRegion(
                    reducer=ee.Reducer.minMax().combine(
                        ee.Reducer.mean().combine(
                            ee.Reducer.stdDev(), sharedInputs=True
                        ), sharedInputs=True
                    ),
                    geometry=aoi,
                    scale=20,
                    maxPixels=1e8,
                    bestEffort=True,
                ).getInfo() or {}
                
                stats_data.append({
                    "Index": idx_name,
                    "Min": round(stats.get(f"{idx_name}_min", float('nan')), 4),
                    "Mean": round(stats.get(f"{idx_name}_mean", float('nan')), 4),
                    "Max": round(stats.get(f"{idx_name}_max", float('nan')), 4),
                    "Std Dev": round(stats.get(f"{idx_name}_stdDev", float('nan')), 4),
                    "Description": VEGETATION_INDICES[idx_name]["description"]
                })
                
            except Exception as e:
                st.warning(f"Gagal menghitung statistik {idx_name}: {e}")
        
        status.update(label="✅ Statistics calculated", state="complete")
    
    if stats_data:
        # Display as dataframe
        df_stats = pd.DataFrame(stats_data)
        st.dataframe(df_stats, use_container_width=True, hide_index=True)
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Bar chart of mean values
            fig_bar = px.bar(
                df_stats, 
                x='Index', 
                y='Mean',
                title='Nilai Mean Indeks Vegetasi',
                color='Mean',
                color_continuous_scale='RdYlGn',
                text='Mean'
            )
            fig_bar.update_traces(texttemplate='%{text:.3f}', textposition='outside')
            fig_bar.update_layout(height=400)
            st.plotly_chart(fig_bar, use_container_width=True)
        
        with col2:
            # Range chart
            fig_range = go.Figure()
            
            for idx in df_stats['Index']:
                row = df_stats[df_stats['Index'] == idx].iloc[0]
                fig_range.add_trace(go.Scatter(
                    x=[row['Min'], row['Mean'], row['Max']],
                    y=[idx, idx, idx],
                    mode='lines+markers',
                    name=idx,
                    line=dict(width=3),
                    marker=dict(size=10)
                ))
            
            fig_range.update_layout(
                title='Range Nilai Indeks (Min-Mean-Max)',
                xaxis_title='Nilai',
                yaxis_title='Index',
                height=400,
                showlegend=False
            )
            st.plotly_chart(fig_range, use_container_width=True)
        
        # Correlation matrix if multiple indices
        if len(index_layers) > 1:
            st.write("### Matriks Korelasi Indeks")
            
            # Sample points for correlation
            try:
                with st.spinner("Calculating correlations..."):
                    # Create a multi-band image
                    multi_band = ee.Image.cat(list(index_layers.values()))
                    
                    # Sample random points
                    sample = multi_band.sample(
                        region=aoi,
                        scale=30,
                        numPixels=1000,
                        geometries=False
                    )
                    
                    # Convert to dataframe
                    sample_dict = sample.getInfo()
                    features = sample_dict['features']
                    
                    if features:
                        data = []
                        for feature in features:
                            data.append(feature['properties'])
                        
                        df_corr = pd.DataFrame(data)
                        
                        # Calculate correlation
                        correlation = df_corr.corr()
                        
                        # Create heatmap
                        fig_corr = px.imshow(
                            correlation,
                            labels=dict(x="Index", y="Index", color="Correlation"),
                            x=correlation.columns,
                            y=correlation.columns,
                            color_continuous_scale='RdBu',
                            aspect="auto",
                            title="Korelasi Antar Indeks Vegetasi",
                            zmin=-1, zmax=1
                        )
                        fig_corr.update_layout(height=500)
                        st.plotly_chart(fig_corr, use_container_width=True)
                        
            except Exception as e:
                st.info("Tidak dapat menghitung korelasi: " + str(e))

# Land Cover Statistics
if any(lc in layers for lc in ["Dynamic World", "ESA WorldCover", "ESRI Land Cover"]):
    with st.status("📊 Calculating land cover statistics...", expanded=True) as status:
        st.write("### Statistik Land Cover")
        
        for lc_name in ["Dynamic World", "ESA WorldCover", "ESRI Land Cover"]:
            if lc_name in layers:
                try:
                    status.write(f"Processing {lc_name}...")
                    
                    if lc_name == "Dynamic World":
                        band_name = "label_mode" if dw_mode == "mode" else "label"
                        legend = LAND_COVER_LEGENDS["Dynamic_World"]
                        scale = 30
                    elif lc_name == "ESA WorldCover":
                        band_name = "Map"
                        legend = LAND_COVER_LEGENDS["ESA_WorldCover"]
                        scale = 10
                    else:  # ESRI
                        band_name = "b1"
                        legend = LAND_COVER_LEGENDS["ESRI_LandCover"]
                        scale = 10
                    
                    # Calculate pixel counts
                    pixel_counts = layers[lc_name].select(band_name).reduceRegion(
                        reducer=ee.Reducer.frequencyHistogram(),
                        geometry=aoi,
                        scale=scale,
                        maxPixels=1e8,
                        bestEffort=True
                    ).getInfo()
                    
                    if pixel_counts and band_name in pixel_counts:
                        st.write(f"**{lc_name}**")
                        
                        # Convert to DataFrame
                        counts = pixel_counts[band_name]
                        data = []
                        for class_val, count in counts.items():
                            if str(class_val) in legend:
                                area_ha = count * (scale * scale) / 10000
                                data.append({
                                    "Class": legend[str(class_val)]["label"],
                                    "Color": legend[str(class_val)]["color"],
                                    "Area (ha)": round(area_ha, 2),
                                    "Pixels": count
                                })
                        
                        if data:
                            df = pd.DataFrame(data)
                            total_area = df["Area (ha)"].sum()
                            df["Percentage"] = round(df["Area (ha)"] / total_area * 100, 1)
                            df = df.sort_values("Area (ha)", ascending=False)
                            
                            # Display table
                            col1, col2 = st.columns([2, 3])
                            
                            with col1:
                                st.dataframe(
                                    df[["Class", "Area (ha)", "Percentage"]],
                                    use_container_width=True,
                                    hide_index=True
                                )
                                
                                # Summary
                                st.metric("Total Area", f"{total_area:.2f} ha")
                            
                            with col2:
                                # Pie chart
                                fig_pie = px.pie(
                                    df, 
                                    values='Area (ha)', 
                                    names='Class',
                                    title=f'Distribusi {lc_name}',
                                    color='Class',
                                    color_discrete_map={row['Class']: row['Color'] for _, row in df.iterrows()}
                                )
                                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                                fig_pie.update_layout(height=400)
                                st.plotly_chart(fig_pie, use_container_width=True)
                    
                except Exception as e:
                    st.warning(f"Gagal menghitung statistik {lc_name}: {e}")
        
        status.update(label="✅ Land cover statistics complete", state="complete")

# =========================
# 12) TIME SERIES ANALYSIS (if applicable)
# =========================
if analysis_type in ["Vegetation Indices Analysis", "Combined Analysis"] and selected_indices:
    with st.expander("📈 Analisis Time Series (Optional)", expanded=False):
        st.write("Analisis perubahan indeks vegetasi sepanjang tahun")
        
        # Time series parameters
        col1, col2, col3 = st.columns(3)
        with col1:
            ts_index = st.selectbox("Pilih Indeks", selected_indices)
        with col2:
            ts_year = st.selectbox("Tahun", list(range(2017, 2026)))
        with col3:
            ts_interval = st.selectbox("Interval", ["Monthly", "Bi-weekly"])
        
        if st.button("Generate Time Series", type="primary"):
            with st.spinner("Memproses time series..."):
                try:
                    # Create date ranges
                    if ts_interval == "Monthly":
                        date_ranges = []
                        for month in range(1, 13):
                            start = f"{ts_year}-{month:02d}-01"
                            if month == 12:
                                end = f"{ts_year}-12-31"
                            else:
                                end = f"{ts_year}-{month+1:02d}-01"
                            date_ranges.append((start, end, f"{month:02d}"))
                    
                    # Calculate index for each period
                    time_series_data = []
                    
                    progress_bar = st.progress(0)
                    for idx, (start, end, label) in enumerate(date_ranges):
                        s2_period = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                                    .filterBounds(aoi)
                                    .filterDate(start, end)
                                    .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", 40))
                                    .map(mask_s2_clouds))
                        
                        if s2_period.size().getInfo() > 0:
                            median = s2_period.median()
                            index = calculate_index(median, ts_index)
                            
                            stats = index.reduceRegion(
                                reducer=ee.Reducer.mean(),
                                geometry=aoi,
                                scale=20,
                                maxPixels=1e8,
                                bestEffort=True
                            ).getInfo()
                            
                            time_series_data.append({
                                "Period": label,
                                "Value": stats.get(ts_index, None),
                                "Date": f"{ts_year}-{label}-15"
                            })
                        
                        progress_bar.progress((idx + 1) / len(date_ranges))
                    
                    # Create time series plot
                    if time_series_data:
                        df_ts = pd.DataFrame(time_series_data)
                        df_ts = df_ts.dropna(subset=['Value'])
                        df_ts['Date'] = pd.to_datetime(df_ts['Date'])
                        
                        fig_ts = px.line(
                            df_ts, 
                            x='Date', 
                            y='Value',
                            title=f'Time Series {ts_index} - {ts_year} ({region_name})',
                            markers=True
                        )
                        fig_ts.update_layout(
                            xaxis_title="Bulan",
                            yaxis_title=f"{ts_index} Value",
                            height=400
                        )
                        st.plotly_chart(fig_ts, use_container_width=True)
                        
                        # Summary statistics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Min", f"{df_ts['Value'].min():.3f}")
                        with col2:
                            st.metric("Max", f"{df_ts['Value'].max():.3f}")
                        with col3:
                            st.metric("Mean", f"{df_ts['Value'].mean():.3f}")
                        with col4:
                            st.metric("Std Dev", f"{df_ts['Value'].std():.3f}")
                        
                except Exception as e:
                    st.error(f"Error: {e}")

# =========================
# 13) DOWNLOAD OPTIONS
# =========================
st.subheader("💾 Download Hasil")

col1, col2, col3 = st.columns(3)

with col1:
    with st.expander("📥 Export ke Google Drive"):
        export_options = {name: layer for name, layer in layers.items() 
                         if not name.startswith("DW") or name == "Dynamic World"}
        
        selected_export = st.selectbox("Pilih layer", list(export_options.keys()))
        export_desc = st.text_input("Deskripsi file", value=f"{selected_export}_{region_name}_{year}")
        
        if st.button("Start Export", type="primary"):
            try:
                with st.spinner("Starting export..."):
                    export_image = export_options[selected_export]
                    scale = 10 if "Land Cover" in selected_export else 20
                    
                    task = ee.batch.Export.image.toDrive(
                        image=export_image,
                        description=export_desc.replace(" ", "_"),
                        scale=scale,
                        region=aoi,
                        fileFormat='GeoTIFF',
                        maxPixels=1e9
                    )
                    task.start()
                    st.success(f"✅ Export dimulai!\n\n**Task ID:** `{task.id}`")
                    st.info("Cek progress di [Google Earth Engine Tasks](https://code.earthengine.google.com/tasks)")
            except Exception as e:
                st.error(f"❌ Gagal export: {e}")

with col2:
    # Download statistics
    all_stats = {
        "analysis_date": datetime.now().isoformat(),
        "region": region_name,
        "year": year,
        "analysis_type": analysis_type
    }
    
    if 'stats_data' in locals() and stats_data:
        all_stats["vegetation_indices"] = stats_data
    
    json_str = json.dumps(all_stats, indent=2)
    st.download_button(
        "📊 Download Statistik (JSON)",
        data=json_str,
        file_name=f"stats_{region_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )

with col3:
    # Generate report
    if st.button("📄 Generate Report", type="secondary"):
        report = f"""# Laporan Analisis GEE
        
**Tanggal Analisis:** {datetime.now().strftime('%Y-%m-%d %H:%M')}
**Wilayah:** {region_name}
**Tahun Data:** {year}
**Tipe Analisis:** {analysis_type}

## Area of Interest
- Mode: {aoi_mode}
- Nama Wilayah: {region_name}
"""
        
        if 'stats_data' in locals() and stats_data:
            report += "\n## Statistik Indeks Vegetasi\n"
            for stat in stats_data:
                report += f"\n### {stat['Index']} ({stat['Description']})\n"
                report += f"- Min: {stat['Min']}\n"
                report += f"- Mean: {stat['Mean']}\n"
                report += f"- Max: {stat['Max']}\n"
                report += f"- Std Dev: {stat['Std Dev']}\n"
        
        report += "\n---\n*Generated by GEE Land Cover & Vegetation Analysis Platform*"
        
        st.download_button(
            "Download Report (Markdown)",
            data=report,
            file_name=f"report_{region_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown"
        )

# Footer
st.markdown("---")
st.caption(
    "🛰️ Aplikasi ini menggunakan Google Earth Engine untuk analisis land cover dan vegetasi. "
    "Mendukung multiple indeks vegetasi (NDVI, NDWI, MNDWI, NDBI, EVI, SAVI, BSI, NDMI) "
    "dan dataset land cover global (Dynamic World, ESA WorldCover, ESRI Land Cover). "
    f"Data batas wilayah Indonesia dari [SP3STAB API]({API_BASE_URL})."
)
"""
app.py
Urban Heat Explorer
Interactive Geospatial AI Dashboard for Landsat 8/9 Level-2 Analysis
Features interactive Folium slippy maps with water-masked alpha transparencies,
floating thermal legends, Plotly density contours, and benchmark vulnerability metrics.
"""

import os
import streamlit as st
import numpy as np
import rasterio
import folium
from streamlit_folium import st_folium
from dotenv import load_dotenv

from src.indices import (
    calculate_ndvi,
    calculate_ndbi,
    scale_surface_temperature,
    extract_thermal_metrics,
)
from src.visualizer import (
    array_to_colormap_image,
    array_to_rgba_overlay,
    create_folium_legend,
    generate_true_color,
    get_raster_wgs84_bounds,
    create_interactive_scatter_density,
)
from src.ai_analyst import analyze_satellite_scene

load_dotenv()

st.set_page_config(
    page_title="Urban Heat Island Explorer",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    [data-testid="stSidebarCollapseButton"], section[data-testid="stSidebar"] {
        display: none !important;
    }
    .metric-card {
        background-color: #1A1C24;
        border: 1px solid #2B2F3E;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 10px;
    }
    .metric-title {
        color: #8E94A5;
        font-size: 13px;
        font-weight: 500;
        margin-bottom: 4px;
    }
    .metric-val {
        color: #F0F2F6;
        font-size: 24px;
        font-weight: 700;
    }
    .metric-sub {
        color: #2DD4BF;
        font-size: 12px;
    }
    .anomaly-hotspot {
        border-left: 4px solid #FF4B4B;
        background: rgba(255, 75, 75, 0.08);
        padding: 10px 14px;
        border-radius: 4px;
        margin-bottom: 8px;
    }
    .anomaly-cool {
        border-left: 4px solid #00D4BB;
        background: rgba(0, 212, 187, 0.08);
        padding: 10px 14px;
        border-radius: 4px;
        margin-bottom: 8px;
    }
    .equity-card {
        background: #141721;
        border: 1px solid #2E334D;
        border-radius: 8px;
        padding: 16px;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

CITY_REGISTRY = {
    "phoenix": {
        "name": "Phoenix, AZ (Arid Desert)",
        "center": [33.46, -112.06],
        "svi_score": "High (0.84)",
        "canopy_equity_gap": "Neighborhood canopy ranges from 4% (South Phoenix) to 19% (Paradise Valley).",
        "human_vulnerability_note": "Arid biomes exhibit extreme diurnal surface heating where high NDBI industrial corridors create nighttime heat traps."
    },
    "atlanta": {
        "name": "Atlanta, GA (Humid Forest)",
        "center": [33.77, -84.39],
        "svi_score": "Moderate-High (0.71)",
        "canopy_equity_gap": "Significant tree loss along I-20 and SW industrial zones; canopy drops below 20% in low-income corridors.",
        "human_vulnerability_note": "Canopy transpiration buffers daytime heat, but high humidity elevates the Canopy Air Layer heat index."
    },
    "chicago": {
        "name": "Chicago, IL (Temperate Lakefront)",
        "center": [41.88, -87.64],
        "svi_score": "High (0.79)",
        "canopy_equity_gap": "South/West Side industrial corridors exhibit <10% canopy vs. 35%+ in lakefront North Side neighborhoods.",
        "human_vulnerability_note": "Strong lake breeze gradient; high-density inland asphalt surfaces amplify localized heat stress."
    },
    "nyc": {
        "name": "New York City, NY (Ultra-Dense Megacity)",
        "center": [40.76, -73.97],
        "svi_score": "High (0.81)",
        "canopy_equity_gap": "South Bronx and East New York lack mature street trees (<12% canopy) compared to Upper Manhattan and Brownstone Brooklyn (>30%).",
        "human_vulnerability_note": "Extreme vertical thermal mass and canyon trapping; Central Park acts as an intense 4°C–6°C cooling oasis amidst midtown concrete."
    },
}

st.title("Urban Heat Explorer")
st.caption("Landsat 8/9 Thermal & Vegetation Indices")

with st.container():
    c1, c2 = st.columns([3, 1])
    with c1:
        selected_city = st.selectbox(
            "Target City",
            options=list(CITY_REGISTRY.keys()),
            format_func=lambda k: CITY_REGISTRY[k]["name"],
            key="main_city_select"
        )
    with c2:
        st.caption("**Sensor Suite:** Landsat 9 OLI/TIRS")
        st.caption("**Spatial Resolution:** 30m / pixel (EPSG:UTM)")

st.divider()


@st.cache_data
def load_and_process_scene(city_key: str):
    """Dynamically ingests GeoTIFFs, computing indices and georeferenced spatial bounds."""
    city_dir = os.path.join("data", "raw", city_key)
    
    b4_path = os.path.join(city_dir, f"{city_key}_B4.TIF")
    b5_path = os.path.join(city_dir, f"{city_key}_B5.TIF")
    b6_path = os.path.join(city_dir, f"{city_key}_B6.TIF")
    b10_path = os.path.join(city_dir, f"{city_key}_B10.TIF")

    if not all(os.path.exists(p) for p in [b4_path, b5_path, b6_path, b10_path]):
        return None

    with rasterio.open(b4_path) as r, rasterio.open(b5_path) as n, rasterio.open(b6_path) as s, rasterio.open(b10_path) as t:
        red = r.read(1)
        nir = n.read(1)
        swir = s.read(1)
        thermal = t.read(1)
        bounds_wgs84 = get_raster_wgs84_bounds(t.transform, t.width, t.height, t.crs)

    ndvi = calculate_ndvi(red, nir)
    ndbi = calculate_ndbi(nir, swir)
    _, lst_c = scale_surface_temperature(thermal)
    metrics = extract_thermal_metrics(lst_c, ndvi, ndbi, biome=city_key)

    img_comp = generate_true_color(red, nir, swir)
    img_lst = array_to_colormap_image(lst_c, cmap_name="magma")
    img_ndvi = array_to_colormap_image(ndvi, cmap_name="YlGn")
    rgba_lst, vmin_c, vmax_c = array_to_rgba_overlay(lst_c, ndvi_data=ndvi, cmap_name="magma", opacity=0.75)

    os.makedirs("data/processed", exist_ok=True)
    lst_cache = f"data/processed/{city_key}_lst.png"
    comp_cache = f"data/processed/{city_key}_composite.png"
    img_lst.save(lst_cache)
    img_comp.save(comp_cache)

    return {
        "metrics": metrics,
        "img_comp": img_comp,
        "img_lst": img_lst,
        "img_ndvi": img_ndvi,
        "rgba_lst": rgba_lst,
        "vmin_c": vmin_c,
        "vmax_c": vmax_c,
        "bounds_wgs84": bounds_wgs84,
        "ndvi": ndvi,
        "lst_c": lst_c,
        "lst_cache": lst_cache,
        "comp_cache": comp_cache,
    }


scene = load_and_process_scene(selected_city)
if scene is None:
    st.error(f"Datasets for `{selected_city}` not found in `data/raw/{selected_city}/`.")
    st.stop()

metrics = scene["metrics"]
city_meta = CITY_REGISTRY[selected_city]

# Telemetry KPIs
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">UHI INTENSITY DELTA</div>
        <div class="metric-val">+{metrics['uhi_intensity_delta_c']} °C</div>
        <div class="metric-sub">Built Grid vs. Vegetated Canopy</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">BUILT CORE MEAN TEMP</div>
        <div class="metric-val">{metrics['urban_mean_temp_c']} °C</div>
        <div class="metric-sub">Peak LST: {metrics['max_temp_c']} °C</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">CANOPY MEAN TEMP</div>
        <div class="metric-val">{metrics['canopy_mean_temp_c']} °C</div>
        <div class="metric-sub">Min LST: {metrics['min_temp_c']} °C</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">NDVI-LST CORRELATION</div>
        <div class="metric-val">{metrics['ndvi_lst_correlation']}</div>
        <div class="metric-sub">Biophysical Cooling Efficiency</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

tab1, tab2, tab3 = st.tabs(["Interactive Thermal Map", "Biophysical & Equity Analytics", "Diagnostic Assessment"])

with tab1:
    st.subheader(f"Georeferenced Land Surface Temperature Overlay: {city_meta['name']}")
    st.caption("Pan, zoom, and inspect underlying street networks, building density, and park infrastructure.")

    bounds = scene["bounds_wgs84"]
    center_lat = (bounds[0][0] + bounds[1][0]) / 2.0
    center_lon = (bounds[0][1] + bounds[1][1]) / 2.0

    m = folium.Map(
        location=[center_lat, center_lon],
        tiles=None,
        control_scale=True
    )

    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}",
        attr="Esri, HERE, Garmin, © OpenStreetMap contributors",
        name="Dark Canvas (Esri)",
        control=True
    ).add_to(m)

    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri, Maxar, Earthstar Geographics",
        name="Satellite Imagery (Esri)",
        control=True
    ).add_to(m)

    folium.TileLayer(
        tiles="OpenStreetMap",
        name="OpenStreetMap",
        control=True
    ).add_to(m)

    # Add water-masked thermal raster
    folium.raster_layers.ImageOverlay(
        image=scene["rgba_lst"],
        bounds=bounds,
        opacity=0.75,
        name="Calibrated Surface Temperature (°C)",
        interactive=True,
        cross_origin=False,
        zindex=2
    ).add_to(m)

    # Add floating radiometric colorbar legend
    legend = create_folium_legend(scene["vmin_c"], scene["vmax_c"])
    legend.add_to(m)

    m.fit_bounds(bounds)

    folium.LayerControl(position="topright").add_to(m)
    st_folium(m, width="100%", height=580, returned_objects=[])

with tab2:
    st.subheader("Biophysical Gradient Analysis & Social Equity Context")

    pcol1, pcol2 = st.columns([3, 2])
    with pcol1:
        scatter_fig = create_interactive_scatter_density(scene["ndvi"], scene["lst_c"])
        st.plotly_chart(scatter_fig, use_container_width=True)

    with pcol2:
        st.markdown(f"""
        <div class="equity-card">
            <h4 style="margin-top:0; color:#2DD4BF;">Metropolitan Baseline & Demographic Context</h4>
            <p><strong>CDC Social Vulnerability Index (SVI):</strong> {city_meta['svi_score']}</p>
            <p><strong>Tree Equity & Canopy Disparity:</strong><br>{city_meta['canopy_equity_gap']}</p>
            <p><strong>Microclimate Vulnerability Dynamics:</strong><br>{city_meta['human_vulnerability_note']}</p>
            <hr style="border-color:#2E334D;">
            <p style="font-size:12px; color:#8E94A5;">
                <em>Note on Sensor Physics:</em> Satellites measure <strong>Land Surface Temperature (LST)</strong> (radiant skin temperature of ground/roof materials). Human heat illness occurs in the <strong>Canopy Air Layer (CAL)</strong>, influenced by wind, shade, and humidity.
            </p>
            <p style="font-size:11px; color:#6B7280; margin-bottom:0;">
                Sources: CDC SVI Benchmark & Municipal Urban Forestry Assessments
            </p>
        </div>
        """, unsafe_allow_html=True)

with tab3:
    st.subheader(f"Scene Assessment: {city_meta['name']}")
    st.caption("Submits calibrated false-color & thermal arrays with biophysical summary metrics to Gemini for structured environmental evaluation.")

    if st.button("Run Gemini Scene Evaluation", type="primary"):
        active_key = os.getenv("GEMINI_API_KEY", "")
        if not active_key:
            st.error("No Gemini API key detected in .env.")
        else:
            with st.spinner(f"Evaluating thermal distribution and vulnerability for {selected_city}..."):
                try:
                    report = analyze_satellite_scene(
                        lst_image_path=scene["lst_cache"],
                        composite_image_path=scene["comp_cache"],
                        metrics=metrics,
                        api_key=active_key,
                    )

                    st.success(f"**UHI Severity Rating: {report.uhi_severity_tier}**")
                    st.write(report.scene_summary)

                    acol1, acol2 = st.columns(2)
                    with acol1:
                        st.markdown("### Detected Thermal Anomalies")
                        for a in report.detected_anomalies:
                            css_class = "anomaly-hotspot" if a.feature_type == "hotspot" else "anomaly-cool"
                            st.markdown(f"""
                            <div class="{css_class}">
                                <strong>[{a.feature_type.upper()}] {a.feature_name}</strong><br>
                                Model Estimated Surface Temp: ~{a.estimated_temp_c}°C<br>
                                <span style="font-size: 12px; color: #BBB;">Biophysical Driver: {a.cooling_factor}</span>
                            </div>
                            """, unsafe_allow_html=True)

                    with acol2:
                        st.markdown("### Recommended Mitigations")
                        for intervention in report.actionable_interventions:
                            st.markdown(f"• {intervention}")

                        st.markdown("### Canopy Buffering Assessment")
                        st.info(report.vegetation_buffering_assessment)

                except Exception as e:
                    st.error(f"Inference failed: {e}")
import streamlit as st
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap
import numpy as np
import pandas as pd

# --- PAGE CONFIG ---
st.set_page_config(page_title="NASA Urban Sustainability Dashboard", layout="wide")

# --- CSS STYLE ---
st.markdown("""
    <style>
        body {
            background-color: #0b0c10;
            color: #f5f5f5;
            font-family: 'Segoe UI', sans-serif;
        }
        h1, h2, h3 {
            color: #66fcf1;
        }
        .stApp {
            background-color: #0b0c10;
        }
        .metric {
            font-size: 18px;
            color: #45a29e;
        }
    </style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.markdown("<h1 style='text-align: center;'>🌍 NASA Urban Sustainability Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color:#c5c6c7;'>Analyzing air quality and vegetation using NASA Earth observation data.</p>", unsafe_allow_html=True)

# --- SIDEBAR FILTERS ---
st.sidebar.header("🛰️ Filters")
pollution_level = st.sidebar.selectbox("Select Pollution Level", ["All", "Low", "Moderate", "High"])
show_heatmap = st.sidebar.checkbox("Show Heatmap", value=True)

# --- CREATE SAMPLE DATA ---
np.random.seed(42)
districts = ["Almaty", "Saryarka", "Yesil", "Baikonur", "Nura"]
data = []

for d in districts:
    for _ in range(10):
        lat = 51.1 + np.random.rand()/5
        lon = 71.3 + np.random.rand()/5
        pm25 = np.random.uniform(10, 40)
        data.append([d, lat, lon, pm25])

df = pd.DataFrame(data, columns=["district", "lat", "lon", "pm25"])

# --- FILTER DATA ---
if pollution_level != "All":
    if pollution_level == "Low":
        df = df[df["pm25"] < 15]
    elif pollution_level == "Moderate":
        df = df[(df["pm25"] >= 15) & (df["pm25"] < 25)]
    else:
        df = df[df["pm25"] >= 25]

# --- MAP ---
m = folium.Map(location=[51.1694, 71.4491], zoom_start=11, tiles="CartoDB dark_matter")

# Add heatmap if selected
if show_heatmap:
    heat_data = [[row["lat"], row["lon"], row["pm25"]] for _, row in df.iterrows()]
    HeatMap(heat_data, radius=30, blur=25, min_opacity=0.4).add_to(m)

# Add pollution markers
for _, row in df.iterrows():
    color = "green" if row["pm25"] < 15 else "orange" if row["pm25"] < 25 else "red"
    folium.CircleMarker(
        location=[row["lat"], row["lon"]],
        radius=8,
        color=color,
        fill=True,
        fill_opacity=0.8,
        popup=f"{row['district']}<br>PM2.5: {row['pm25']:.1f} µg/m³"
    ).add_to(m)

st_folium(m, width=900, height=600)

# --- INSIGHTS SECTION ---
st.markdown("""
---
### 🧭 Insights for Urban Planners

- 🟥 **High pollution (PM2.5 > 25)** — consider green corridors and traffic restrictions.  
- 🟧 **Moderate pollution (15–25)** — enhance vegetation density.  
- 🟩 **Low pollution (<15)** — maintain and monitor air quality.

### 🌱 Recommendations
NASA data (Aerosol Optical Depth + NDVI) helps cities:
- Identify environmental risks before they grow.  
- Optimize green infrastructure placement.  
- Improve overall wellbeing through smart data-driven policies.
""")

# --- FOOTER ---
st.markdown("""
---
<p style='text-align: center; color:#45a29e;'>Developed by Amina for NASA Space Apps Astana 🌎</p>
""", unsafe_allow_html=True)

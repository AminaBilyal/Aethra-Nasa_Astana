import streamlit as st
import folium
from streamlit_folium import st_folium

# ------------------------- PAGE CONFIG -------------------------
st.set_page_config(page_title="NASA Hackathon Project", layout="wide")

# ------------------------- HEADER -------------------------
st.title("🌍 NASA Hackathon: Data Pathways to Healthy Cities")

st.markdown("""
### Environmental Monitoring for Sustainable Cities  
This project analyzes **air quality** and **vegetation levels** using NASA Earth observation data.  
It helps identify areas with high pollution or low green coverage — supporting better **urban planning** and **sustainability**.
""")

# ------------------------- MAP SECTION -------------------------
st.subheader("🛰️ Satellite Environmental Map: Astana")

# Create base map (OpenStreetMap for reliability)
m = folium.Map(location=[51.1694, 71.4491], zoom_start=11, tiles="OpenStreetMap")

# Example pollution data (for demo purposes)
pollution_points = [
    {"lat": 51.17, "lon": 71.43, "value": "High pollution"},
    {"lat": 51.18, "lon": 71.47, "value": "Moderate pollution"},
    {"lat": 51.16, "lon": 71.45, "value": "Low pollution"},
]

# Add markers
for point in pollution_points:
    color = "red" if "High" in point["value"] else "orange" if "Moderate" in point["value"] else "green"
    folium.CircleMarker(
        location=[point["lat"], point["lon"]],
        radius=8,
        color=color,
        fill=True,
        fill_opacity=0.7,
        popup=point["value"]
    ).add_to(m)

# Display folium map
st_data = st_folium(m, width=800, height=500)

# ------------------------- PROJECT INFO -------------------------
st.markdown("""
---
### 🧠 About the Project
- **Data Sources:** NASA Earth Observation (Aerosol Optical Depth, NDVI, etc.)  
- **Technologies:** Python, Streamlit, Folium, Open Data APIs, ML Models  
- **Goal:** Enable cities to make data-driven decisions about sustainability  
- **Focus City:** Astana, Kazakhstan  
- **Use Case:** Detecting air pollution zones and green space changes  

### 👩‍💻 Team
**Amina Bilialova** — Lead Developer & Researcher  

### 🚀 Status
✅ Data collected from NASA  
✅ Visual analysis and ML model under development  
✅ Streamlit demo ready for integration  
""")

st.success("✅ Interactive map loaded successfully! Your project page is ready to share with judges.")

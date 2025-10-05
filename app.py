import streamlit as st
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="NASA Hackathon Project", layout="wide")

st.title("NASA Hackathon: Data Pathways to Healthy Cities")

st.markdown("""
### Environmental Monitoring for Sustainable Cities
This project analyzes urban air quality and green zones using NASA Earth observation data.  
It helps identify areas with high pollution levels or low vegetation coverage, supporting better urban planning and sustainability.
""")

st.subheader("Satellite Environmental Map (Astana)")

# --- Create map with satellite layer ---
m = folium.Map(location=[51.1694, 71.4491], zoom_start=11, tiles="Stamen Terrain")

# Example pollution points (replace with real data later)
pollution_points = [
    {"lat": 51.17, "lon": 71.43, "value": "High pollution"},
    {"lat": 51.18, "lon": 71.47, "value": "Moderate pollution"},
    {"lat": 51.16, "lon": 71.45, "value": "Low pollution"},
]

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

st_folium(m, width=800, height=550)

st.markdown("""
---
#### About the Project
- Data Sources: NASA Earth Observation (Aerosol Optical Depth, Vegetation Index)  
- Technologies: Streamlit, Python, Folium, Open Data APIs  
- Goal: To support sustainable city development through environmental data analysis  
- Target Audience: Municipalities, environmental researchers, and citizens  

#### Team
Amina and her research team on sustainable cities.
""")

st.success("Demo project with Folium map loaded successfully. Ready for integration with NASA satellite data.")

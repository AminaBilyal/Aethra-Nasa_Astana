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

st.subheader("Satellite Monitoring Map (Astana)")
m = folium.Map(location=[51.1694, 71.4491], zoom_start=11)
folium.Marker(
    [51.1694, 71.4491],
    tooltip="Astana",
    popup="City Center",
    icon=folium.Icon(color="green", icon="leaf")
).add_to(m)
st_folium(m, width=750, height=500)

st.markdown("""
---
#### About the Project
- Data Sources: NASA Earth Observation (Aerosol Optical Depth, Vegetation Index)  
- Technologies: Streamlit, Python, Folium, Open Data APIs  
- Goal: To help cities develop sustainably through environmental data analysis  
- Target Audience: Municipalities, environmental organizations, urban planners, and citizens  

#### Team
Amina and her research team on sustainable cities.
""")

st.success("Demo project loaded successfully. Ready for integration with NASA APIs and real-world environmental data.")

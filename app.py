
# NASA Hackathon – Data Pathways to Healthy Cities
# Author: Amina Bilyalova


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from streamlit_folium import st_folium
from sklearn.ensemble import RandomForestRegressor
import joblib

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="NASA Hackathon Project",
    layout="wide",
    page_icon="🌍"
)

# --- TITLE AND HEADER ---
st.title("NASA Hackathon: Data Pathways to Healthy Cities")
st.markdown("""
**Environmental Monitoring & Urban Planning Tool**  
Using NASA Earth Observation data to analyze air quality, green zones, and population density in Astana.  
Interactive visualizations, ML predictions, and heatmaps help plan sustainable city growth.
""")
st.image("https://upload.wikimedia.org/wikipedia/commons/e/e5/NASA_logo.svg", width=120)
st.markdown("#### Project by Amina Bilyalova and team")

# --- LOAD DATA ---
@st.cache_data
def load_data():
    df = pd.read_csv("astana_analysis.csv")
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.month
    return df

df = load_data()

# --- LOAD MODEL ---
model = joblib.load("pollution_model.pkl")

# --- TABS ---
tabs = st.tabs(["📊 Graphs", "🤖 ML Model", "🗺️ Interactive Map"])

# --------------------------
# TAB 1: GRAPHS
# --------------------------
with tabs[0]:
    st.subheader("Graphs & Analysis")
    col1, col2 = st.columns(2)
    
    # Average PM2.5 by district
    with col1:
        pm25_avg = df.groupby('district')['pm25'].mean().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(5,4))
        ax.bar(pm25_avg.index, pm25_avg.values, color='orangered')
        ax.set_title("Average PM2.5 by District")
        ax.set_ylabel("PM2.5")
        ax.axhline(y=25, color='red', linestyle='--', label='Unhealthy')
        ax.axhline(y=15, color='orange', linestyle='--', label='Moderate')
        ax.legend()
        plt.xticks(rotation=45)
        st.pyplot(fig)
    
    # Green spaces vs PM2.5
    with col2:
        green_pm25 = df.groupby('district').agg({'green_spaces':'first', 'pm25':'mean'})
        fig2, ax2 = plt.subplots(figsize=(5,4))
        ax2.scatter(green_pm25['green_spaces'], green_pm25['pm25'], s=150, c='green', alpha=0.6)
        for i, row in green_pm25.iterrows():
            ax2.annotate(i, (row['green_spaces'], row['pm25']), xytext=(5,5), textcoords='offset points')
        ax2.set_xlabel("Green Spaces (km²)")
        ax2.set_ylabel("PM2.5")
        ax2.set_title("Green Spaces vs Pollution")
        st.pyplot(fig2)
    
    # Seasonal PM2.5
    st.markdown("**Seasonal Pollution Patterns**")
    monthly_pm25 = df.groupby('month')['pm25'].mean()
    fig3, ax3 = plt.subplots(figsize=(8,4))
    ax3.plot(monthly_pm25.index, monthly_pm25.values, marker='o', linewidth=2)
    ax3.set_xticks(range(1,13))
    ax3.set_xlabel("Month")
    ax3.set_ylabel("PM2.5")
    ax3.set_title("Seasonal PM2.5 Trends")
    st.pyplot(fig3)
    
    # Feature importance
    st.markdown("**Feature Importance for PM2.5 Prediction**")
    features = ['temperature', 'humidity', 'population', 'green_spaces', 'traffic_volume', 'month']
    importances = model.feature_importances_
    feat_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values('Importance', ascending=True)
    fig4, ax4 = plt.subplots(figsize=(5,4))
    ax4.barh(feat_df['Feature'], feat_df['Importance'], color='skyblue')
    ax4.set_title("Feature Importance")
    st.pyplot(fig4)

# --------------------------
# TAB 2: ML MODEL
# --------------------------
with tabs[1]:
    st.subheader("ML Model Prediction")
    st.markdown("Predict PM2.5 pollution for given district parameters")
    
    # Input sliders
    temp = st.slider("Temperature (°C)", float(df['temperature'].min()), float(df['temperature'].max()), 20.0)
    humidity = st.slider("Humidity (%)", float(df['humidity'].min()), float(df['humidity'].max()), 60.0)
    population = st.number_input("Population", min_value=1000, max_value=1000000, value=300000)
    green_spaces = st.number_input("Green spaces (km²)", min_value=1, max_value=50, value=10)
    traffic_volume = st.number_input("Traffic volume", min_value=10000, max_value=200000, value=80000)
    month = st.slider("Month", 1, 12, 6)
    
    # Make prediction
    input_df = pd.DataFrame([[temp, humidity, population, green_spaces, traffic_volume, month]],
                            columns=features)
    pred = model.predict(input_df)[0]
    st.success(f"Predicted PM2.5: {pred:.1f} µg/m³")

# --------------------------
# TAB 3: INTERACTIVE MAP
# --------------------------
with tabs[2]:
    st.subheader("Interactive Pollution Map of Astana")
    
    # Base map
    m = folium.Map(location=[51.1694, 71.4491], zoom_start=11, tiles="Stamen Terrain")
    
    # Heatmap: average PM2.5 per district
    from folium.plugins import HeatMap
    heat_data = df.groupby(['latitude','longitude'])['pm25'].mean().reset_index()
    heat_data = heat_data[['latitude','longitude','pm25']].values.tolist()
    HeatMap(heat_data, radius=25, blur=15, gradient={0.2: 'green', 0.5:'orange', 0.8:'red'}).add_to(m)
    
    # Add markers
    for _, row in df.groupby('district').first().iterrows():
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=f"{row['district']}<br>Population: {row['population']}<br>Green: {row['green_spaces']} km²",
            icon=folium.Icon(color='blue', icon='info-sign')
        ).add_to(m)
    
    # Display map
    st_folium(m, width=900, height=600)

# --- FOOTER ---
st.markdown("""
---
**Data Sources:** NASA Earth Observation  
**Team:** Amina Bilyalova and research team  
**Goal:** Support sustainable city development through environmental monitoring and planning
""")

import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# -----------------------------
# PAGE CONFIG
st.set_page_config(
    page_title="NASA Hackathon: Healthy Cities",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set page background color (black)
st.markdown(
    """
    <style>
    .stApp {
        background-color: #0b0c10;
        color: #f0f0f0;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# HEADER
st.markdown("<h1 style='text-align: center; color: #ffffff;'>NASA Hackathon: Healthy Cities</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #a0a0a0;'>Urban Environmental Monitoring & Air Quality Analysis</h3>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color: #c0c0c0;'>Demonstrating how urban planners can use NASA Earth observation data to develop smart strategies for city growth, maintaining wellbeing of people and environment.<br>Author: Amina Bilyalova</p>", unsafe_allow_html=True)
st.markdown("---")

# -----------------------------
# LOAD DATA
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("astana_analysis.csv")
    except:
        # If CSV not found, create sample data
        districts = {
            'Almaty': {'lat': 51.15, 'lon': 71.45, 'population': 350000, 'green_spaces': 12},
            'Saryarka': {'lat': 51.20, 'lon': 71.40, 'population': 280000, 'green_spaces': 8},
            'Yesil': {'lat': 51.18, 'lon': 71.35, 'population': 320000, 'green_spaces': 15},
            'Nura': {'lat': 51.12, 'lon': 71.50, 'population': 190000, 'green_spaces': 10},
            'Baikonur': {'lat': 51.22, 'lon': 71.48, 'population': 150000, 'green_spaces': 9}
        }
        all_data = []
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
        for district, info in districts.items():
            for i, date in enumerate(dates):
                base_temp = 5 + 25 * np.sin(2 * np.pi * i / 365)
                pm25 = np.random.normal(20,5)
                humidity = 60 + 20*np.sin(2*np.pi*i/365 + np.pi/2) + np.random.normal(0,5)
                traffic = np.random.normal(50000,10000)
                park_need = max(0,(pm25-10)*info['population']/10000)
                all_data.append({
                    'date': date,
                    'district': district,
                    'temperature': round(base_temp + np.random.normal(0,2),1),
                    'humidity': round(max(30,min(95,humidity)),1),
                    'pm25': round(max(5,pm25),1),
                    'population': info['population'],
                    'green_spaces': info['green_spaces'],
                    'traffic_volume': max(10000, traffic),
                    'park_need_index': round(park_need,1),
                    'latitude': info['lat'],
                    'longitude': info['lon']
                })
        df = pd.DataFrame(all_data)
    df['month'] = pd.to_datetime(df['date']).dt.month
    df['month_name'] = pd.to_datetime(df['date']).dt.strftime('%B')
    return df

df = load_data()

# -----------------------------
# SIDEBAR FILTERS
st.sidebar.header("Filters")
districts = df['district'].unique().tolist()
selected_districts = st.sidebar.multiselect("Select district(s):", districts, default=districts)
months = df['month_name'].unique().tolist()
selected_months = st.sidebar.multiselect("Select month(s):", months, default=months)
pm_levels = st.sidebar.multiselect("Select PM2.5 levels:",
                                   ["Low (<15)","Moderate (15-25)","High (>25)"],
                                   default=["Low (<15)","Moderate (15-25)","High (>25)"])

df_filtered = df[(df['district'].isin(selected_districts)) & (df['month_name'].isin(selected_months))]

# -----------------------------
# KPI PANEL
st.subheader("Key Environmental Indicators")
col1, col2, col3 = st.columns(3)
col1.metric("Average PM2.5", f"{df_filtered['pm25'].mean():.1f}")
col2.metric("Most Polluted District", df_filtered.groupby('district')['pm25'].mean().idxmax())
col3.metric("Cleanest District", df_filtered.groupby('district')['pm25'].mean().idxmin())

# -----------------------------
# MAP
st.subheader("Interactive Pollution Map (Astana)")
m = folium.Map(location=[51.1694, 71.4491], zoom_start=11, tiles="CartoDB dark_matter")
heat_data = df_filtered[['latitude','longitude','pm25']].values.tolist()
HeatMap(heat_data, min_opacity=0.3, radius=15, blur=10, max_val=50).add_to(m)

for _, row in df_filtered.iterrows():
    if row['pm25']<15 and "Low (<15)" in pm_levels:
        color = "green"
    elif 15<=row['pm25']<=25 and "Moderate (15-25)" in pm_levels:
        color = "orange"
    elif row['pm25']>25 and "High (>25)" in pm_levels:
        color = "red"
    else:
        continue
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=12,
        color=color,
        fill=True,
        fill_opacity=0.8,
        popup=f"{row['district']} - {row['pm25']} µg/m³"
    ).add_to(m)

st_folium(m, width=900, height=600)

# -----------------------------
# GRAPHS
st.subheader("Air Quality & Green Spaces Analytics")
fig, axes = plt.subplots(2,2,figsize=(15,10))
sns.barplot(data=df_filtered.groupby('district')['pm25'].mean().reset_index(), x='district', y='pm25', ax=axes[0,0], palette="Reds")
axes[0,0].set_title("Average PM2.5 by District", color='white')
axes[0,0].set_ylabel("PM2.5")
sns.scatterplot(data=df_filtered.groupby('district').agg({'green_spaces':'first','pm25':'mean'}).reset_index(),
                x='green_spaces', y='pm25', hue='district', s=100, ax=axes[0,1'])
axes[0,1].set_title("Green Spaces vs PM2.5", color='white')
monthly_pm25 = df_filtered.groupby('month')['pm25'].mean().reset_index()
sns.lineplot(data=monthly_pm25, x='month', y='pm25', marker='o', ax=axes[1,0])
axes[1,0].set_xticks(range(1,13))
axes[1,0].set_title("Seasonal PM2.5 Patterns", color='white')
sns.boxplot(data=df_filtered, x='district', y='humidity', ax=axes[1,1], palette="Blues")
axes[1,1].set_title("Humidity by District", color='white')
plt.tight_layout()
st.pyplot(fig)

st.markdown("---")
st.markdown("<p style='text-align:center; color: gray;'>Interactive demo for NASA Hackathon 2025 – Amina Bilyalova</p>", unsafe_allow_html=True)

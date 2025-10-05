# =======================
# NASA Hackathon Streamlit App
# Author: Amina Bilyalova
# =======================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from streamlit_folium import st_folium
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# ---------- Page setup ----------
st.set_page_config(page_title="NASA Hackathon: Healthy Cities", layout="wide")

# ---------- CSS for black background and custom styles ----------
st.markdown(
    """
    <style>
    body {
        background-color: #000000;
        color: white;
    }
    .sidebar .sidebar-content {
        background-color: #111111;
        color: white;
    }
    .stButton>button {
        background-color: #0b3d91;
        color: white;
    }
    .stMarkdown h1, .stMarkdown h3 {
        color: white;
    }
    </style>
    """, unsafe_allow_html=True
)

# ---------- NASA Logo ----------
st.markdown(
    """
    <div style="text-align:center;">
        <img src="https://upload.wikimedia.org/wikipedia/commons/e/e5/NASA_logo.svg" width="200">
    </div>
    """, unsafe_allow_html=True
)

# ---------- Title ----------
st.markdown("<h1 style='text-align: center;'>NASA Hackathon: Healthy Cities</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: lightgray;'>Environmental Monitoring Dashboard | Amina Bilyalova</h3>", unsafe_allow_html=True)
st.markdown("<hr style='border:1px solid white'>", unsafe_allow_html=True)

# ---------- Month dictionary ----------
month_dict = {
    'January': 1, 'February': 2, 'March': 3, 'April': 4,
    'May': 5, 'June': 6, 'July': 7, 'August': 8,
    'September': 9, 'October': 10, 'November': 11, 'December': 12
}

# ---------- Dataset creation ----------
@st.cache_data
def create_data():
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
            pm25 = np.random.normal(20,5) + np.random.randint(-5,6)
            humidity = 60 + 20*np.sin(2*np.pi*i/365 + np.pi/2) + np.random.normal(0,5)
            traffic = np.random.randint(50000,100000)
            park_need = max(0, (pm25-10)*info['population']/10000)

            all_data.append({
                'date': date,
                'district': district,
                'temperature': round(base_temp + np.random.normal(0,2),1),
                'humidity': max(30,min(95,round(humidity,1))),
                'pm25': max(5, round(pm25,1)),
                'population': info['population'],
                'green_spaces': info['green_spaces'],
                'traffic_volume': traffic,
                'park_need_index': round(park_need,1),
                'latitude': info['lat'],
                'longitude': info['lon'],
                'month': date.month
            })
    return pd.DataFrame(all_data)

df = create_data()

# ---------- Sidebar Filters ----------
st.sidebar.header("Filters")
district_select = st.sidebar.multiselect("Select District(s)", options=df['district'].unique(), default=df['district'].unique())
month_select = st.sidebar.selectbox("Select Month", options=list(month_dict.keys()), index=0)
month_num = month_dict[month_select]

df_filtered = df[(df['month']==month_num) & (df['district'].isin(district_select))]

# ---------- Map ----------
st.subheader(f"Air Pollution Map - {month_select}")
m = folium.Map(location=[51.1694, 71.4491], zoom_start=11, tiles='CartoDB dark_matter')

for idx, row in df_filtered.iterrows():
    color = 'green' if row['pm25'] <= 15 else 'orange' if row['pm25'] <=25 else 'red'
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=8,
        color=color,
        fill=True,
        fill_opacity=0.7,
        popup=(f"{row['district']}<br>PM2.5: {row['pm25']} µg/m³<br>Temp: {row['temperature']}°C<br>Humidity: {row['humidity']}%")
    ).add_to(m)

st_folium(m, width=900, height=500)

# ---------- ML Model ----------
features = ['temperature', 'humidity', 'population', 'green_spaces', 'traffic_volume', 'month']
X = df[features]
y = df['pm25']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
st.subheader("ML Model Accuracy")
st.write(f"Random Forest MAE: {mae:.2f} µg/m³")

# ---------- Graphs ----------
st.subheader("Data Visualizations")
fig, axes = plt.subplots(2,2, figsize=(15,10))
fig.patch.set_facecolor('black')

# PM2.5 bar
pm25_avg = df_filtered.groupby('district')['pm25'].mean()
axes[0,0].bar(pm25_avg.index, pm25_avg.values, color=['green','orange','red','yellow','purple'][:len(pm25_avg)])
axes[0,0].set_title("Average PM2.5 by District", color='white')
axes[0,0].set_ylabel("PM2.5 µg/m³", color='white')
axes[0,0].tick_params(colors='white')

# Green spaces vs PM2.5
sns.scatterplot(data=df_filtered, x='green_spaces', y='pm25', hue='district', s=100, ax=axes[0,1])
axes[0,1].set_title("Green Spaces vs PM2.5", color='white')
axes[0,1].set_ylabel("PM2.5 µg/m³", color='white')
axes[0,1].tick_params(colors='white')

# Seasonal PM2.5
monthly_pm = df_filtered.groupby('month')['pm25'].mean()
axes[1,0].plot(monthly_pm.index, monthly_pm.values, marker='o', color='cyan')
axes[1,0].set_xticks(range(1,13))
axes[1,0].set_title("Seasonal PM2.5 Trend", color='white')
axes[1,0].set_xlabel("Month", color='white')
axes[1,0].set_ylabel("PM2.5 µg/m³", color='white')
axes[1,0].tick_params(colors='white')

# Feature importance
feat_import = pd.DataFrame({'feature': features, 'importance': model.feature_importances_}).sort_values('importance', ascending=True)
axes[1,1].barh(feat_import['feature'], feat_import['importance'], color='lightblue')
axes[1,1].set_title("Feature Importance for PM2.5 Prediction", color='white')
axes[1,1].tick_params(colors='white')

plt.tight_layout()
st.pyplot(fig)

# ---------- Recommendations ----------
st.subheader("Urban Planning Recommendations")
worst_district = df_filtered.groupby('district')['pm25'].mean().idxmax()
best_district = df_filtered.groupby('district')['pm25'].mean().idxmin()
st.markdown(f"- **Most Polluted District:** {worst_district} – focus on green infrastructure, traffic control, and air quality monitoring")
st.markdown(f"- **Best Practice District:** {best_district} – study and replicate strategies")

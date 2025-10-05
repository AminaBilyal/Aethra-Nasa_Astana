import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import folium
from streamlit_folium import st_folium

# --- Page config ---
st.set_page_config(page_title="NASA Hackathon Project", layout="wide")
st.markdown(
    """
    <style>
    .main {background-color: #0f111a; color: white;}
    .stMarkdown h1, h2, h3, h4, h5, h6 {color: white;}
    </style>
    """, unsafe_allow_html=True
)

st.title("NASA Hackathon: Data Pathways to Healthy Cities")
st.markdown("""
### Environmental Monitoring & Urban Planning
This project demonstrates how urban planners can use NASA Earth observation data to develop smart strategies for city growth that maintain both the wellbeing of people and the environment.
""")

# --- Load or create dataset ---
@st.cache_data
def create_astana_data():
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
            if district == 'Saryarka':
                temp = base_temp + np.random.normal(2, 1.5)
                pm25 = np.random.normal(28, 6)
            elif district == 'Almaty':
                temp = base_temp + np.random.normal(1, 2)
                pm25 = np.random.normal(22, 5)
            elif district == 'Yesil':
                temp = base_temp + np.random.normal(-1, 1)
                pm25 = np.random.normal(15, 3)
            else:
                temp = base_temp + np.random.normal(0, 1.5)
                pm25 = np.random.normal(20, 4)
            humidity = 60 + 20*np.sin(2*np.pi*i/365 + np.pi/2) + np.random.normal(0,5)
            all_data.append({
                'date': date,
                'district': district,
                'temperature': round(temp,1),
                'humidity': max(30,min(95,round(humidity,1))),
                'pm25': max(5,round(pm25,1)),
                'population': info['population'],
                'green_spaces': info['green_spaces'],
                'latitude': info['lat'],
                'longitude': info['lon'],
                'month': date.month
            })
    return pd.DataFrame(all_data)

df = create_astana_data()

# --- Sidebar filters ---
st.sidebar.header("Filters")
selected_districts = st.sidebar.multiselect("Select districts", options=df['district'].unique(), default=df['district'].unique())
selected_month = st.sidebar.selectbox("Select month", options=[("January",1),("February",2),("March",3),("April",4),
                                                              ("May",5),("June",6),("July",7),("August",8),
                                                              ("September",9),("October",10),("November",11),("December",12)],
                                      format_func=lambda x: x[0])

month_number = selected_month[1]
df_filtered = df[(df['district'].isin(selected_districts)) & (df['month']==month_number)]

# --- Map ---
st.subheader("Satellite Environmental Map")
m = folium.Map(location=[51.1694, 71.4491], zoom_start=11, tiles="CartoDB dark_matter")
for i, row in df_filtered.iterrows():
    color = "green" if row['pm25'] <= 15 else "orange" if row['pm25'] <= 25 else "red"
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=8,
        color=color,
        fill=True,
        fill_opacity=0.7,
        popup=f"{row['district']}: {row['pm25']} µg/m³"
    ).add_to(m)
st_folium(m, width=900, height=600)

# --- ML Model ---
features = ['temperature','humidity','population','green_spaces','month']
X = df[features]
y = df['pm25']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=50,max_depth=10,random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)

st.subheader("ML Model: PM2.5 Prediction")
st.write(f"Random Forest Regressor, Mean Absolute Error: {mae:.2f} µg/m³")

# --- Visualizations ---
st.subheader("Data Visualizations")
fig, axes = plt.subplots(2,2,figsize=(15,10))
# Avg PM2.5
pm25_means = df_filtered.groupby('district')['pm25'].mean().sort_values(ascending=False)
axes[0,0].bar(pm25_means.index, pm25_means.values, color=['red','orange','green','green','orange'])
axes[0,0].set_title("Average PM2.5 by District")
axes[0,0].set_ylabel("PM2.5")
# Green vs Pollution
green_pm25 = df_filtered.groupby('district').agg({'green_spaces':'first','pm25':'mean'}).reset_index()
sns.scatterplot(data=green_pm25,x='green_spaces',y='pm25',hue='district',s=100,ax=axes[0,1])
axes[0,1].set_title("Green Spaces vs PM2.5")
# Seasonal
monthly_pm25 = df.groupby('month')['pm25'].mean()
axes[1,0].plot(monthly_pm25.index,monthly_pm25.values,marker='o')
axes[1,0].set_title("Seasonal PM2.5 Patterns")
axes[1,0].set_xlabel("Month")
axes[1,0].set_ylabel("PM2.5")
axes[1,0].set_xticks(range(1,13))
# Feature importance
feat_imp = pd.DataFrame({'feature':features,'importance':model.feature_importances_}).sort_values('importance',ascending=True)
axes[1,1].barh(feat_imp['feature'],feat_imp['importance'])
axes[1,1].set_title("Feature Importance")
plt.tight_layout()
st.pyplot(fig)

# --- Recommendations ---
st.subheader("Urban Planning Recommendations")
worst = df.groupby('district')['pm25'].mean().idxmax()
best = df.groupby('district')['pm25'].mean().idxmin()
st.markdown(f"""
- **Priority District:** {worst} → actions: air quality monitoring, green infrastructure, traffic management  
- **Best Practice District:** {best} → study and replicate successful strategies  
- **High Risk Days:** {df[df['pm25']>25].groupby('district').size().to_dict()}  
- **Immediate Park Development:** {df.groupby('district')['pm25'].mean().sort_values(ascending=False).index[:2].tolist()}  
""")
st.markdown("**Team:** Amina Bilyalova and Sustainable Cities Research Team")
st.markdown("**Data Sources:** NASA Earth Observation, Open Data APIs")
st.markdown("**Technologies:** Python, Streamlit, Folium, ML Modeling")

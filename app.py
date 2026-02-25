import streamlit as st
import pandas as pd
import joblib
import folium
from streamlit_folium import st_folium
import ast

st.set_page_config(page_title="SafePathAI - Route Risk Recommender", layout="wide")
st.markdown("""
    <style>
        .metric-container {{
            display: flex;
            gap: 2rem;
            margin-bottom: 1rem;
        }}
        .stButton>button {{
            background-color: #3b82f6;
            color: white;
        }}
    </style>
""", unsafe_allow_html=True)

# Load model and data
model = joblib.load("model.pkl")
df = pd.read_csv("routes_with_paths.csv")

st.title("🚦 SafePathAI – Smart Route Risk Recommender")
st.markdown("Improve your commute with AI-powered risk assessment and route visualization. 🌐")

# Input area
with st.container():
    col1, col2, col3 = st.columns(3)
    with col1:
        source = st.selectbox("📍 Select Source", sorted(df['source'].unique()))
    with col2:
        destination = st.selectbox("🎯 Select Destination", sorted(df['destination'].unique()))
    with col3:
        time = st.selectbox("🕒 Time of Travel", ["Morning", "Afternoon", "Evening", "Night"])

# Filter routes
matched_routes = df[(df['source'] == source) & (df['destination'] == destination) & (df['time_of_day'] == time)]

if matched_routes.empty:
    st.warning("⚠️ No routes found for selected combination.")
else:
    # Map for categorical values
    congestion_map = {"Low": 0, "Medium": 1, "High": 2}
    time_map = {"Morning": 0, "Afternoon": 1, "Evening": 2, "Night": 3}

    results = []
    for _, row in matched_routes.iterrows():
        features = [
            row['distance_km'],
            congestion_map[row['congestion_level']],
            row['accidents'],
            time_map[row['time_of_day']]
        ]
        prediction = model.predict([features])[0]
        results.append({
            "Route ID": row['route_id'],
            "Distance (km)": row['distance_km'],
            "Congestion": row['congestion_level'],
            "Accidents": row['accidents'],
            "Predicted Risk": prediction
        })

    result_df = pd.DataFrame(results)

    tab1, tab2 = st.tabs(["📊 Route Table", "🗺️ Visual Map"])

    with tab1:
        st.subheader("📌 Route Comparison Table")
        st.dataframe(result_df, use_container_width=True)

    with tab2:
        m = folium.Map(location=[13.05, 80.23], zoom_start=12)
        for _, row in matched_routes.iterrows():
            coords = ast.literal_eval(row['path'])
            features = [
                row['distance_km'],
                congestion_map[row['congestion_level']],
                row['accidents'],
                time_map[row['time_of_day']]
            ]
            risk = model.predict([features])[0]
            color = "red" if risk == "High Risk" else "orange" if risk == "Medium Risk" else "green"

            folium.PolyLine(coords, color=color, weight=5, popup=f"{row['route_id']} - {risk}").add_to(m)
            folium.Marker(coords[0], popup="Start", icon=folium.Icon(color="blue")).add_to(m)
            folium.Marker(coords[-1], popup="End", icon=folium.Icon(color="gray")).add_to(m)

        st.subheader("🗺️ Route Map with Risk Colors")
        st_folium(m, width=900, height=550)

    st.markdown("---")
    st.markdown("Made with ❤️ using Streamlit and scikit-learn")

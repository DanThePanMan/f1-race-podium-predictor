import streamlit as st
import requests
import pandas as pd

st.title("F1 Podium Prediction")

# API URL configuration
env = st.radio("Environment", ["Local", "Production"], horizontal=True)
if env == "Local":
    API_URL = "http://localhost:9696/predict"
else:
    API_URL = "https://f1-race-podium-predictor.onrender.com/predict"

st.divider()

# Load data
@st.cache_data
def load_data():
    drivers = pd.read_csv('data/drivers.csv')
    circuits = pd.read_csv('data/circuits.csv')
    constructors = pd.read_csv('data/constructors.csv')
    return drivers, circuits, constructors

drivers_df, circuits_df, constructors_df = load_data()

# Create display options
driver_options = {f"{row['forename']} {row['surname']}": row['driverRef'] 
                  for _, row in drivers_df.iterrows()}
circuit_options = {row['name']: row['circuitRef'] 
                   for _, row in circuits_df.iterrows()}
constructor_options = {row['name']: row['constructorRef'] 
                       for _, row in constructors_df.iterrows()}

# Input fields
driver_info = {}

driver_name = st.selectbox("Driver", [""] + list(driver_options.keys()))
if driver_name:
    driver_info['driverRef'] = driver_options[driver_name]

constructor_name = st.selectbox("Constructor", [""] + list(constructor_options.keys()))
if constructor_name:
    driver_info['constructorRef'] = constructor_options[constructor_name]

driver_info['grid'] = st.number_input("Grid Position", min_value=1, max_value=20, value=1)
driver_info['year'] = st.number_input("Year", min_value=1980, max_value=2030, value=2026)
driver_info['round'] = st.number_input("Round", min_value=1, max_value=24, value=1)
driver_info['podium_rate_last_year'] = st.slider("Podium Rate Last Year", 0.0, 1.0, 0.0, 0.01)
driver_info['podium_rate_curr_year'] = st.slider("Podium Rate Current Year", 0.0, 1.0, 0.0, 0.01)
driver_info['podium_rate_all_time'] = st.slider("Podium Rate All Time", 0.0, 1.0, 0.0, 0.01)

if st.button("Predict"):
    filtered_info = {k: v for k, v in driver_info.items() if v != ""}
    
    try:
        response = requests.post(API_URL, json=filtered_info)
        
        if response.status_code == 200:
            result = response.json()
            probability = result.get("podium probability", 0)
            podium = result.get("podium", False)
            
            st.metric("Podium Probability", f"{probability:.1%}")
            st.write("Prediction:", "PODIUM" if podium else "NO PODIUM")
        else:
            st.error(f"API Error: {response.status_code}")
    except:
        st.error(f"Cannot connect to API at {API_URL}")

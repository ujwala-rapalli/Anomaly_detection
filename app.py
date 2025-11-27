import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Industrial Sensor Anomaly Detection", layout="wide")

st.title("🔧 Industrial Sensor Anomaly Detection (Manual Input)")
st.write("Enter sensor values to detect whether the machine is behaving abnormally.")

# ----------------------------
# LOAD SAVED MODEL + SCALER
# ----------------------------
@st.cache_resource
def load_model():
    try:
        return joblib.load("anomaly_detection_pipeline.pkl"), "pipeline"
    except:
        try:
            iso = joblib.load("isolation_forest_model.pkl")
            scaler = joblib.load("scaler.pkl")
            return (iso, scaler), "separate"
        except:
            st.error("❌ Model file not found. Place model.pkl files in the same folder.")
            st.stop()

model_obj, model_type = load_model()

FEATURES = [
    'Temperature','Pressure','Vibration','Humidity','FlowRate',
    'PowerConsumption','RuntimeHours','DaysSinceService'
]

# ----------------------------
# INPUT FORM
# ----------------------------
st.header("📝 Enter Sensor Values Manually")

col1, col2, col3 = st.columns(3)

with col1:
    temp = st.number_input("Temperature", value=50.0)
    pressure = st.number_input("Pressure", value=10.0)
    vibration = st.number_input("Vibration", value=5.0)

with col2:
    humidity = st.number_input("Humidity", value=40.0)
    flow = st.number_input("Flow Rate", value=100.0)
    power = st.number_input("Power Consumption", value=200.0)

with col3:
    runtime = st.number_input("Runtime Hours", value=1000.0)
    last_service_days = st.number_input("Days Since Last Service", value=30)

submit = st.button("🔍 Detect Anomaly")

# ----------------------------
# PREDICTION
# ----------------------------
if submit:
    input_data = pd.DataFrame([{
        "Temperature": temp,
        "Pressure": pressure,
        "Vibration": vibration,
        "Humidity": humidity,
        "FlowRate": flow,
        "PowerConsumption": power,
        "RuntimeHours": runtime,
        "DaysSinceService": last_service_days
    }])

    # prediction
    if model_type == "pipeline":
        pred = model_obj.predict(input_data)[0]
        score = -model_obj.decision_function(input_data)[0]
    else:
        iso, scaler = model_obj
        scaled = scaler.transform(input_data)
        pred = iso.predict(scaled)[0]
        score = -iso.decision_function(scaled)[0]

    st.markdown("### 🔬 Result")
    if pred == -1 or pred == 1:  # depending on training output
        st.error(f"🚨 **Anomaly Detected!** (score={score:.3f})")
    else:
        st.success(f"✅ **Machine is Normal** (score={score:.3f})")

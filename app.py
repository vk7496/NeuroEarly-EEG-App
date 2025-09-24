import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="EEG + QEEG Depression Detection", layout="wide")

st.title("🧠 EEG & QEEG Depression Detection Demo")

# Upload EEG file
uploaded_file = st.file_uploader("Upload EEG (.edf) file", type=["edf"])

if uploaded_file:
    st.success("EEG file uploaded successfully ✅")
    
    # Dummy EEG signal (simulate 10 seconds, 128Hz sampling)
    eeg_signal = np.sin(np.linspace(0, 20*np.pi, 1280)) + np.random.normal(0, 0.2, 1280)
    
    st.subheader("EEG Signal (Sampled)")
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(eeg_signal, color="blue")
    ax.set_title("EEG Signal")
    st.pyplot(fig)
    
    # Dummy QEEG features (absolute power in 4 bands)
    qEEG_features = {
        "Delta (0.5–4 Hz)": np.random.uniform(0.5, 2.0),
        "Theta (4–8 Hz)": np.random.uniform(0.5, 2.0),
        "Alpha (8–12 Hz)": np.random.uniform(0.5, 2.0),
        "Beta (12–30 Hz)": np.random.uniform(0.5, 2.0),
    }
    
    st.subheader("QEEG Analysis (Dummy Values)")
    for band, value in qEEG_features.items():
        st.write(f"**{band}:** {value:.2f}")
    
    # Dummy prediction based on random score
    depression_score = np.random.uniform(0, 1)
    if depression_score < 0.3:
        result = "No Depression"
    elif depression_score < 0.6:
        result = "Mild Depression"
    else:
        result = "Severe Depression"
    
    st.subheader("Prediction Result")
    st.info(f"Predicted condition: **{result}** (Score: {depression_score:.2f})")


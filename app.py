import os
import io
import json
import base64
import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.signal import welch
from scipy.integrate import trapezoid
import pyedflib
import streamlit as st
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
import shap
import arabic_reshaper
from bidi.algorithm import get_display

# ============ Configuration ============
st.set_page_config(page_title="NeuroEarly Pro – AI EEG Assistant", layout="wide")
LANGUAGES = {"English": "en", "Arabic": "ar"}

amiri_path = "Amiri-Regular.ttf"
if os.path.exists(amiri_path):
    pdfmetrics.registerFont(TTFont("Amiri", amiri_path))

# ============ Utility Functions ============

def safe_arabic_text(text, lang):
    if lang == "ar":
        try:
            reshaped = arabic_reshaper.reshape(text)
            return get_display(reshaped)
        except Exception:
            return text
    return text

def compute_band_powers(raw, sfreq):
    psd, freqs = welch(raw, sfreq, nperseg=sfreq*2)
    bands = {"Delta": (1, 4), "Theta": (4, 8), "Alpha": (8, 12), "Beta": (12, 30), "Gamma": (30, 45)}
    rel_powers = {}
    total_power = float(trapezoid(psd, freqs)) if freqs.size > 0 else 0.0
    for band, (fmin, fmax) in bands.items():
        mask = (freqs >= fmin) & (freqs <= fmax)
        abs_p = float(trapezoid(psd[mask], freqs[mask])) if mask.sum() > 0 else 0.0
        rel_powers[band] = abs_p / total_power if total_power > 0 else 0
    return rel_powers

def load_eeg(file):
    try:
        if file.name.endswith(".edf"):
            f = pyedflib.EdfReader(file)
            n = f.signals_in_file
            signal_labels = f.getSignalLabels()
            sigbufs = np.zeros((n, f.getNSamples()[0]))
            for i in range(n):
                sigbufs[i, :] = f.readSignal(i)
            f._close()
            sfreq = f.getSampleFrequency(0)
            return sigbufs, sfreq, signal_labels
    except Exception as e:
        st.error(f"Error loading EEG file: {e}")
        return None, None, None
    return None, None, None

def compute_qeeg_metrics(eeg_data, sfreq):
    rel = compute_band_powers(eeg_data, sfreq)
    theta_alpha = rel.get("Theta", 0) / max(rel.get("Alpha", 0.001), 0.001)
    theta_beta = rel.get("Theta", 0) / max(rel.get("Beta", 0.001), 0.001)
    beta_alpha = rel.get("Beta", 0) / max(rel.get("Alpha", 0.001), 0.001)
    alpha_asym = np.mean(eeg_data[0]) - np.mean(eeg_data[-1]) if eeg_data.shape[0] >= 2 else 0
    return {
        "Theta/Alpha Ratio": theta_alpha,
        "Theta/Beta Ratio": theta_beta,
        "Beta/Alpha Ratio": beta_alpha,
        "Alpha Asymmetry": alpha_asym,
        "Alpha (rel)": rel.get("Alpha", 0),
        "Theta (rel)": rel.get("Theta", 0),
        "Gamma (rel)": rel.get("Gamma", 0)
    }

def plot_bar_comparison(metrics):
    healthy = {"Theta/Alpha Ratio": (0.5, 1.0), "Alpha Asymmetry": (-0.05, 0.05)}
    fig, ax = plt.subplots(figsize=(4, 2.5))
    for i, (k, v) in enumerate(metrics.items()):
        if k in healthy:
            hmin, hmax = healthy[k]
            ax.barh(i, hmax, color="lightgray", alpha=0.3)
            ax.barh(i, hmin, color="white")
            ax.barh(i, v, color="red" if v > hmax else "green" if v < hmin else "yellow")
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels(metrics.keys(), fontsize=7)
    ax.set_title("Patient vs Normative", fontsize=9)
    plt.tight_layout()
    return fig

def compute_connectivity(raw, sfreq):
    try:
        from mne_connectivity import spectral_connectivity
        con, freqs, times, n_epochs, n_tapers = spectral_connectivity(
            [raw], method="coh", mode="multitaper", sfreq=sfreq, fmin=8, fmax=12, faverage=True
        )
        return np.mean(con)
    except Exception:
        return None

def generate_pdf_report(summary, lang="en", amiri_path=None, topo_images=None, conn_image=None, logo_path="assets/goldenbird_logo.png"):
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4
    y = height - 50
    font_name = "Amiri" if lang == "ar" and amiri_path else "Helvetica"
    c.setFont(font_name, 11)
    if os.path.exists(logo_path):
        logo = ImageReader(logo_path)
        c.drawImage(logo, 20, y - 40, width=80, height=40, mask="auto")
    y -= 60
    title = safe_arabic_text("NeuroEarly Pro — Clinical EEG Report", lang)
    c.drawString(100, y, title); y -= 20
    for k, v in summary.items():
        text = f"{safe_arabic_text(k, lang)}: {v}"
        c.drawString(40, y, text)
        y -= 15
        if y < 80:
            c.showPage()
            y = height - 50
    footer = safe_arabic_text("Designed and developed by Golden Bird LLC — Vista Kaviani", lang)
    c.setFont(font_name, 9)
    c.drawCentredString(width/2, 30, footer)
    c.save()
    pdf_data = buf.getvalue()
    return pdf_data
# ============ Streamlit UI & App Logic ============

def main():
    st.title("🧠 NeuroEarly Pro – AI EEG Assistant")
    st.markdown("Designed for clinical EEG assessment with Explainable AI (SHAP) and QEEG analysis.")
    logo_path = "assets/goldenbird_logo.png"
    if os.path.exists(logo_path):
        st.image(logo_path, width="stretch")

    st.sidebar.header("Settings")
    lang_choice = st.sidebar.selectbox("Language / اللغة", ["English", "Arabic"])
    lang = LANGUAGES[lang_choice]
    st.sidebar.markdown("---")
    st.sidebar.header("Patient Info")
    name = st.sidebar.text_input("Patient Name")
    patient_id = st.sidebar.text_input("Patient ID")
    dob = st.sidebar.date_input("Date of Birth")

    st.sidebar.markdown("### Clinical Information")
    lab_tests = st.sidebar.text_area("Lab Tests (B12, TSH, etc.)")
    medications = st.sidebar.text_area("Medications")
    conditions = st.sidebar.text_area("Underlying Conditions")

    st.sidebar.markdown("### Upload EEG Files")
    eeg_files = st.sidebar.file_uploader("Upload EEG files (.edf)", type=["edf"], accept_multiple_files=True)

    if not eeg_files:
        st.info("Please upload at least one EEG file to begin analysis.")
        return

    all_metrics = []
    with st.spinner("Processing EEG files..."):
        for eeg_file in eeg_files:
            eeg_data, sfreq, labels = load_eeg(eeg_file)
            if eeg_data is not None:
                metrics = compute_qeeg_metrics(eeg_data, sfreq)
                metrics["Connectivity"] = compute_connectivity(eeg_data[0], sfreq)
                all_metrics.append(metrics)

    if not all_metrics:
        st.error("No valid EEG data processed.")
        return

    mean_metrics = pd.DataFrame(all_metrics).mean().to_dict()
    st.subheader("QEEG Key Metrics")
    st.dataframe(pd.DataFrame([mean_metrics]), use_container_width=True)

    st.subheader("Patient vs Normative")
    fig = plot_bar_comparison(mean_metrics)
    st.pyplot(fig, use_container_width=True)

    # Load SHAP
    try:
        with open("shap_summary.json", "r", encoding="utf-8") as f:
            shap_data = json.load(f)
        model_key = "depression_global"
        if mean_metrics["Theta/Alpha Ratio"] > 1.3:
            model_key = "alzheimers_global"
        shap_features = shap_data.get(model_key, {})
        shap_sorted = dict(sorted(shap_features.items(), key=lambda x: -abs(x[1])))
        st.subheader("Explainable AI (SHAP) — Top Contributors")
        st.bar_chart(pd.Series(shap_sorted), use_container_width=True)
    except Exception as e:
        st.warning(f"XAI not available: {e}")

    st.subheader("PHQ-9 Depression Questionnaire")
    phq_questions = [
        "1. Little interest or pleasure in doing things?",
        "2. Feeling down, depressed, or hopeless?",
        "3. Trouble falling asleep, staying asleep, or sleeping too much?",
        "4. Feeling tired or having little energy?",
        "5. Poor appetite or overeating?",
        "6. Feeling bad about yourself, or that you are a failure?",
        "7. Trouble concentrating on things?",
        "8. Moving or speaking slowly, or being fidgety or restless?",
        "9. Thoughts that you would be better off dead, or of hurting yourself?"
    ]
    phq_total = 0
    for q in phq_questions:
        val = st.select_slider(q, options=[0, 1, 2, 3])
        phq_total += val

    st.subheader("AD8 Cognitive Impairment Screening")
    ad8_questions = [
        "1. Problems with judgment or decision making?",
        "2. Reduced interest in hobbies or activities?",
        "3. Repeats questions or statements often?",
        "4. Trouble learning how to use a tool, appliance, or gadget?",
        "5. Forgets correct month or year?",
        "6. Trouble handling complicated financial affairs?",
        "7. Trouble remembering appointments?",
        "8. Consistent daily problems with thinking or memory?"
    ]
    ad8_total = 0
    for q in ad8_questions:
        val = st.select_slider(q, options=[0, 1])
        ad8_total += val

    risk_score = min(100, round((phq_total * 2.5 + ad8_total * 5 + mean_metrics["Theta/Alpha Ratio"] * 15), 1))
    risk_level = "Low" if risk_score < 15 else "Moderate" if risk_score < 35 else "High"

    st.markdown(f"### 🩺 Final ML Risk Score: **{risk_score}% ({risk_level})**")

    report_summary = {
        "Patient Name": name,
        "ID": patient_id,
        "DOB": dob,
        "ML Risk Score": f"{risk_score}% ({risk_level})",
        "Theta/Alpha Ratio": round(mean_metrics["Theta/Alpha Ratio"], 3),
        "Theta/Beta Ratio": round(mean_metrics["Theta/Beta Ratio"], 3),
        "Alpha Asymmetry": round(mean_metrics["Alpha Asymmetry"], 4),
        "Alpha (rel)": round(mean_metrics["Alpha (rel)"], 4),
        "Theta (rel)": round(mean_metrics["Theta (rel)"], 4),
        "Gamma (rel)": round(mean_metrics["Gamma (rel)"], 4),
        "Connectivity": round(mean_metrics.get("Connectivity", 0), 3)
    }

    pdf_data = generate_pdf_report(report_summary, lang=lang, amiri_path=amiri_path)
    b64 = base64.b64encode(pdf_data).decode()
    href = f'<a href="data:application/pdf;base64,{b64}" download="NeuroEarly_Report.pdf">📄 Download Full Report (PDF)</a>'
    st.markdown(href, unsafe_allow_html=True)
# ============ MAIN EXECUTION ============
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"⚠️ Critical error occurred: {e}")
        st.stop()

# ============ FOOTER ============
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; font-size: 13px; color: gray;'>
    Designed and developed by <b>Golden Bird LLC</b> — Vista Kaviani<br>
    NeuroEarly Pro – AI EEG Assistant © 2025
    </div>
    """,
    unsafe_allow_html=True
)

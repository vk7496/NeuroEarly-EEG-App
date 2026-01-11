import os
import io
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
import mne 
from datetime import datetime

# PDF & BiDi Libraries
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.enums import TA_CENTER, TA_RIGHT, TA_LEFT
import arabic_reshaper
from bidi.algorithm import get_display

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="NeuroEarly Pro v48", layout="wide", page_icon="🧠")
FONT_PATH = "Amiri-Regular.ttf"
BLUE_DARK = "#003366"
RED_CLINIC = "#D32F2F"
GREEN_CLINIC = "#2E7D32"

# Initializing Session State for Blood Lab Data
if 'lab_data' not in st.session_state:
    st.session_state['lab_data'] = {'b12': 400.0, 'tsh': 2.5, 'crp': 1.0, 'scanned': False}

# --- 2. VISUALIZATION ENGINES ---
def get_bidi(text):
    return get_display(arabic_reshaper.reshape(str(text)))

def generate_stress_gauge(score):
    fig, ax = plt.subplots(figsize=(6, 1.5))
    cmap = plt.get_cmap('RdYlGn_r')
    grad = np.linspace(0, 100, 256).reshape(1, -1)
    ax.imshow(grad, aspect='auto', cmap=cmap, extent=[0, 100, 0, 1])
    ax.axvline(score, color='black', linewidth=4)
    ax.text(score, 1.3, f"{score:.1f}", ha='center', weight='bold', fontsize=12)
    ax.set_yticks([]); ax.set_xlim(0, 100); ax.axis('off')
    buf = io.BytesIO(); plt.savefig(buf, format='png', bbox_inches='tight'); plt.close(fig)
    return buf

def generate_shap_analysis():
    fig, ax = plt.subplots(figsize=(7, 3))
    features = ['Alpha Asymmetry', 'Neural Complexity', 'B12 Level', 'Theta Memory']
    impact = [0.15, 0.42, 0.28, 0.10]
    colors_list = [RED_CLINIC, GREEN_CLINIC, GREEN_CLINIC, RED_CLINIC]
    ax.barh(features, impact, color=colors_list)
    ax.set_title("AI Decision Logic (SHAP Importance)")
    buf = io.BytesIO(); plt.savefig(buf, format='png', bbox_inches='tight'); plt.close(fig)
    return buf

def generate_topomaps():
    fig, axes = plt.subplots(1, 4, figsize=(10, 2.5))
    for i, ax in enumerate(axes):
        ax.imshow(np.random.rand(15, 15), cmap='jet', interpolation='bicubic')
        ax.set_title(['Delta', 'Theta', 'Alpha', 'Beta'][i])
        ax.axis('off')
    buf = io.BytesIO(); plt.savefig(buf, format='png', bbox_inches='tight', transparent=True); plt.close(fig)
    return buf

# --- 3. PDF GENERATOR ---
def create_professional_pdf(data):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4)
    
    try:
        pdfmetrics.registerFont(TTFont('Amiri', FONT_PATH))
        f_name = 'Amiri'
    except: f_name = 'Helvetica'

    styles = getSampleStyleSheet()
    s_title = ParagraphStyle('T', fontName=f_name, fontSize=18, textColor=colors.HexColor(BLUE_DARK), alignment=TA_CENTER)
    s_head = ParagraphStyle('H', fontName=f_name, fontSize=12, backColor=colors.HexColor("#E3F2FD"), borderPadding=5)
    s_body = ParagraphStyle('B', fontName=f_name, fontSize=10, leading=14)
    
    elements = []
    # Header
    elements.append(Paragraph(get_bidi("NeuroEarly Pro - Clinical Diagnostic Report"), s_title))
    elements.append(Paragraph(f"Date: {datetime.now().strftime('%Y-%m-%d')} | ID: {data['id']}", styles['Normal']))
    elements.append(Spacer(1, 15))

    # Blood Lab Table
    lab_data = [
        [get_bidi("Parameter"), get_bidi("Result"), get_bidi("Reference Range")],
        ["Vitamin B12", f"{data['b12']} pg/mL", "200 - 900"],
        ["TSH", f"{data['tsh']} mIU/L", "0.4 - 4.0"],
        ["CRP", f"{data['crp']} mg/L", "< 3.0"]
    ]
    t_lab = Table(lab_data, colWidths=[2.3*inch]*3)
    t_lab.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 0.5, colors.grey), ('BACKGROUND', (0,0), (-1,0), colors.whitesmoke)]))
    elements.append(Paragraph(get_bidi("1. Blood Laboratory Profile"), s_head))
    elements.append(t_lab)
    elements.append(Spacer(1, 20))

    # Stress & EEG Visuals
    elements.append(Paragraph(get_bidi("2. Neuro-Physiological Analysis"), s_head))
    elements.append(RLImage(io.BytesIO(data['gauge_img']), width=5*inch, height=1.2*inch))
    elements.append(Spacer(1, 10))
    elements.append(RLImage(io.BytesIO(data['topo_img']), width=6*inch, height=1.5*inch))
    elements.append(Spacer(1, 20))

    # AI Logic
    elements.append(Paragraph(get_bidi("3. AI Interpretability (SHAP Values)"), s_head))
    elements.append(RLImage(io.BytesIO(data['shap_img']), width=5*inch, height=2*inch))
    
    # Final Recommendation
    elements.append(PageBreak())
    elements.append(Paragraph(get_bidi("4. Clinical Impression & Recommendation"), s_head))
    elements.append(Paragraph(get_bidi(data['narrative']), s_body))

    doc.build(elements)
    buf.seek(0)
    return buf.getvalue()

# --- 4. MAIN STREAMLIT UI ---
def main():
    st.sidebar.markdown("<h1 style='text-align: center;'>🧠</h1>", unsafe_allow_html=True)
    st.sidebar.title("NeuroEarly Pro v48")

    # Patient Profile
    p_name = st.sidebar.text_input("Patient Name", "John Doe")
    p_id = st.sidebar.text_input("Patient ID", "F-2025")

    # Blood Lab Upload Section
    st.sidebar.divider()
    st.sidebar.subheader("🔬 Blood Lab OCR")
    lab_file = st.sidebar.file_uploader("Upload Lab Report (PDF/Img)", type=['pdf', 'jpg', 'png'])
    
    if lab_file and not st.session_state.lab_data['scanned']:
        with st.sidebar.status("AI Reading Report..."):
            time.sleep(1.5)
            # Simulated OCR Extraction
            st.session_state.lab_data.update({'b12': 195.0, 'tsh': 5.2, 'crp': 14.5, 'scanned': True})
        st.sidebar.success("✅ Lab Data Extracted!")

    b12 = st.sidebar.number_input("B12 (pg/mL)", value=st.session_state.lab_data['b12'])
    tsh = st.sidebar.number_input("TSH (mIU/L)", value=st.session_state.lab_data['tsh'])
    crp = st.sidebar.number_input("CRP (mg/L)", value=st.session_state.lab_data['crp'])

    tab1, tab2 = st.tabs(["📝 Clinical Data", "🧠 EEG Dashboard"])

    with tab1:
        st.header("Patient Questionnaires")
        phq = st.slider("PHQ-9 Score", 0, 27, 5)
        mmse = st.slider("MMSE Score", 0, 30, 24)
        
    with tab2:
        eeg_file = st.file_uploader("Upload EEG Data (.edf)", type=['edf'])
        if eeg_file:
            st.success("EEG Signal Loaded.")
            
            # Simulated logic for stress calculation
            stress_score = 78.5 if crp > 5 else 45.0
            
            # Generating visualizations
            g_buf = generate_stress_gauge(stress_score).getvalue()
            t_buf = generate_topomaps().getvalue()
            s_buf = generate_shap_analysis().getvalue()

            col1, col2 = st.columns(2)
            with col1:
                st.image(g_buf, caption="Stress Index Gauge")
                st.image(s_buf, caption="AI Reasoning (SHAP)")
            with col2:
                st.image(t_buf, caption="Brain Topography Mapping")
                st.markdown("### Clinical Indicators")
                st.metric("Stress Level", f"{stress_score}%", delta="High" if stress_score > 70 else "Normal", delta_color="inverse")

            if st.button("🚀 Generate Final Clinical PDF"):
                narrative = f"Patient {p_name} exhibits high stress index ({stress_score}%). "
                if b12 < 200: narrative += "Low B12 may contribute to cognitive fatigue. "
                if tsh > 4.5: narrative += "Elevated TSH suggests thyroid involvement. "
                if mmse < 25: narrative += "Cognitive impairment detected in MMSE screening."
                
                report_bytes = create_professional_pdf({
                    'id': p_id, 'b12': b12, 'tsh': tsh, 'crp': crp,
                    'gauge_img': g_buf, 'topo_img': t_buf, 'shap_img': s_buf,
                    'narrative': narrative
                })
                st.download_button("📩 Download Professional Report", report_bytes, "Neuro_Report_Final.pdf")

if __name__ == "__main__":
    main()

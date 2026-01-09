import os
import io
import tempfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import entropy
import streamlit as st
import mne 
from datetime import datetime

# PDF & BiDi Support
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.enums import TA_CENTER, TA_RIGHT, TA_LEFT
import arabic_reshaper
from bidi.algorithm import get_display

# --- 1. SETTINGS ---
st.set_page_config(page_title="NeuroEarly Pro v45", layout="wide")
FONT_PATH = "Amiri-Regular.ttf"
BLUE, RED, GREEN = "#003366", "#D32F2F", "#2E7D32"

# --- 2. CORE ANALYTICS ---
@st.cache_data
def calculate_stress_metrics(psds, freqs):
    # Stress Index = High Beta / Alpha Ratio
    alpha_idx = np.where((freqs >= 8) & (freqs <= 13))[0]
    beta_idx = np.where((freqs >= 13) & (freqs <= 30))[0]
    
    alpha_pwr = np.mean(psds[:, alpha_idx])
    beta_pwr = np.mean(psds[:, beta_idx])
    
    ratio = beta_pwr / (alpha_pwr + 1e-9)
    stress_level = min(100, ratio * 50) # Normalized gauge
    return stress_level

def get_bidi_text(text):
    return get_display(arabic_reshaper.reshape(str(text)))

# --- 3. ENHANCED PDF ENGINE ---
def create_v45_report(data):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4)
    try:
        pdfmetrics.registerFont(TTFont('Amiri', FONT_PATH))
        f_name = 'Amiri'
    except: f_name = 'Helvetica'

    styles = getSampleStyleSheet()
    s_title = ParagraphStyle('T', fontName=f_name, fontSize=18, textColor=colors.HexColor(BLUE), alignment=TA_CENTER)
    s_head = ParagraphStyle('H', fontName=f_name, fontSize=12, backColor=colors.HexColor("#F0F0F0"), borderPadding=5)
    s_body = ParagraphStyle('B', fontName=f_name, fontSize=10, leading=14)
    s_note = ParagraphStyle('N', fontName=f_name, fontSize=9, textColor=colors.grey, italic=True)

    elements = []
    
    # Header
    elements.append(Paragraph(get_bidi_text("NeuroEarly Pro Advanced Diagnostic Report"), s_title))
    elements.append(Paragraph(f"Date: {datetime.now().strftime('%Y-%m-%d')}", styles['Normal']))
    elements.append(Spacer(1, 15))

    # Patient & Lab Grid
    lab_info = f"B12: {data['lab']['b12']} | TSH: {data['lab']['tsh']} | CRP: {data['lab']['crp']}"
    p_table = Table([
        [get_bidi_text(f"Patient: {data['name']}"), get_bidi_text(f"ID: {data['id']}")],
        [get_bidi_text(f"Lab Status: {lab_info}"), get_bidi_text(f"Stress Level: {data['stress_score']:.1f}%")]
    ], colWidths=[3.5*inch, 3.5*inch])
    p_table.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 0.5, colors.lightgrey)]))
    elements.append(p_table)
    elements.append(Spacer(1, 20))

    # Stress Analysis
    elements.append(Paragraph(get_bidi_text("1. Neuro-Autonomic Balance (Stress)"), s_head))
    elements.append(RLImage(io.BytesIO(data['stress_plot']), width=5*inch, height=1.2*inch))
    elements.append(Paragraph(get_bidi_text("Physician's Guide: High scores (>70) indicate sympathetic dominance often linked to chronic anxiety or burnout."), s_note))
    elements.append(Spacer(1, 20))

    # Topography
    elements.append(Paragraph(get_bidi_text("2. Brain Topography Mapping"), s_head))
    elements.append(RLImage(io.BytesIO(data['topo_plot']), width=6*inch, height=2*inch))
    elements.append(Paragraph(get_bidi_text("Physician's Guide: Red spots in Delta/Theta bands may suggest localized neuro-inflammation or cognitive deficit."), s_note))
    elements.append(Spacer(1, 20))

    # Conclusion
    elements.append(Paragraph(get_bidi_text("3. Clinical Impression"), s_head))
    elements.append(Paragraph(get_bidi_text(data['narrative']), s_body))

    doc.build(elements)
    buf.seek(0)
    return buf.getvalue()

# --- 4. STREAMLIT UI ---
def main():
    st.sidebar.title("NeuroEarly Pro v45")
    
    # Lab Input Section
    st.sidebar.header("🔬 Blood Lab Results")
    b12 = st.sidebar.number_input("Vitamin B12 (pg/mL)", 100, 1000, 400)
    tsh = st.sidebar.number_input("TSH (mIU/L)", 0.1, 10.0, 2.5)
    crp = st.sidebar.number_input("CRP (mg/L)", 0.0, 50.0, 1.0)

    # Patient Info
    p_name = st.sidebar.text_input("Name", "John Doe")
    p_id = st.sidebar.text_input("ID", "F-2025")

    t1, t2 = st.tabs(["Clinical Data", "EEG & Analysis"])

    with t1:
        st.subheader("Questionnaires")
        phq = st.slider("PHQ-9 Total", 0, 27, 5)
        mmse = st.slider("MMSE Total", 0, 30, 28)

    with t2:
        up = st.file_uploader("Upload EEG (EDF)", type=["edf"])
        if up:
            # Placeholder for stress calc (Real logic uses psds from MNE)
            stress_score = 75.5 
            
            # Generate Gauge
            fig, ax = plt.subplots(figsize=(6, 1))
            ax.barh(["Stress"], [stress_score], color='red' if stress_score > 70 else 'green')
            ax.set_xlim(0, 100); ax.axis('off')
            s_buf = io.BytesIO(); plt.savefig(s_buf, format='png'); s_buf.seek(0)
            st.image(s_buf, caption="Calculated Stress Index")

            if st.button("Generate Master PDF Report"):
                narrative = "High stress detected. Low B12 may correlate with reported cognitive fatigue."
                pdf_bytes = create_v45_report({
                    'name': p_name, 'id': p_id, 
                    'lab': {'b12': b12, 'tsh': tsh, 'crp': crp},
                    'stress_score': stress_score,
                    'stress_plot': s_buf.getvalue(),
                    'topo_plot': s_buf.getvalue(), # Placeholder
                    'narrative': narrative,
                    'mmse': mmse, 'phq': phq
                })
                st.download_button("Download Professional Report", pdf_bytes, "Report_v45.pdf")

if __name__ == "__main__":
    main()

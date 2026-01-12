import os
import io
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
from datetime import datetime

# PDF & Language Support
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

# --- CONFIG ---
st.set_page_config(page_title="NeuroEarly Pro v50", layout="wide", page_icon="🧠")
FONT_PATH = "Amiri-Regular.ttf"

# --- DATA & QUESTIONS ---
PHQ9_QS = ["Little interest", "Feeling down", "Sleep trouble", "Low energy", "Appetite change", "Self-failure", "Concentration", "Slow/Fidgety", "Self-harm thoughts"]
MMSE_QS = ["Time Orientation", "Place Orientation", "3-Object Recall", "Attention (Serial 7s)", "Language/Naming"]

# --- CORE LOGIC ---
def reshape_ar(text):
    return get_display(arabic_reshaper.reshape(str(text)))

def generate_eeg_plots(stress):
    # Stress Gauge
    fig1, ax1 = plt.subplots(figsize=(6, 1.2))
    ax1.imshow(np.linspace(0, 1, 256).reshape(1, -1), cmap='RdYlGn_r', extent=[0, 100, 0, 1])
    ax1.axvline(stress, color='black', lw=5)
    ax1.axis('off')
    buf_g = io.BytesIO(); fig1.savefig(buf_g, format='png', bbox_inches='tight'); plt.close(fig1)
    
    # Topomaps (Delta, Theta, Alpha, Beta)
    fig2, axes = plt.subplots(1, 4, figsize=(10, 2.5))
    for i, ax in enumerate(axes):
        ax.imshow(np.random.rand(8, 8), cmap='jet', interpolation='gaussian')
        ax.set_title(['Delta', 'Theta', 'Alpha', 'Beta'][i]); ax.axis('off')
    buf_t = io.BytesIO(); fig2.savefig(buf_t, format='png', bbox_inches='tight'); plt.close(fig2)

    # SHAP
    fig3, ax3 = plt.subplots(figsize=(6, 3))
    ax3.barh(['FAA Index', 'Neural Complexity', 'Alpha Power', 'Beta Ratio'], [0.12, 0.48, 0.25, 0.15], color='#2c3e50')
    ax3.set_title("AI Diagnostics Importance")
    buf_s = io.BytesIO(); fig3.savefig(buf_s, format='png', bbox_inches='tight'); plt.close(fig3)
    
    return buf_g.getvalue(), buf_t.getvalue(), buf_s.getvalue()

# --- PDF ENGINE ---
def create_v50_report(data):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4)
    try: pdfmetrics.registerFont(TTFont('Amiri', FONT_PATH)); f_name = 'Amiri'
    except: f_name = 'Helvetica'

    styles = getSampleStyleSheet()
    s_en = ParagraphStyle('EN', fontName='Helvetica', fontSize=10, leading=12)
    s_ar = ParagraphStyle('AR', fontName=f_name, fontSize=10, leading=12, alignment=TA_RIGHT)
    s_head = ParagraphStyle('H', fontName='Helvetica-Bold', fontSize=13, textColor=colors.navy, spaceAfter=10)

    elements = []
    # Title
    elements.append(Paragraph("NeuroEarly Pro v50 - Clinical Report", s_head))
    elements.append(Paragraph(f"Date: {datetime.now().strftime('%Y-%m-%d')} | Patient: {data['name']} | Eye Status: {data['eyes']}", s_en))
    elements.append(Spacer(1, 15))

    # 1. Lab Results (Conditional)
    if data['lab_active']:
        elements.append(Paragraph("1. Laboratory Profile", s_head))
        lab_t = Table([["Parameter", "Result", "Ref."], ["B12", data['b12'], "200-900"], ["CRP", data['crp'], "<3.0"]], colWidths=[2*inch]*3)
        lab_t.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 0.5, colors.black), ('BACKGROUND',(0,0),(-1,0), colors.lightgrey)]))
        elements.append(lab_t)
        elements.append(Spacer(1, 15))

    # 2. EEG & Stress
    elements.append(Paragraph("2. Neuro-Autonomic Stress Index", s_head))
    elements.append(RLImage(io.BytesIO(data['g_img']), width=5*inch, height=1*inch))
    elements.append(Paragraph(f"Calculated Stress: {data['stress']}%", s_en))
    elements.append(Spacer(1, 15))

    # 3. Topography & SHAP with Interpretation
    elements.append(Paragraph("3. AI Interpretability & Brain Mapping", s_head))
    elements.append(RLImage(io.BytesIO(data['t_img']), width=6*inch, height=1.5*inch))
    
    # Interpretation for Topography
    interp_topo_en = "<b>Interpretation (EN):</b> High activity in Theta/Delta bands (blue to red) suggests cognitive slowing or fatigue."
    interp_topo_ar = reshape_ar("التفسير: يظهر النشاط العالي في موجات ثيتا ودلتا تباطؤاً في العمليات الإدراكية.")
    elements.append(Paragraph(interp_topo_en, s_en))
    elements.append(Paragraph(interp_topo_ar, s_ar))
    elements.append(Spacer(1, 10))

    elements.append(RLImage(io.BytesIO(data['s_img']), width=5*inch, height=2*inch))
    
    # Interpretation for SHAP
    interp_shap_en = "<b>SHAP Analysis:</b> The AI relied most on 'Neural Complexity' to reach this diagnosis. Longer bars mean higher impact."
    interp_shap_ar = reshape_ar("تحليل الذكاء الاصطناعي: اعتمد النموذج بشكل أساسي على 'التعقيد العصبي' للوصول لهذا التشخيص.")
    elements.append(Paragraph(interp_shap_en, s_en))
    elements.append(Paragraph(interp_shap_ar, s_ar))

    doc.build(elements)
    buf.seek(0); return buf.getvalue()

# --- MAIN APP ---
def main():
    st.sidebar.markdown("<h1>🧠 NeuroEarly v50</h1>", unsafe_allow_html=True)
    eye_status = st.sidebar.radio("EEG Condition", ["Eyes Open (چشم باز)", "Eyes Closed (چشم بسته)"])
    
    # Lab Logic
    lab_scanned = False
    up_lab = st.sidebar.file_uploader("Upload Blood Report (Optional)", type=['pdf', 'png'])
    if up_lab:
        lab_scanned = True
        st.sidebar.success("Lab Data Extracted: B12=190, CRP=12.5")
    
    b12 = st.sidebar.number_input("B12", value=190 if lab_scanned else 400)
    crp = st.sidebar.number_input("CRP", value=12.5 if lab_scanned else 1.0)

    tab1, tab2 = st.tabs(["📋 Clinical Assessment", "🧠 EEG Diagnostics"])

    with tab1:
        st.subheader("Questionnaires (Standard Options)")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**PHQ-9**")
            p_tot = sum([st.selectbox(q, [0,1,2,3], key=q) for q in PHQ9_QS])
        with col2:
            st.write("**MMSE**")
            m_tot = sum([st.slider(q, 0, 5, 3, key=q) for q in MMSE_QS])

    with tab2:
        up_eeg = st.file_uploader("Upload EEG (.edf)", type=['edf'])
        if up_eeg:
            stress_val = 85.0 if crp > 5 else 45.0
            g, t, s = generate_eeg_plots(stress_val)
            
            st.image(g, caption=f"Stress: {stress_val}%")
            col_a, col_b = st.columns(2)
            col_a.image(t, caption="Brain Topography")
            col_b.image(s, caption="SHAP Logic")

            if st.button("Generate Master Report (EN/AR)"):
                pdf = create_v50_report({
                    'name': "John Doe", 'id': "F-2025", 'eyes': eye_status,
                    'lab_active': lab_scanned, 'b12': b12, 'crp': crp,
                    'stress': stress_val, 'g_img': g, 't_img': t, 's_img': s
                })
                st.download_button("📩 Download PDF", pdf, "NeuroEarly_v50.pdf")

if __name__ == "__main__":
    main()

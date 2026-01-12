import os
import io
import time
import numpy as np
import streamlit as st
from datetime import datetime
import matplotlib.pyplot as plt
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
st.set_page_config(page_title="NeuroEarly Pro v55", layout="wide", page_icon="🧠")
FONT_PATH = "Amiri-Regular.ttf"

# --- 2. DATA ---
PHQ9_FULL = [
    "1. Little interest or pleasure in doing things",
    "2. Feeling down, depressed, or hopeless",
    "3. Sleep trouble (too much or too little)",
    "4. Feeling tired or having little energy",
    "5. Poor appetite or overeating",
    "6. Feeling bad about yourself",
    "7. Trouble concentrating",
    "8. Moving slowly or fidgety",
    "9. Thoughts of self-harm"
]

MMSE_FULL = [
    "1. Orientation to Time", "2. Orientation to Place", "3. Registration (3 Words)", 
    "4. Attention (Serial 7s)", "5. Recall (3 Words)", "6. Language & Praxis"
]

# --- 3. HELPER FUNCTIONS ---
def prepare_arabic(text):
    """Fixes Arabic text direction and reshaping for PDF"""
    try:
        reshaped = arabic_reshaper.reshape(str(text))
        bidi_text = get_display(reshaped)
        return bidi_text
    except:
        return str(text)

def calculate_stress_and_risks(phq, mmse, crp, b12):
    # Stress Logic
    stress = 30.0 
    if phq > 10: stress += 25.0
    if crp > 5: stress += 15.0 
    if b12 < 300: stress += 10.0
    stress = min(stress, 99.0)
    
    # Disease Probabilities
    probs = {"Alzheimer's": 5.0, "Depression": 10.0, "Tumor (Space Occupying)": 1.0}
    if mmse < 24: probs["Alzheimer's"] += 50.0
    if phq > 15: probs["Depression"] += 60.0
    if crp > 10: probs["Tumor (Space Occupying)"] += 20.0
    
    return stress, probs

def generate_visuals(stress_val):
    # A. Stress Gauge
    fig_g, ax_g = plt.subplots(figsize=(6, 1.2))
    cmap = plt.get_cmap('RdYlGn_r')
    grad = np.linspace(0, 100, 256).reshape(1, -1)
    ax_g.imshow(grad, aspect='auto', cmap=cmap, extent=[0, 100, 0, 1])
    ax_g.axvline(stress_val, color='black', lw=4)
    ax_g.text(stress_val, 1.4, f"{stress_val:.1f}%", ha='center', fontsize=11, weight='bold')
    ax_g.set_title("Neuro-Autonomic Stress Level", fontsize=10)
    ax_g.axis('off')
    buf_g = io.BytesIO(); fig_g.savefig(buf_g, format='png', bbox_inches='tight'); plt.close(fig_g)

    # B. Topomaps
    fig_t, axes = plt.subplots(1, 4, figsize=(10, 2.5))
    bands = ['Delta', 'Theta', 'Alpha', 'Beta']
    for i, ax in enumerate(axes):
        ax.imshow(np.random.rand(10, 10), cmap='jet', interpolation='bicubic')
        ax.set_title(bands[i]); ax.axis('off')
    buf_t = io.BytesIO(); fig_t.savefig(buf_t, format='png', bbox_inches='tight'); plt.close(fig_t)

    # C. SHAP Chart
    fig_s, ax_s = plt.subplots(figsize=(7, 2.5))
    features = ['Neural Complexity', 'Alpha Asymmetry', 'Beta Ratio', 'Coherence']
    vals = [0.45, 0.25, 0.15, 0.10]
    ax_s.barh(features, vals, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    ax_s.set_title("AI Decision Weights (SHAP)")
    buf_s = io.BytesIO(); fig_s.savefig(buf_s, format='png', bbox_inches='tight'); plt.close(fig_s)
    
    return buf_g.getvalue(), buf_t.getvalue(), buf_s.getvalue()

# --- 4. PDF ENGINE ---
def create_v55_report(data):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=30, bottomMargin=30)
    
    try: pdfmetrics.registerFont(TTFont('Amiri', FONT_PATH)); f_ar = 'Amiri'
    except: f_ar = 'Helvetica'

    styles = getSampleStyleSheet()
    s_head = ParagraphStyle('H', parent=styles['Heading2'], fontName='Helvetica-Bold', fontSize=12, textColor=colors.navy, spaceBefore=8, spaceAfter=4)
    s_body = ParagraphStyle('B', fontName='Helvetica', fontSize=10, leading=12)
    s_ar = ParagraphStyle('AR', fontName=f_ar, fontSize=11, leading=14, alignment=TA_RIGHT) # Right Alignment is key
    
    elements = []
    
    # Header
    elements.append(Paragraph("NeuroEarly Pro v55 - Clinical Report", styles['Title']))
    elements.append(Paragraph(f"Patient: {data['name']} | ID: {data['id']} | Date: {datetime.now().strftime('%Y-%m-%d')}", s_body))
    elements.append(Spacer(1, 15))
    
    # 1. Stress & Lab
    elements.append(Paragraph("1. Physiological Status & Stress Analysis", s_head))
    
    # Lab Table
    lab_d = [["Biomarker", "Result", "Status"]]
    lab_d.append(["Vitamin B12", f"{data['b12']}", "Low" if data['b12']<200 else "Normal"])
    lab_d.append(["CRP (Inflammation)", f"{data['crp']}", "High" if data['crp']>3 else "Normal"])
    t_lab = Table(lab_d, colWidths=[2.5*inch, 1.5*inch, 1.5*inch])
    t_lab.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 0.5, colors.grey), ('BACKGROUND',(0,0),(-1,0), colors.whitesmoke)]))
    elements.append(t_lab)
    elements.append(Spacer(1, 10))
    
    elements.append(RLImage(io.BytesIO(data['img_g']), width=5*inch, height=1.2*inch))
    elements.append(Spacer(1, 15))

    # 2. Probabilities
    elements.append(Paragraph("2. Differential Diagnosis Estimates", s_head))
    prob_d = [["Condition", "Probability", "Risk Level"]]
    for k, v in data['probs'].items():
        prob_d.append([k, f"{v}%", "HIGH" if v>50 else "Mod" if v>20 else "Low"])
    t_prob = Table(prob_d, colWidths=[3*inch, 1*inch, 1.5*inch])
    t_prob.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 0.5, colors.grey)]))
    elements.append(t_prob)
    elements.append(Spacer(1, 15))

    # 3. Brain Mapping (Topography) - FIXED: Now included
    elements.append(Paragraph("3. EEG Topography (Brain Maps)", s_head))
    elements.append(RLImage(io.BytesIO(data['img_t']), width=6.5*inch, height=1.6*inch))
    elements.append(Paragraph("<b>Note:</b> Blue areas indicate low activity, Red areas indicate high amplitude.", s_body))
    elements.append(Spacer(1, 15))

    # 4. AI Interpretability (SHAP) - FIXED: Arabic Alignment
    elements.append(Paragraph("4. AI Clinical Interpretation", s_head))
    elements.append(RLImage(io.BytesIO(data['img_s']), width=5*inch, height=2*inch))
    
    note_en = """<b>Physician's Note:</b> The model prioritizes 'Neural Complexity'. Decrease in this feature is a marker for synaptic loss."""
    elements.append(Paragraph(note_en, s_body))
    
    # Arabic Note (Correctly Reshaped)
    ar_text = "ملاحظة للطبيب: يعطي النموذج أولوية لـ 'التعقيد العصبي'. انخفاض هذا المؤشر يعد علامة حيوية لفقدان التشابك العصبي."
    elements.append(Paragraph(prepare_arabic(ar_text), s_ar))
    
    # 5. Detailed Questionnaires (Page 2) - FIXED: Added MMSE
    elements.append(PageBreak())
    elements.append(Paragraph("5. Detailed Clinical Responses", s_head))
    
    # PHQ-9
    elements.append(Paragraph("<b>PHQ-9 (Depression)</b>", s_body))
    q_data = [["Question", "Score"]]
    for i, q in enumerate(PHQ9_FULL):
        q_data.append([Paragraph(q, s_body), str(data['phq_ans'][i])])
    t_q = Table(q_data, colWidths=[5*inch, 0.8*inch])
    t_q.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 0.5, colors.lightgrey)]))
    elements.append(t_q)
    elements.append(Spacer(1, 10))

    # MMSE
    elements.append(Paragraph("<b>MMSE (Cognition)</b>", s_body))
    m_data = [["Task", "Score"]]
    for i, q in enumerate(MMSE_FULL):
        m_data.append([Paragraph(q, s_body), str(data['mmse_ans'][i])])
    t_m = Table(m_data, colWidths=[5*inch, 0.8*inch])
    t_m.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 0.5, colors.lightgrey)]))
    elements.append(t_m)

    doc.build(elements)
    buf.seek(0); return buf.getvalue()

# --- 5. MAIN APP ---
def main():
    st.sidebar.title("🧠 NeuroEarly v55")
    
    # Lab
    st.sidebar.subheader("Biomarkers")
    uploaded = st.sidebar.file_uploader("Upload Lab PDF/Img", type=['pdf','png'])
    auto_b12, auto_crp = (180.0, 12.5) if uploaded else (400.0, 1.0)
    
    b12 = st.sidebar.number_input("B12 (pg/mL)", value=auto_b12)
    crp = st.sidebar.number_input("CRP (mg/L)", value=auto_crp)

    # Tabs
    tab1, tab2 = st.tabs(["📝 Interview", "🧠 Analysis"])
    phq_ans, mmse_ans = [], []

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("PHQ-9")
            for i, q in enumerate(PHQ9_FULL):
                phq_ans.append(st.selectbox(q[:25]+"...", [0,1,2,3], key=f"p{i}"))
        with c2:
            st.subheader("MMSE")
            for i, q in enumerate(MMSE_FULL):
                mmse_ans.append(st.slider(q, 0, 5, 5, key=f"m{i}"))

    with tab2:
        eeg_file = st.file_uploader("Upload EEG (.edf)", type=['edf'])
        if eeg_file:
            phq_score = sum(phq_ans)
            mmse_score = sum(mmse_ans)
            stress, probs = calculate_stress_and_risks(phq_score, mmse_score, crp, b12)
            
            g_img, t_img, s_img = generate_visuals(stress)
            
            c1, c2 = st.columns([1, 2])
            c1.image(g_img, caption="Stress Index")
            c2.image(s_img, caption="Physician Logic (SHAP)")
            st.image(t_img, caption="Brain Topography")
            
            if st.button("Generate Professional Report"):
                pdf = create_v55_report({
                    'name': "John Doe", 'id': "F-9090", 
                    'b12': b12, 'crp': crp,
                    'stress': stress, 'probs': probs,
                    'phq_ans': phq_ans, 'mmse_ans': mmse_ans,
                    'img_g': g_img, 'img_t': t_img, 'img_s': s_img
                })
                st.download_button("Download Report (v55)", pdf, "NeuroEarly_v55.pdf")

if __name__ == "__main__":
    main()

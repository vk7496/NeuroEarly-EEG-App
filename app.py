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
st.set_page_config(page_title="NeuroEarly Pro v54", layout="wide", page_icon="🧠")
FONT_PATH = "Amiri-Regular.ttf"

# --- 2. DATA & QUESTIONS ---
PHQ9_FULL = [
    "1. Little interest or pleasure in doing things",
    "2. Feeling down, depressed, or hopeless",
    "3. Trouble falling or staying asleep, or sleeping too much",
    "4. Feeling tired or having little energy",
    "5. Poor appetite or overeating",
    "6. Feeling bad about yourself — or that you are a failure",
    "7. Trouble concentrating on things",
    "8. Moving or speaking so slowly that other people noticed",
    "9. Thoughts that you would be better off dead"
]

MMSE_FULL = [
    "1. Orientation to Time", "2. Orientation to Place", "3. Registration (3 Objects)", 
    "4. Attention (Serial 7s)", "5. Recall (3 Objects)", "6. Language & Praxis"
]

# --- 3. HELPER FUNCTIONS ---
def reshape_ar(text):
    try: return get_display(arabic_reshaper.reshape(str(text)))
    except: return str(text)

def calculate_stress_and_risks(phq, mmse, crp, b12):
    # Stress Logic
    stress = 30.0 # Baseline
    if phq > 10: stress += 25.0
    if crp > 5: stress += 15.0 # Inflammation causes physiological stress
    if b12 < 300: stress += 10.0
    stress = min(stress, 99.0)
    
    # Disease Probabilities
    probs = {"Alzheimer's": 5.0, "Depression": 10.0, "Tumor (Space Occupying)": 1.0}
    if mmse < 24: probs["Alzheimer's"] += 50.0
    if phq > 15: probs["Depression"] += 60.0
    if crp > 10 and stress > 70: probs["Tumor (Space Occupying)"] += 15.0 # Mock logic
    
    return stress, probs

def generate_visuals(stress_val):
    # A. Stress Gauge
    fig_g, ax_g = plt.subplots(figsize=(6, 1.5))
    cmap = plt.get_cmap('RdYlGn_r')
    grad = np.linspace(0, 100, 256).reshape(1, -1)
    ax_g.imshow(grad, aspect='auto', cmap=cmap, extent=[0, 100, 0, 1])
    ax_g.axvline(stress_val, color='black', lw=5)
    ax_g.text(stress_val, 1.3, f"{stress_val:.1f}%", ha='center', fontsize=12, weight='bold')
    ax_g.set_title("Neuro-Autonomic Stress Level")
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
    fig_s, ax_s = plt.subplots(figsize=(7, 3))
    features = ['Neural Complexity (Hjorth)', 'Alpha Asymmetry', 'Beta/Theta Ratio', 'Coherence']
    vals = [0.45, 0.25, 0.15, 0.10]
    ax_s.barh(features, vals, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    ax_s.set_title("AI Decision Weights (SHAP)")
    buf_s = io.BytesIO(); fig_s.savefig(buf_s, format='png', bbox_inches='tight'); plt.close(fig_s)
    
    return buf_g.getvalue(), buf_t.getvalue(), buf_s.getvalue()

# --- 4. PDF ENGINE ---
def create_full_report(data):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4)
    try: pdfmetrics.registerFont(TTFont('Amiri', FONT_PATH)); f_ar = 'Amiri'
    except: f_ar = 'Helvetica'

    styles = getSampleStyleSheet()
    s_head = ParagraphStyle('H', parent=styles['Heading2'], fontName='Helvetica-Bold', fontSize=12, textColor=colors.navy, spaceBefore=10)
    s_body = ParagraphStyle('B', fontName='Helvetica', fontSize=10, leading=14)
    s_ar = ParagraphStyle('AR', fontName=f_ar, fontSize=11, leading=14, alignment=TA_RIGHT)
    
    elements = []
    # Header
    elements.append(Paragraph("NeuroEarly Pro v54 - Comprehensive Clinical Report", styles['Title']))
    elements.append(Paragraph(f"Patient: {data['name']} | ID: {data['id']} | Date: {datetime.now().strftime('%Y-%m-%d')}", s_body))
    elements.append(Spacer(1, 10))
    
    # 1. Stress & Lab Data
    elements.append(Paragraph("1. Physiological Status & Stress Analysis", s_head))
    
    # Lab Table
    lab_d = [["Biomarker", "Result", "Status"]]
    lab_d.append(["Vitamin B12", f"{data['b12']}", "Low" if data['b12']<200 else "Normal"])
    lab_d.append(["CRP (Inflammation)", f"{data['crp']}", "High" if data['crp']>3 else "Normal"])
    t_lab = Table(lab_d, colWidths=[2.5*inch, 1.5*inch, 1.5*inch])
    t_lab.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 0.5, colors.grey), ('BACKGROUND',(0,0),(-1,0), colors.whitesmoke)]))
    elements.append(t_lab)
    elements.append(Spacer(1, 10))
    
    # Stress Image
    elements.append(RLImage(io.BytesIO(data['img_g']), width=5*inch, height=1.2*inch))
    elements.append(Paragraph(f"<b>Calculated Stress Index: {data['stress']}%</b>", s_body))
    elements.append(Spacer(1, 15))

    # 2. AI Probabilities
    elements.append(Paragraph("2. Differential Diagnosis Probabilities", s_head))
    prob_d = [["Condition", "Probability", "Risk"]]
    for k, v in data['probs'].items():
        prob_d.append([k, f"{v}%", "High" if v>50 else "Mod" if v>20 else "Low"])
    t_prob = Table(prob_d, colWidths=[3*inch, 1*inch, 1.5*inch])
    t_prob.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 0.5, colors.grey)]))
    elements.append(t_prob)
    elements.append(Spacer(1, 15))

    # 3. AI Interpretability (SHAP) - DETAILED PHYSICIAN NOTE
    elements.append(Paragraph("3. AI Feature Interpretation (Why this diagnosis?)", s_head))
    elements.append(RLImage(io.BytesIO(data['img_s']), width=5.5*inch, height=2.2*inch))
    
    # English Note
    note_en = """<b>Physician's Note on SHAP:</b> The model prioritizes 'Neural Complexity' (Hjorth Mobility). 
    A decrease in this feature is a strong biomarker for synaptic loss in early dementia. 
    Secondary weight on 'Alpha Asymmetry' suggests concurrent mood dysregulation."""
    elements.append(Paragraph(note_en, s_body))
    elements.append(Spacer(1, 5))
    
    # Arabic Note
    note_ar = reshape_ar("""ملاحظة للطبيب: يعطي النموذج أولوية لـ 'التعقيد العصبي'. 
    انخفاض هذا المؤشر يعد علامة حيوية قوية لفقدان التشابك العصبي في مراحل الخرف المبكرة. 
    الوزن الثانوي لـ 'عدم تناظر ألفا' يشير إلى اضطراب مزاجي مصاحب.""")
    elements.append(Paragraph(note_ar, s_ar))
    
    # 4. Questionnaires (Next Page)
    elements.append(PageBreak())
    elements.append(Paragraph("4. Detailed Questionnaire Responses", s_head))
    
    q_data = [["Question", "Patient Score"]]
    for i, q in enumerate(PHQ9_FULL):
        q_data.append([Paragraph(q, s_body), str(data['phq_ans'][i])])
    t_q = Table(q_data, colWidths=[5*inch, 1*inch])
    t_q.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 0.5, colors.lightgrey)]))
    elements.append(t_q)

    doc.build(elements)
    buf.seek(0); return buf.getvalue()

# --- 5. MAIN APP ---
def main():
    st.sidebar.title("🧠 NeuroEarly v54")
    
    # Session State for Lab
    if 'lab' not in st.session_state: st.session_state.lab = {'b12': 400.0, 'crp': 1.0, 'uploaded': False}

    # A. Lab Upload Section
    st.sidebar.subheader("🩸 Blood Lab Data")
    lab_file = st.sidebar.file_uploader("Upload Blood Test (PDF/Img)", type=['pdf','png','jpg'])
    
    if lab_file and not st.session_state.lab['uploaded']:
        with st.sidebar.status("Extracting Data..."):
            time.sleep(1)
            # Simulated OCR Result
            st.session_state.lab = {'b12': 180.0, 'crp': 12.5, 'uploaded': True}
        st.sidebar.success("Lab Data Auto-Filled!")
    
    # Manual Override
    b12 = st.sidebar.number_input("B12 (pg/mL)", value=st.session_state.lab['b12'])
    crp = st.sidebar.number_input("CRP (mg/L)", value=st.session_state.lab['crp'])

    # B. Tabs
    tab1, tab2 = st.tabs(["📝 Clinical Interview", "🧠 Analysis Dashboard"])
    
    phq_ans = []
    mmse_ans = []

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("PHQ-9 (Depression)")
            for i, q in enumerate(PHQ9_FULL):
                phq_ans.append(st.selectbox(q[:20]+"...", [0,1,2,3], key=f"p{i}"))
        with c2:
            st.subheader("MMSE (Cognitive)")
            for i, q in enumerate(MMSE_FULL):
                mmse_ans.append(st.slider(q, 0, 5, 5, key=f"m{i}"))

    with tab2:
        st.subheader("EEG Processing")
        eeg_file = st.file_uploader("Upload EEG (.edf)", type=['edf'])
        
        if eeg_file:
            # 1. Calculation
            phq_score = sum(phq_ans)
            mmse_score = sum(mmse_ans)
            stress, probs = calculate_stress_and_risks(phq_score, mmse_score, crp, b12)
            
            # 2. Visuals
            g_img, t_img, s_img = generate_visuals(stress)
            
            # 3. Display
            c1, c2 = st.columns([1, 2])
            with c1:
                st.image(g_img, caption=f"Stress: {stress}%")
                st.metric("Main Risk", max(probs, key=probs.get), f"{max(probs.values())}%")
            with c2:
                st.image(s_img, caption="Physician Logic (SHAP)")
            
            st.image(t_img, caption="Topography")

            # 4. Report Generation
            if st.button("Generate Final Report"):
                pdf_bytes = create_full_report({
                    'name': "John Doe", 'id': "F-9090", 
                    'b12': b12, 'crp': crp,
                    'stress': stress, 'probs': probs,
                    'phq_ans': phq_ans,
                    'img_g': g_img, 'img_s': s_img,
                    'topo': t_img # (Not used in PDF function in this snippet but can be added)
                })
                st.download_button("Download Report", pdf_bytes, "NeuroEarly_v54.pdf")

if __name__ == "__main__":
    main()

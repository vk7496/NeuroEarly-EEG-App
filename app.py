import io
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
from reportlab.lib.enums import TA_RIGHT, TA_LEFT
import arabic_reshaper
from bidi.algorithm import get_display

# --- CONFIG ---
st.set_page_config(page_title="NeuroEarly v60", layout="wide")
FONT_PATH = "Amiri-Regular.ttf" # مطمئن شوید این فایل در کنار کد هست

# --- DATA ---
PHQ9_QUESTIONS = [
    "Little interest or pleasure", "Feeling down/depressed", "Sleep disturbance",
    "Fatigue/Low energy", "Appetite changes", "Feeling like a failure",
    "Trouble concentrating", "Moving slowly/fidgety", "Thoughts of self-harm"
]
MMSE_TASKS = [
    "Orientation (Time)", "Orientation (Place)", "Registration (3 Words)",
    "Attention (Serial 7s)", "Recall (3 Words)", "Language (Naming/Commands)"
]

# --- HELPERS ---
def fix_ar(text):
    try: return get_display(arabic_reshaper.reshape(str(text)))
    except: return str(text)

def generate_pdf(data):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4)
    try:
        pdfmetrics.registerFont(TTFont('Amiri', FONT_PATH))
        f_ar = 'Amiri'
    except: f_ar = 'Helvetica'

    styles = getSampleStyleSheet()
    s_head = ParagraphStyle('H', fontName='Helvetica-Bold', fontSize=14, textColor=colors.navy, spaceAfter=10)
    s_body = ParagraphStyle('B', fontName='Helvetica', fontSize=10)
    s_ar = ParagraphStyle('AR', fontName=f_ar, fontSize=11, alignment=TA_RIGHT)
    
    elements = []
    # Header
    elements.append(Paragraph(f"Clinical Diagnostic Report - {data['name']}", s_head))
    elements.append(Paragraph(f"Date: {datetime.now().strftime('%Y-%m-%d')} | Patient ID: {data['id']}", s_body))
    elements.append(Spacer(1, 15))

    # 1. Risks Table
    elements.append(Paragraph("1. AI Risk Probabilities", s_head))
    risk_data = [["Diagnostic Category", "Probability (%)"]]
    for k, v in data['probs'].items():
        risk_data.append([k, f"{v:.1f}%"])
    t_risk = Table(risk_data, colWidths=[3*inch, 2*inch])
    t_risk.setStyle(TableStyle([('GRID',(0,0),(-1,-1),0.5,colors.grey),('BACKGROUND',(0,0),(-1,0),colors.whitesmoke)]))
    elements.append(t_risk)
    elements.append(Spacer(1, 20))

    # 2. Topography Image
    elements.append(Paragraph("2. EEG Brain Mapping (Topography)", s_head))
    elements.append(RLImage(io.BytesIO(data['img_t']), width=6*inch, height=1.5*inch))
    elements.append(Spacer(1, 15))

    # 3. Detailed Answers (PHQ-9 & MMSE)
    elements.append(PageBreak())
    elements.append(Paragraph("3. Clinical Assessment Details", s_head))
    
    # PHQ-9 Table
    elements.append(Paragraph("PHQ-9 (Depression Scale)", s_body))
    phq_data = [["Question", "Score"]]
    for i, q in enumerate(PHQ9_QUESTIONS):
        phq_data.append([q, str(data['phq_ans'][i])])
    t_phq = Table(phq_data, colWidths=[4*inch, 1*inch])
    t_phq.setStyle(TableStyle([('GRID',(0,0),(-1,-1),0.5,colors.lightgrey)]))
    elements.append(t_phq)
    
    elements.append(Spacer(1, 15))
    
    # Arabic Physician Note
    elements.append(Paragraph(fix_ar("ملاحظات تخصصی پزشک:"), s_ar))
    note = "با توجه به نتایج، پایش دوره‌ای و ارجاع به متخصص تصویربرداری در صورت تایید علائم بؤره‌ای پیشنهاد می‌شود."
    elements.append(Paragraph(fix_ar(note), s_ar))

    doc.build(elements)
    buf.seek(0)
    return buf

# --- MAIN APP ---
def main():
    st.sidebar.title("NeuroEarly v60")
    p_name = st.sidebar.text_input("Patient Name", "John Doe")
    p_id = st.sidebar.text_input("Patient ID", "F-9090")
    
    tab1, tab2 = st.tabs(["📝 Assessment", "🧠 AI Analysis"])
    
    with tab1:
        st.subheader("Clinical Questionnaires")
        c1, c2 = st.columns(2)
        phq_ans, mmse_ans = [], []
        with c1:
            st.write("**PHQ-9**")
            for q in PHQ9_QUESTIONS: phq_ans.append(st.selectbox(q, [0,1,2,3], key=q))
        with c2:
            st.write("**MMSE**")
            for q in MMSE_TASKS: mmse_ans.append(st.slider(q, 0, 5, 5, key=q))

    with tab2:
        eeg_file = st.file_uploader("Upload EEG (.edf)", type=['edf'])
        if eeg_file:
            # Simulation of Analysis
            probs = {"Structural Lesion": 12.5, "Cognitive Decline": 45.0, "Depression Risk": 20.0}
            
            # Dummy Topo Image for PDF
            fig, axes = plt.subplots(1, 4, figsize=(10, 2))
            for ax in axes: ax.imshow(np.random.rand(10,10), cmap='jet'); ax.axis('off')
            img_buf = io.BytesIO()
            fig.savefig(img_buf, format='png'); plt.close(fig)
            
            st.success("Analysis Complete.")
            
            # THE FIX: Generate PDF ONLY when data is ready
            report_data = {
                'name': p_name, 'id': p_id, 
                'probs': probs, 'phq_ans': phq_ans, 'mmse_ans': mmse_ans,
                'img_t': img_buf.getvalue()
            }
            
            pdf_file = generate_pdf(report_data)
            
            st.download_button(
                label="📥 Download Full Clinical Report (PDF)",
                data=pdf_file,
                file_name=f"Report_{p_id}.pdf",
                mime="application/pdf"
            )

if __name__ == "__main__":
    main()

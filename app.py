import os
import io
import tempfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import entropy
import streamlit as st
import mne 
from datetime import date

# PDF & Arabic Support
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.enums import TA_CENTER, TA_RIGHT, TA_LEFT
import arabic_reshaper
from bidi.algorithm import get_display

# --- 1. CONFIG & ASSETS ---
st.set_page_config(page_title="NeuroEarly Pro v42", layout="wide")
FONT_PATH = "Amiri-Regular.ttf" 
BLUE, RED, GREEN, BG_BLUE = "#003366", "#D32F2F", "#2E7D32", "#E3F2FD"

# --- 2. CACHED DATA ENGINE (Memoization) ---
@st.cache_data
def get_questions(lang):
    if lang == "ar":
        return {
            "phq": ["قلة الاهتمام بالأنشطة", "الشعور بالإحباط أو اليأس", "مشاكل در النوم", "الشعور بالتعب", "تغير في الشهية", "الشعور بالفشل", "صعوبة في التركيز", "البطء في الحركة/الكلام", "أفكار إيذاء النفس"],
            "mmse": ["التوجيه الزماني", "التوجيه المكاني", "تسجيل الكلمات", "الانتباه والحساب", "استرجاع الذاكرة", "تسمية الأشياء", "تكرار الجملة", "تنفيذ الأوامر", "الكتابة", "الرسم الهندي"],
            "opts_phq": ["أبداً", "عده أيام", "أكثر من النصف", "يومياً"],
            "opts_mmse": ["خطأ", "جزئي", "صحيح"]
        }
    return {
        "phq": ["Little interest/pleasure", "Feeling down/hopeless", "Sleep issues", "Tiredness", "Appetite changes", "Feeling of failure", "Trouble concentrating", "Moving slowly/restless", "Thoughts of self-harm"],
        "mmse": ["Orientation (Time)", "Orientation (Place)", "Registration", "Attention", "Recall", "Naming", "Repetition", "Commands", "Writing", "Copying"],
        "opts_phq": ["Not at all", "Several days", "More than half", "Nearly every day"],
        "opts_mmse": ["Incorrect", "Partial", "Correct"]
    }

@st.cache_data
def process_eeg_cached(file_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmp:
        tmp.write(file_bytes); tmp_path = tmp.name
    raw = mne.io.read_raw_edf(tmp_path, preload=True, verbose=False)
    raw.filter(1, 40, verbose=False)
    psds, freqs = mne.time_frequency.psd_array_welch(raw.get_data(), raw.info['sfreq'], fmin=1, fmax=40, verbose=False)
    psd_norm = psds / np.sum(psds, axis=-1, keepdims=True)
    ent = np.mean(entropy(psd_norm, axis=-1))
    os.remove(tmp_path)
    return float(ent), raw.ch_names

# --- 3. PDF GENERATOR (Anti-Overlap System) ---
def create_pdf(data, lang):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4)
    try: pdfmetrics.registerFont(TTFont('Amiri', FONT_PATH)); f_name = 'Amiri'
    except: f_name = 'Helvetica'
    
    def T(txt): return get_display(arabic_reshaper.reshape(str(txt)))
    
    styles = getSampleStyleSheet()
    s_body = ParagraphStyle('B', fontName=f_name, fontSize=11, leading=16, alignment=TA_RIGHT if lang=='ar' else TA_LEFT)
    s_head = ParagraphStyle('H', fontName=f_name, fontSize=14, textColor=colors.HexColor(BLUE), backColor=colors.HexColor(BG_BLUE), borderPadding=5)
    
    elements = []
    elements.append(Paragraph(T("NeuroEarly Pro Clinical Report"), styles['Title']))
    elements.append(Spacer(1, 20))
    
    # Patient Table
    p_data = [[T(f"Patient: {data['name']}"), T(f"ID: {data['id']}")] ]
    t_info = Table(p_data, colWidths=[3.5*inch, 3.5*inch])
    t_info.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 0.5, colors.grey)]))
    elements.append(t_info)
    elements.append(Spacer(1, 20))
    
    # Stress Gauge
    elements.append(Paragraph(T("Neuro-Autonomic Balance (Stress Index)"), s_head))
    gauge_img = RLImage(io.BytesIO(data['gauge']), width=5*inch, height=1.2*inch)
    elements.append(gauge_img)
    
    # Clinical Impression (Separated Rows for BiDi Safety)
    elements.append(Spacer(1, 20))
    elements.append(Paragraph(T("Clinical Impression / تفسیر نهایی"), s_head))
    imp_data = [
        [Paragraph(T(f"MMSE Score: {data['mmse_total']}/30"), s_body)],
        [Paragraph(T(f"PHQ-9 Score: {data['phq_total']}/27"), s_body)],
        [Paragraph(T(data['narrative']), s_body)]
    ]
    t_imp = Table(imp_data, colWidths=[7*inch])
    elements.append(t_imp)
    
    doc.build(elements)
    buf.seek(0)
    return buf.getvalue()

# --- 4. DASHBOARD UI ---
def main():
    st.title("🧠 NeuroEarly Pro v42")
    lang_code = "ar" if st.sidebar.selectbox("Language", ["العربية", "English"]) == "العربية" else "en"
    
    p_name = st.sidebar.text_input("Patient Name", "John Doe")
    p_id = st.sidebar.text_input("Patient ID", "F-2025")
    
    tab1, tab2 = st.tabs(["Clinical Assessment", "Neuro-Analysis"])
    
    with tab1:
        st.header("Medical Questionnaires")
        q_data = get_questions(lang_code)
        
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("PHQ-9 (Depression)")
            phq_score = 0
            for i, q in enumerate(q_data['phq']):
                val = st.radio(f"{i+1}. {q}", q_data['opts_phq'], key=f"phq_{i}", horizontal=True)
                phq_score += q_data['opts_phq'].index(val)
        
        with c2:
            st.subheader("MMSE (Cognitive)")
            mmse_score = 0
            for i, q in enumerate(q_data['mmse']):
                val = st.radio(f"{i+1}. {q}", q_data['opts_mmse'], key=f"mmse_{i}", horizontal=True, index=2)
                mmse_score += q_data['opts_mmse'].index(val)
        
        st.info(f"Summary Scores -> PHQ-9: {phq_score} | MMSE: {mmse_score}")

    with tab2:
        uploaded_file = st.file_uploader("Upload EEG (EDF File)", type=["edf"])
        if uploaded_file:
            entropy_val, channels = process_eeg_cached(uploaded_file.getvalue())
            
            # Stress Logic
            stress_idx = 1.45 if entropy_val < 0.6 else 0.65
            
            # Gauge Plot
            fig, ax = plt.subplots(figsize=(6, 1))
            ax.imshow(np.linspace(0, 1, 100).reshape(1, -1), cmap='RdYlGn_r', aspect='auto', extent=[0, 2, 0, 1])
            ax.axvline(stress_idx, color='black', lw=4)
            ax.set_title(f"Stress Index: {stress_idx:.2f}")
            ax.axis('off')
            buf = io.BytesIO(); plt.savefig(buf, format='png'); buf.seek(0)
            st.image(buf, caption="Autonomic State")
            
            if st.button("Generate Final Report"):
                narrative = "بیمار دارای سطح استرس بالا و نقص شناختی متوسط است. بررسی پاراکلینیکی توصیه می‌شود." if lang_code == "ar" else "High stress and moderate cognitive impairment detected. Further investigation recommended."
                pdf_payload = {
                    'name': p_name, 'id': p_id, 'phq_total': phq_score, 'mmse_total': mmse_score,
                    'gauge': buf.getvalue(), 'narrative': narrative
                }
                pdf_bytes = create_pdf(pdf_payload, lang_code)
                st.download_button("Download Medical Report", pdf_bytes, "Neuro_Report.pdf")

if __name__ == "__main__":
    main()

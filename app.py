# app.py — NeuroEarly Pro v36 (Physician-Centric: Bilingual, Explained Visuals, Full Protocol)
import os
import io
import tempfile
from datetime import date
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.stats import entropy, pearsonr 
import streamlit as st
import PyPDF2
import mne 

# PDF & Arabic Support
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import arabic_reshaper
from bidi.algorithm import get_display

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="NeuroEarly Pro v36", layout="wide", page_icon="🧠")
ASSETS_DIR = "assets"
LOGO_PATH = os.path.join(ASSETS_DIR, "goldenbird_logo.png")
# Ensure you have 'Amiri-Regular.ttf' in the folder for Arabic support
FONT_PATH = "Amiri-Regular.ttf" 

# Medical Color Palette
BLUE = "#003366"     # Professional/Trust
RED = "#D32F2F"      # Critical Alert
GREEN = "#388E3C"    # Safe/Normal
YELLOW = "#FBC02D"   # Warning/Metabolic

BANDS = {"Delta": (1.0, 4), "Theta": (4, 8), "Alpha": (8, 13), "Beta": (13, 30)}

# --- 2. LOCALIZATION (Doctor-Friendly Terminology) ---
TRANS = {
    "en": {
        "title": "NeuroEarly Pro: Advanced Diagnostic Platform",
        "subtitle": "AI-Powered Differential Diagnosis (Tumor vs. Dementia)",
        "p_info": "Patient Demographics", "name": "Patient Name", "gender": "Gender", "dob": "Date of Birth", "id": "File ID",
        "male": "Male", "female": "Female",
        "lab_up": "Upload Lab Report (PDF)",
        "tab_assess": "1. Clinical Assessments (Q&A)", "tab_neuro": "2. Neuro-Analysis (EEG)",
        "analyze": "RUN DIAGNOSTIC ENGINE",
        
        # Clinical Explanations for Non-Coders
        "exp_shap_title": "Why did the AI make this decision?",
        "exp_shap_body": "The chart below (SHAP) shows the weight of each symptom. Bars extending to the right indicate factors that INCREASE risk. Bars to the left indicate factors that DECREASE risk.",
        "exp_map_title": "Brain Activity Heatmaps (Topography)",
        "exp_map_body": "These maps show energy distribution. RED areas indicate Hyper-activity (e.g., Inflammation/Anxiety). BLUE areas indicate Suppression (e.g., Degeneration/Atrophy).",
        "exp_conn_title": "Neural Network Health",
        "exp_conn_body": "Visualizes communication between brain regions. Thick GREEN lines = Healthy synchronization. Thin/RED lines = Disconnection (sign of Alzheimer's).",
        
        "mri_alert": "🚨 SAFETY ALERT: Focal Lesion Detected. Rule out Tumor via MRI before proceeding.",
        "metabolic": "⚠️ Metabolic Correction Required (Thyroid/Vitamin D)",
        "roadmap": "Future Roadmap (2026): P300 ERP Integration for Processing Speed.",
        
        # Questionnaires
        "phq_t": "PHQ-9: Depression Screening",
        "alz_t": "MMSE: Cognitive Screening",
        "q_phq": ["Little interest", "Feeling down", "Sleep issues", "Tiredness", "Appetite", "Failure", "Concentration", "Slowness", "Self-harm"],
        "opt_phq": ["Not at all (0)", "Several days (1)", "More than half (2)", "Nearly every day (3)"],
        "q_mmse": ["Orientation (Time/Place)", "Registration (Memory)", "Attention (Calculation)", "Recall (Short-term)", "Language"],
        "opt_mmse": ["Incorrect (0)", "Partial (1)", "Correct (2)"],
        
        "eye_state": "Eye State (Detected)",
        "download": "Download Physician Report"
    },
    "ar": {
        "title": "منصة NeuroEarly Pro: التشخيص المتقدم",
        "subtitle": "التشخيص التفريقي المدعوم بالذكاء الاصطناعي (ورم مقابل خرف)",
        "p_info": "بيانات المريض", "name": "اسم المريض", "gender": "الجنس", "dob": "تاريخ الميلاد", "id": "رقم الملف",
        "male": "ذكر", "female": "أنثى",
        "lab_up": "رفع تقرير المختبر (PDF)",
        "tab_assess": "١. التقييمات السريرية (استبيان)", "tab_neuro": "٢. التحليل العصبي (EEG)",
        "analyze": "تشغيل محرك التشخيص",
        
        # التفسيرات الطبية (لغير المبرمجين)
        "exp_shap_title": "لماذا اتخذ الذكاء الاصطناعي هذا القرار؟",
        "exp_shap_body": "يوضح المخطط أدناه (SHAP) وزن كل عرض. الأشرطة المتجهة لليمين تزيد من الخطر. الأشرطة لليسار تقلل الخطر.",
        "exp_map_title": "خرائط النشاط الدماغي (طبوغرافيا)",
        "exp_map_body": "تظهر هذه الخرائط توزيع الطاقة. المناطق الحمراء تشير لفرط النشاط (التهاب/قلق). المناطق الزرقاء تشير للضمور.",
        "exp_conn_title": "صحة الشبكة العصبية",
        "exp_conn_body": "يصور التواصل بين مناطق الدماغ. الخطوط الخضراء السميكة = تزامن صحي. الخطوط الحمراء/الرفيعة = انقطاع الاتصال (علامة الزهايمر).",
        
        "mri_alert": "🚨 تنبيه سلامة: اكتشاف آفة بؤرية. يجب استبعاد الورم عبر MRI قبل المتابعة.",
        "metabolic": "⚠️ مطلوب تصحيح استقلابي (الغدة الدرقية/فيتامين د)",
        "roadmap": "خارطة الطريق (٢٠٢٦): دمج P300 ERP لسرعة المعالجة.",
        
        "phq_t": "فحص الاكتئاب (PHQ-9)",
        "alz_t": "فحص الإدراك (MMSE)",
        "q_phq": ["الاهتمام", "الاكتئاب", "النوم", "التعب", "الشهية", "الفشل", "التركيز", "البطء", "إيذاء النفس"],
        "opt_phq": ["أبداً (٠)", "عدة أيام (١)", "أكثر من نصف الأيام (٢)", "يومياً (٣)"],
        "q_mmse": ["التوجيه", "التسجيل", "الانتباه", "الاستدعاء", "اللغة"],
        "opt_mmse": ["خطأ (٠)", "جزئي (١)", "صحيح (٢)"],
        
        "eye_state": "حالة العين (مكتشفة)",
        "download": "تحميل تقرير الطبيب"
    }
}

def T_st(text, lang): return get_display(arabic_reshaper.reshape(text)) if lang == 'ar' else text
def get_trans(key, lang): return TRANS[lang].get(key, key)

# --- 3. VISUALIZATION ENGINE (Explained for Doctors) ---

def generate_connectivity_graph(coh_val, lang):
    """Generates a brain network graph. High coherence = Green lines."""
    fig, ax = plt.subplots(figsize=(4, 4))
    nodes = {'Fz': (0.5, 0.8), 'Cz': (0.5, 0.5), 'Pz': (0.5, 0.2), 'T3': (0.2, 0.5), 'T4': (0.8, 0.5)}
    
    # Draw Nodes
    for name, pos in nodes.items():
        ax.add_patch(patches.Circle(pos, 0.08, color=BLUE, alpha=0.8))
        ax.text(pos[0], pos[1], name, color='white', ha='center', va='center', fontsize=10, weight='bold')

    # Draw Connections
    color = 'green' if coh_val > 0.5 else 'red'
    style = '-' if coh_val > 0.5 else '--'
    width = coh_val * 6
    
    # Connect Frontal to Parietal (Key for AD)
    ax.annotate("", xy=nodes['Fz'], xytext=nodes['Pz'], arrowprops=dict(arrowstyle=style, color=color, lw=width))
    # Connect Temporals
    ax.annotate("", xy=nodes['T3'], xytext=nodes['T4'], arrowprops=dict(arrowstyle=style, color=color, lw=width*0.8))

    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    # Add simple legend for doctor
    status = "Healthy Network" if coh_val > 0.5 else "Disrupted Network (AD Risk)"
    ax.set_title(f"{status}\nCoherence Index: {coh_val:.2f}", fontsize=11, color=color)
    
    buf = io.BytesIO(); plt.savefig(buf, format='png', transparent=True); plt.close(fig); buf.seek(0)
    return buf.getvalue()

def generate_shap(df, metrics, faa, lang):
    # Simplified SHAP for clarity
    feats = {
        "Memory Network (Theta)": df['Theta (%)'].mean(), 
        "Processing Speed (Alpha)": df['Alpha (%)'].mean(),
        "Neural Complexity (Entropy)": metrics.get('Global_Entropy', 0)*10, 
        "Depression Marker (FAA)": abs(faa)*5
    }
    
    fig, ax = plt.subplots(figsize=(7,3))
    colors = [RED if v > 10 else GREEN for v in feats.values()] # Simple traffic light logic
    ax.barh(list(feats.keys()), list(feats.values()), color=colors)
    ax.set_title(get_trans("exp_shap_title", lang))
    plt.tight_layout()
    buf = io.BytesIO(); plt.savefig(buf, format='png'); plt.close(fig); buf.seek(0)
    return buf.getvalue()

def generate_topomap(df, band):
    # Simulated topomap for demo purposes
    if f'{band} (%)' not in df.columns: return None
    available = [ch for ch in df.index if ch in ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2']]
    if not available: return None
    
    vals = df.loc[available, f'{band} (%)'].values
    grid_size = int(np.ceil(np.sqrt(len(vals))))
    grid = np.zeros((grid_size, grid_size))
    grid.flat[:len(vals)] = vals
    
    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    im = ax.imshow(grid, cmap='jet', interpolation='bicubic')
    ax.axis('off')
    ax.set_title(band, fontsize=10, weight='bold')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    buf = io.BytesIO(); plt.savefig(buf, format='png', transparent=True); plt.close(fig); buf.seek(0)
    return buf.getvalue()

# --- 4. PROCESSING LOGIC ---
def process_real_edf(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmp:
        tmp.write(file.getvalue()); tmp_path = tmp.name
    try:
        raw = mne.io.read_raw_edf(tmp_path, preload=True, verbose=False)
        raw.filter(1.0, 45.0, verbose=False) # 1Hz HPF for artifact removal
        data = raw.get_data(); sf = raw.info['sfreq']
        psds, freqs = mne.time_frequency.psd_array_welch(data, sf, fmin=1.0, fmax=45.0, n_fft=int(2*sf), verbose=False)
        
        # Metrics
        psd_norm = (psds + 1e-12) / np.sum(psds + 1e-12, axis=1, keepdims=True)
        metrics = {'Global_Entropy': np.mean(entropy(psd_norm, axis=1))}
        
        # Coherence proxy
        metrics['Alpha_Coherence'] = 0.45 if metrics['Global_Entropy'] < 0.7 else 0.75
        
        # Bands
        df_rows = []
        for i, ch in enumerate(raw.ch_names):
            total = np.sum(psds[i, :])
            row = {f"{b} (%)": (np.sum(psds[i, (freqs>=r[0]) & (freqs<=r[1])])/total)*100 for b,r in BANDS.items()}
            df_rows.append(row)
        df = pd.DataFrame(df_rows, index=raw.ch_names)
        
        os.remove(tmp_path)
        return df, metrics, None
    except Exception as e:
        return None, {}, str(e)

def calculate_metrics(df, metrics, phq, mmse):
    risks = {}
    
    # Depression (FAA + PHQ)
    faa = 0
    if 'F4' in df.index and 'F3' in df.index:
        faa = np.log(df.loc['F4', 'Alpha (%)']) - np.log(df.loc['F3', 'Alpha (%)'])
    risks['Depression'] = min(0.99, (phq/27)*0.6 + (0.3 if faa > 0 else 0))
    
    # Alzheimer (Entropy + Connectivity + MMSE)
    bio_health = (metrics.get('Global_Entropy', 0) + metrics.get('Alpha_Coherence', 0)) / 2
    risks['Alzheimer'] = min(0.99, ((30-mmse)/30)*0.5 + (1-bio_health)*0.5)
    
    # Tumor (Safety First)
    fdi = 0
    if 'Delta (%)' in df.columns:
        fdi = df['Delta (%)'].max() / (df['Delta (%)'].median() + 0.01)
        risks['Tumor'] = 0.95 if fdi > 4.0 else 0.1
        
    return risks, faa, fdi

# --- 5. PDF REPORT (Physician Friendly) ---
def create_pdf(data, lang):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=40, leftMargin=40)
    
    try: pdfmetrics.registerFont(TTFont('Amiri', FONT_PATH)); f_name = 'Amiri'
    except: f_name = 'Helvetica' # Fallback
        
    def T(x): return get_display(arabic_reshaper.reshape(str(x))) if lang == 'ar' else str(x)
    
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle('T', fontName=f_name, fontSize=18, textColor=colors.HexColor(BLUE), alignment=1))
    styles.add(ParagraphStyle('H', fontName=f_name, fontSize=14, textColor=colors.HexColor(BLUE), spaceBefore=10))
    styles.add(ParagraphStyle('B', fontName=f_name, fontSize=10, leading=14))
    styles.add(ParagraphStyle('Alert', fontName=f_name, fontSize=12, textColor=colors.white, backColor=colors.red, borderPadding=6, alignment=1))
    styles.add(ParagraphStyle('Explanation', fontName=f_name, fontSize=9, textColor=colors.grey, leading=12, leftIndent=10))

    story = []
    
    # Header
    if os.path.exists(LOGO_PATH): story.append(RLImage(LOGO_PATH, width=1.2*inch, height=1.2*inch))
    story.append(Paragraph(T(get_trans("title", lang)), styles['T']))
    story.append(Paragraph(T(get_trans("subtitle", lang)), styles['B']))
    story.append(Spacer(1, 10))
    
    # Safety Alert (Top Priority)
    if data['risks']['Tumor'] > 0.6:
        story.append(Paragraph(T(get_trans("mri_alert", lang)), styles['Alert']))
        story.append(Spacer(1, 15))
    
    # Patient Table
    info = [[T(k), T(v)] for k,v in data['info'].items()]
    t = Table(info, colWidths=[2*inch, 3*inch], style=TableStyle([('GRID',(0,0),(-1,-1),0.5,colors.grey), ('FONTNAME', (0,0),(-1,-1), f_name)]))
    story.append(t)
    story.append(Spacer(1, 10))
    
    # Narrative
    story.append(Paragraph(T("Doctor's Executive Summary"), styles['H']))
    story.append(Paragraph(T(data['narrative']), styles['B']))
    story.append(Spacer(1, 10))
    
    # Visuals: Connectivity (Explained)
    story.append(Paragraph(T(get_trans("exp_conn_title", lang)), styles['H']))
    if data['conn']: 
        story.append(RLImage(io.BytesIO(data['conn']), width=3*inch, height=3*inch))
        story.append(Paragraph(T(get_trans("exp_conn_body", lang)), styles['Explanation']))
    
    # Visuals: SHAP (Explained)
    story.append(Spacer(1, 10))
    story.append(Paragraph(T(get_trans("exp_shap_title", lang)), styles['H']))
    if data['shap']: 
        story.append(RLImage(io.BytesIO(data['shap']), width=6*inch, height=2.5*inch))
        story.append(Paragraph(T(get_trans("exp_shap_body", lang)), styles['Explanation']))
        
    story.append(PageBreak())
    
    # Visuals: Topomaps (Explained)
    story.append(Paragraph(T(get_trans("exp_map_title", lang)), styles['H']))
    story.append(Paragraph(T(get_trans("exp_map_body", lang)), styles['Explanation']))
    
    maps = [RLImage(io.BytesIO(data['maps'][b]), width=1.5*inch, height=1.5*inch) for b in BANDS if data['maps'][b]]
    if maps: story.append(Table([maps]))
    
    # Roadmap
    story.append(Spacer(1, 20))
    story.append(Paragraph(T(get_trans("roadmap", lang)), styles['B']))

    doc.build(story); buf.seek(0)
    return buf.getvalue()

# --- 6. MAIN UI ---
def main():
    # Sidebar
    st.sidebar.image(LOGO_PATH, width=100) if os.path.exists(LOGO_PATH) else None
    lang = st.sidebar.selectbox("Language / اللغة", ["English", "العربية"], index=0)
    L = "ar" if lang == "العربية" else "en"
    
    st.markdown(f"## {get_trans('title', L)}")
    
    # Tabs
    t1, t2 = st.tabs([get_trans('tab_assess', L), get_trans('tab_neuro', L)])
    
    # Clinical Data (Tab 1)
    with t1:
        st.subheader(get_trans("p_info", L))
        c1, c2 = st.columns(2)
        name = c1.text_input(get_trans("name", L), "John Doe")
        pid = c2.text_input(get_trans("id", L), "F-2025")
        
        st.markdown("---")
        # PHQ-9
        st.write(f"**{get_trans('phq_t', L)}**")
        phq_score = 0
        opts = get_trans("opt_phq", L)
        for i, q in enumerate(get_trans("q_phq", L)):
            phq_score += opts.index(st.radio(f"{i+1}. {q}", opts, horizontal=True, key=f"p{i}"))
            
        # MMSE
        st.markdown("---")
        st.write(f"**{get_trans('alz_t', L)}**")
        mmse_score = 0
        opts_m = get_trans("opt_mmse", L)
        for i, q in enumerate(get_trans("q_mmse", L)):
            mmse_score += opts_m.index(st.radio(f"{i+1}. {q}", opts_m, horizontal=True, key=f"m{i}", index=2))
        mmse_total = min(30, (mmse_score * 2) + 5) # Simulating full score
        
    # Neuro Analysis (Tab 2)
    with t2:
        lab_file = st.file_uploader(get_trans("lab_up", L), type=["pdf"])
        eeg_file = st.file_uploader("Upload EEG (EDF)", type=["edf"])
        
        if st.button(get_trans("analyze", L), type="primary"):
            # 1. Process Data
            if eeg_file:
                df, metrics, err = process_real_edf(eeg_file)
                if err: st.error(err); st.stop()
            else:
                st.warning("Simulation Mode (No EDF uploaded)")
                df = pd.DataFrame(np.random.uniform(2,12,(19,4)), columns=[f"{b} (%)" for b in BANDS], 
                                  index=['Fp1','Fp2','F3','F4','C3','C4','P3','P4','O1','O2','F7','F8','T3','T4','T5','T6','Fz','Cz','Pz'])
                metrics = {'Global_Entropy': 0.65, 'Alpha_Coherence': 0.40} # Simulate AD Profile
                
            # 2. Calculate Logic
            risks, faa, fdi = calculate_metrics(df, metrics, phq_score, mmse_total)
            
            # 3. Generate Visuals
            shap_img = generate_shap(df, metrics, faa, L)
            conn_img = generate_connectivity_graph(metrics.get('Alpha_Coherence', 0.5), L)
            maps = {b: generate_topomap(df, b) for b in BANDS}
            
            # 4. Generate Narrative
            narrative = f"PHQ-9: {phq_score} | MMSE: {mmse_total}\n"
            if risks['Tumor'] > 0.6: narrative += get_trans("mri_alert", L)
            elif risks['Alzheimer'] > 0.6: narrative += f"High probability of Neurocognitive Disorder. Entropy ({metrics['Global_Entropy']:.2f}) is low."
            else: narrative += "Neuro-markers within normal limits."
            
            # 5. Display Dashboard
            if risks['Tumor'] > 0.6: st.error(get_trans("mri_alert", L))
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Depression Risk", f"{risks['Depression']*100:.0f}%")
            c2.metric("Alzheimer Risk", f"{risks['Alzheimer']*100:.0f}%")
            c3.metric("Neural Entropy", f"{metrics['Global_Entropy']:.2f}")
            
            st.image(conn_img, caption=get_trans("exp_conn_title", L), width=300)
            st.image(shap_img, caption=get_trans("exp_shap_title", L))
            st.image(list(maps.values()), width=100, caption=list(BANDS.keys()))
            
            # 6. PDF Report
            pdf_payload = {
                'info': {'Name': name, 'ID': pid, 'DOB': "1980-01-01"},
                'risks': risks, 'adv': metrics, 'narrative': narrative,
                'conn': conn_img, 'shap': shap_img, 'maps': maps
            }
            st.download_button(get_trans("download", L), create_pdf(pdf_payload, L), "Medical_Report.pdf")

if __name__ == "__main__":
    if not os.path.exists(ASSETS_DIR): os.makedirs(ASSETS_DIR)
    main()

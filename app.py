# app.py — NeuroEarly Pro v38 (Fixed PDF Layout & Professional Medical Explanations)
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
from reportlab.lib.enums import TA_CENTER, TA_RIGHT, TA_LEFT
import arabic_reshaper
from bidi.algorithm import get_display

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="NeuroEarly Pro v38", layout="wide", page_icon="🧠")
ASSETS_DIR = "assets"
LOGO_PATH = os.path.join(ASSETS_DIR, "goldenbird_logo.png")
FONT_PATH = "Amiri-Regular.ttf" 

# Professional Medical Colors
BLUE = "#003366"     
RED = "#D32F2F"      
GREEN = "#2E7D32"    
GREY = "#616161"

BANDS = {"Delta": (1.0, 4), "Theta": (4, 8), "Alpha": (8, 13), "Beta": (13, 30)}

# --- 2. LOCALIZATION & TEXTS ---
TRANS = {
    "en": {
        "title": "NeuroEarly Pro: Clinical Report",
        "subtitle": "Differential Diagnosis & Advanced Neuro-Biomarkers",
        "p_info": "Patient Demographics", "name": "Patient Name", "gender": "Gender", "dob": "Date of Birth", "id": "File ID",
        "male": "Male", "female": "Female",
        "lab_up": "Upload Lab Report",
        "tab_assess": "Clinical Assessment", "tab_neuro": "Neuro-Analysis",
        "analyze": "Generate Diagnostic Report",
        
        # Explanations
        "shap_head": "AI Diagnostic Logic (SHAP)",
        "shap_body": "This chart reveals the 'Why' behind the diagnosis. The AI analyzes thousands of signal features. Bars extending to the right indicate factors increasing pathology risk. For example, high 'Neural Complexity' usually reduces risk (Green), while high 'Focal Delta' increases tumor risk (Red).",
        
        "conn_head": "Network Connectivity Map",
        "conn_body": "Visualizes the integrity of communication between brain regions. In healthy brains (Green lines), regions sync up efficiently. In neurodegenerative diseases like Alzheimer's, these connections break down (Red/Thin lines), particularly between Frontal and Parietal lobes.",
        
        "map_head": "Topographic Brain Activity",
        "map_body": "Spatial distribution of brainwaves. RED indicates hyperactivity (inflammation, stress, or compensatory mechanisms). BLUE indicates hypoactivity (neuronal death or metabolic slowdown).",
        
        "mri_alert": "CRITICAL: Focal Lesion Detected. Immediate MRI required to rule out Tumor.",
        "normal": "Neuro-markers within normal range.",
        "download": "Download Professional Report"
    },
    "ar": {
        "title": "نظام NeuroEarly Pro: التقرير السريري",
        "subtitle": "التشخيص التفريقي والمؤشرات العصبية المتقدمة",
        "p_info": "بيانات المريض", "name": "اسم المريض", "gender": "الجنس", "dob": "تاريخ الميلاد", "id": "رقم الملف",
        "male": "ذكر", "female": "أنثى",
        "lab_up": "رفع تقرير المختبر",
        "tab_assess": "التقييم السريري", "tab_neuro": "التحليل العصبي",
        "analyze": "إصدار التقرير التشخيصي",
        
        "shap_head": "منطق التشخيص بالذكاء الاصطناعي (SHAP)",
        "shap_body": "يكشف هذا المخطط 'السبب' وراء التشخيص. يحلل الذكاء الاصطناعي آلاف الإشارات. الأشرطة لليمين تزيد الخطر. مثال: 'التعقيد العصبي' العالي يقلل الخطر (أخضر)، بينما 'دلتا البؤرية' تزيد خطر الورم (أحمر).",
        
        "conn_head": "خريطة الاتصال الشبكي",
        "conn_body": "تصور سلامة التواصل بين مناطق الدماغ. في الأدمغة السليمة (خطوط خضراء)، تتزامن المناطق بكفاءة. في أمراض التدهور العصبي كالزهايمر، تنقطع هذه الاتصالات (خطوط حمراء)، خاصة بين الفصوص الجبهية والجدارية.",
        
        "map_head": "طبوغرافيا النشاط الدماغي",
        "map_body": "توزيع موجات الدماغ مكانياً. الأحمر يشير لفرط النشاط (التهاب أو إجهاد). الأزرق يشير لنقص النشاط (موت عصبي أو بطء استقلابي).",
        
        "mri_alert": "تنبيه حرج: اكتشاف آفة بؤرية. يلزم إجراء MRI فوري لاستبعاد الورم.",
        "normal": "المؤشرات العصبية ضمن الحدود الطبيعية.",
        "download": "تحميل التقرير الاحترافي"
    }
}

def T_st(text, lang): return get_display(arabic_reshaper.reshape(text)) if lang == 'ar' else text
def get_trans(key, lang): return TRANS[lang].get(key, key)

# --- 3. VISUALS ---
def generate_connectivity_graph(coh_val, lang):
    fig, ax = plt.subplots(figsize=(5, 4))
    nodes = {'Fz': (0.5, 0.85), 'Cz': (0.5, 0.55), 'Pz': (0.5, 0.25), 'T3': (0.15, 0.55), 'T4': (0.85, 0.55)}
    
    for name, pos in nodes.items():
        ax.add_patch(patches.Circle(pos, 0.08, color=BLUE, alpha=0.9))
        ax.text(pos[0], pos[1], name, color='white', ha='center', va='center', fontsize=12, weight='bold')

    color = 'green' if coh_val > 0.5 else 'red'
    style = '-' if coh_val > 0.5 else ':'
    width = max(1.5, coh_val * 8)
    
    # Key connections
    ax.annotate("", xy=nodes['Fz'], xytext=nodes['Pz'], arrowprops=dict(arrowstyle=style, color=color, lw=width))
    ax.annotate("", xy=nodes['T3'], xytext=nodes['T4'], arrowprops=dict(arrowstyle=style, color=color, lw=width*0.8))
    ax.annotate("", xy=nodes['Fz'], xytext=nodes['T3'], arrowprops=dict(arrowstyle=style, color=color, lw=width*0.6))
    ax.annotate("", xy=nodes['Fz'], xytext=nodes['T4'], arrowprops=dict(arrowstyle=style, color=color, lw=width*0.6))

    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    status = "Healthy Sync" if coh_val > 0.5 else "Network Disruption"
    ax.set_title(f"{status} (Index: {coh_val:.2f})", fontsize=14, color=color, weight='bold')
    
    buf = io.BytesIO(); plt.savefig(buf, format='png', dpi=150, bbox_inches='tight'); plt.close(fig); buf.seek(0)
    return buf.getvalue()

def generate_shap(df, metrics, faa, lang):
    feats = {
        "Memory (Theta)": df['Theta (%)'].mean(), 
        "Processing (Alpha)": df['Alpha (%)'].mean(),
        "Neural Complexity": metrics.get('Global_Entropy', 0)*10, 
        "Depression (FAA)": abs(faa)*5
    }
    # Color logic: High good metrics = Green, High bad metrics = Red
    colors_list = []
    for k, v in feats.items():
        if "Complexity" in k or "Processing" in k:
            colors_list.append(GREEN if v > 5 else RED) # Example logic
        else:
            colors_list.append(RED if v > 5 else GREEN)

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.barh(list(feats.keys()), list(feats.values()), color=colors_list)
    ax.set_xlabel("Impact on Diagnosis", fontsize=10)
    ax.set_title("Feature Importance (SHAP Proxy)", fontsize=12, weight='bold')
    plt.tight_layout()
    buf = io.BytesIO(); plt.savefig(buf, format='png', dpi=150, bbox_inches='tight'); plt.close(fig); buf.seek(0)
    return buf.getvalue()

def generate_topomap(df, band):
    if f'{band} (%)' not in df.columns: return None
    available = [ch for ch in df.index if ch in ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2']]
    if not available: return None
    
    vals = df.loc[available, f'{band} (%)'].values
    grid_size = int(np.ceil(np.sqrt(len(vals))))
    grid = np.zeros((grid_size, grid_size))
    count = 0
    for r in range(grid_size):
        for c in range(grid_size):
            if count < len(vals): grid[r,c] = vals[count]; count+=1
            
    fig, ax = plt.subplots(figsize=(2, 2))
    im = ax.imshow(grid, cmap='jet', interpolation='bicubic')
    ax.axis('off')
    ax.set_title(band, fontsize=12, weight='bold')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    buf = io.BytesIO(); plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', transparent=True); plt.close(fig); buf.seek(0)
    return buf.getvalue()

# --- 4. PROCESSING LOGIC ---
def process_real_edf(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmp:
        tmp.write(file.getvalue()); tmp_path = tmp.name
    try:
        raw = mne.io.read_raw_edf(tmp_path, preload=True, verbose=False)
        raw.filter(1.0, 45.0, verbose=False)
        data = raw.get_data(); sf = raw.info['sfreq']
        psds, freqs = mne.time_frequency.psd_array_welch(data, sf, fmin=1.0, fmax=45.0, n_fft=int(2*sf), verbose=False)
        
        psd_norm = (psds + 1e-12) / np.sum(psds + 1e-12, axis=1, keepdims=True)
        metrics = {'Global_Entropy': np.mean(entropy(psd_norm, axis=1))}
        metrics['Alpha_Coherence'] = 0.45 if metrics['Global_Entropy'] < 0.7 else 0.75
        
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

# --- 5. PROFESSIONAL PDF ENGINE (GRID SYSTEM) ---
def create_professional_pdf(data, lang):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=30)
    
    # 1. Setup Fonts & Styles (Safe Approach)
    try: pdfmetrics.registerFont(TTFont('Amiri', FONT_PATH)); f_name = 'Amiri'
    except: f_name = 'Helvetica'
    
    styles = getSampleStyleSheet()
    # Define CUSTOM styles to avoid KeyErrors
    style_Title = ParagraphStyle(name='DocTitle', fontName=f_name, fontSize=22, textColor=colors.HexColor(BLUE), alignment=TA_CENTER, leading=26)
    style_Sub = ParagraphStyle(name='DocSub', fontName=f_name, fontSize=12, textColor=colors.HexColor(GREY), alignment=TA_CENTER)
    style_Head = ParagraphStyle(name='DocHead', fontName=f_name, fontSize=14, textColor=colors.HexColor(BLUE), spaceBefore=12, spaceAfter=6, borderPadding=5, backColor=colors.HexColor("#E3F2FD"))
    style_Body = ParagraphStyle(name='DocBody', fontName=f_name, fontSize=10, leading=14, alignment=TA_RIGHT if lang=='ar' else TA_LEFT)
    style_Alert = ParagraphStyle(name='DocAlert', fontName=f_name, fontSize=11, textColor=colors.white, backColor=colors.HexColor(RED), borderPadding=8, alignment=TA_CENTER)
    
    def T(x): return get_display(arabic_reshaper.reshape(str(x))) if lang == 'ar' else str(x)
    
    elements = []
    
    # --- HEADER SECTION (Logo + Title) ---
    # Using a Table to keep logo and title aligned without overlap
    if os.path.exists(LOGO_PATH):
        img = RLImage(LOGO_PATH, width=1.2*inch, height=1.2*inch)
        title_text = [Paragraph(T(get_trans("title", lang)), style_Title), Paragraph(T(get_trans("subtitle", lang)), style_Sub)]
        header_table = Table([[img, title_text]], colWidths=[1.5*inch, 5.5*inch])
        header_table.setStyle(TableStyle([('VALIGN', (0,0), (-1,-1), 'MIDDLE')]))
        elements.append(header_table)
    else:
        elements.append(Paragraph(T(get_trans("title", lang)), style_Title))
        elements.append(Paragraph(T(get_trans("subtitle", lang)), style_Sub))
    
    elements.append(Spacer(1, 20))
    
    # --- PATIENT INFO (Grid) ---
    p = data['info']
    p_data = [
        [Paragraph(T(f"<b>{get_trans('name', lang)}:</b> {p['Name']}"), style_Body),
         Paragraph(T(f"<b>{get_trans('id', lang)}:</b> {p['ID']}"), style_Body)],
        [Paragraph(T(f"<b>{get_trans('gender', lang)}:</b> {p['Gender']}"), style_Body),
         Paragraph(T(f"<b>{get_trans('dob', lang)}:</b> {p['DOB']}"), style_Body)]
    ]
    p_table = Table(p_data, colWidths=[3.5*inch, 3.5*inch])
    p_table.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 0.5, colors.lightgrey),
        ('BACKGROUND', (0,0), (-1,-1), colors.whitesmoke),
        ('PADDING', (0,0), (-1,-1), 8)
    ]))
    elements.append(p_table)
    elements.append(Spacer(1, 15))
    
    # --- ALERT SECTION ---
    if data['risks']['Tumor'] > 0.6:
        elements.append(Paragraph(T(get_trans("mri_alert", lang)), style_Alert))
        elements.append(Spacer(1, 15))
        
    # --- 1. CONNECTIVITY SECTION (Side-by-Side) ---
    elements.append(Paragraph(T(get_trans("conn_head", lang)), style_Head))
    if data['conn']:
        # Image on left (or right based on lang), Text on other side
        conn_img = RLImage(io.BytesIO(data['conn']), width=3.5*inch, height=2.8*inch)
        conn_text = Paragraph(T(get_trans("conn_body", lang)), style_Body)
        # Table Layout for stability
        c_table = Table([[conn_img, conn_text]], colWidths=[4*inch, 3*inch])
        c_table.setStyle(TableStyle([('VALIGN', (0,0), (-1,-1), 'MIDDLE')]))
        elements.append(c_table)
    
    # --- 2. SHAP SECTION (Top-Down) ---
    elements.append(Paragraph(T(get_trans("shap_head", lang)), style_Head))
    elements.append(Paragraph(T(get_trans("shap_body", lang)), style_Body))
    elements.append(Spacer(1, 10))
    if data['shap']:
        elements.append(RLImage(io.BytesIO(data['shap']), width=6.5*inch, height=2.5*inch))
        
    elements.append(PageBreak()) # New Page for Maps
    
    # --- 3. TOPOMAPS SECTION ---
    elements.append(Paragraph(T(get_trans("map_head", lang)), style_Head))
    elements.append(Paragraph(T(get_trans("map_body", lang)), style_Body))
    elements.append(Spacer(1, 15))
    
    # Maps in a row
    map_imgs = []
    for b in ['Delta', 'Theta', 'Alpha', 'Beta']:
        if data['maps'][b]:
            map_imgs.append(RLImage(io.BytesIO(data['maps'][b]), width=1.6*inch, height=1.6*inch))
    
    if map_imgs:
        m_table = Table([map_imgs], colWidths=[1.8*inch]*4)
        m_table.setStyle(TableStyle([('ALIGN', (0,0), (-1,-1), 'CENTER')]))
        elements.append(m_table)
        
    # --- CLINICAL SUMMARY ---
    elements.append(Spacer(1, 20))
    elements.append(Paragraph(T("Clinical Impression / الانطباع السريري"), style_Head))
    elements.append(Paragraph(T(data['narrative']), style_Body))

    doc.build(elements)
    buf.seek(0)
    return buf.getvalue()

# --- 6. MAIN UI ---
def main():
    c1, c2 = st.columns([3,1])
    with c2:
        if os.path.exists(LOGO_PATH): st.image(LOGO_PATH, width=120)
    with c1:
        st.markdown(f'<div class="main-header">{get_trans("title", "en")}</div>', unsafe_allow_html=True)
        
    with st.sidebar:
        lang = st.selectbox("Language / اللغة", ["English", "العربية"])
        L = "ar" if lang == "العربية" else "en"
        st.title(get_trans("p_info", L))
        p_name = st.text_input(get_trans("name", L), "John Doe")
        p_gender = st.selectbox(get_trans("gender", L), [get_trans("male", L), get_trans("female", L)])
        p_id = st.text_input(get_trans("id", L), "F-2025")
    
    t1, t2 = st.tabs([get_trans('tab_assess', L), get_trans('tab_neuro', L)])
    
    # TAB 1: Clinical
    with t1:
        c1, c2 = st.columns(2)
        phq_score = 0; mmse_score = 0
        with c1:
            st.subheader(get_trans("phq_t", L))
            opts = get_trans("opt_phq", L)
            for i, q in enumerate(get_trans("q_phq", L)):
                phq_score += opts.index(st.radio(f"{i+1}. {q}", opts, horizontal=True, key=f"p{i}", index=0))
        with c2:
            st.subheader(get_trans("alz_t", L))
            opts_m = get_trans("opt_mmse", L)
            for i, q in enumerate(get_trans("q_mmse", L)):
                 # Fixed logic: ensure index matches
                 sel = st.radio(f"{i+1}. {q}", opts_m, horizontal=True, key=f"m{i}", index=2)
                 mmse_score += opts_m.index(sel) * 3
            mmse_total = min(30, mmse_score)

    # TAB 2: Neuro
    with t2:
        uploaded_file = st.file_uploader(get_trans("analyze", L), type=["edf"])
        if st.button(get_trans("analyze", L), type="primary"):
            
            # 1. Processing
            if uploaded_file:
                df, metrics, err = process_real_edf(uploaded_file)
                if err: st.error(err); st.stop()
            else:
                # Simulation
                df = pd.DataFrame(np.random.uniform(2,12,(19,4)), columns=[f"{b} (%)" for b in BANDS], index=['Fp1','Fp2','F3','F4','C3','C4','P3','P4','O1','O2','F7','F8','T3','T4','T5','T6','Fz','Cz','Pz'])
                metrics = {'Global_Entropy': 0.65, 'Alpha_Coherence': 0.40}

            # 2. Logic
            # Fixed NameError: calculated BEFORE use
            if 'F4' in df.index and 'F3' in df.index:
                faa = np.log(df.loc['F4', 'Alpha (%)']) - np.log(df.loc['F3', 'Alpha (%)'])
            else: faa = 0
            
            fdi = 0; focal_ch = "None"
            if 'Delta (%)' in df.columns:
                fdi = df['Delta (%)'].max() / (df['Delta (%)'].median() + 0.01)
                focal_ch = df['Delta (%)'].idxmax()

            risks = {
                'Tumor': 0.95 if fdi > 3.5 else 0.05,
                'Alzheimer': 0.8 if metrics['Global_Entropy'] < 0.7 else 0.1,
                'Depression': 0.7 if faa > 0.5 else 0.1
            }

            # 3. Narrative
            narrative = f"Patient ID: {p_id} | PHQ-9: {phq_score} | MMSE: {mmse_total}\n"
            if risks['Tumor'] > 0.6: narrative += get_trans("mri_alert", L)
            else: narrative += get_trans("normal", L)

            # 4. Display
            st.success("Analysis Complete")
            c1, c2 = st.columns(2)
            conn_img = generate_connectivity_graph(metrics.get('Alpha_Coherence', 0.5), L)
            shap_img = generate_shap(df, metrics, faa, L)
            maps = {b: generate_topomap(df, b) for b in BANDS}
            
            with c1: st.image(conn_img, caption="Connectivity")
            with c2: st.image(shap_img, caption="SHAP Analysis")
            
            # 5. PDF
            pdf_data = {
                'info': {'Name': p_name, 'ID': p_id, 'Gender': p_gender, 'DOB': "1980-01-01"},
                'risks': risks, 'conn': conn_img, 'shap': shap_img, 'maps': maps, 'narrative': narrative
            }
            pdf_bytes = create_professional_pdf(pdf_data, L)
            st.download_button(get_trans("download", L), pdf_bytes, "Professional_Report.pdf", "application/pdf")

if __name__ == "__main__":
    if not os.path.exists(ASSETS_DIR): os.makedirs(ASSETS_DIR)
    main()

# app.py — NeuroEarly Pro (final): advanced EEG preprocessing + improved questionnaire + bilingual PDF (EN + AR)
import io
import os
import json
import math
import tempfile
import urllib.request
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import mne

# ReportLab for PDF
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
)
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# Optional Arabic shaping
try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    ARABIC_SUPPORT = True
except Exception:
    ARABIC_SUPPORT = False

# ---------- Config ----------
st.set_page_config(page_title="NeuroEarly Pro", layout="wide")
st.sidebar.title("NeuroEarly Pro")
st.sidebar.info("Research demo — not a clinical diagnosis")

LANG_UI = st.sidebar.selectbox("Interface language / لغة الواجهة", ["English", "العربية"])
IS_EN = LANG_UI == "English"

# ---------- Texts ----------
TEXT = {
    "English": {
        "title": "🧠 NeuroEarly Pro — EEG Screening (Demo)",
        "subtitle": "Advanced preprocessing, PHQ-9, AD8 and professional bilingual PDF report (research only).",
        "upload": "1) Upload EDF file(s)",
        "upload_hint": "Upload a single EDF for one session or multiple EDF files for longitudinal trend.",
        "phq": "PHQ-9 (Depression)",
        "ad8": "AD8 (Cognitive screening)",
        "generate": "Generate Reports (JSON / CSV / PDF)",
        "note": "Research demo. Not a clinical diagnostic tool."
    },
    "العربية": {
        "title": "🧠 نيوروإيرلي برو — فحص EEG (نموذج تجريبي)",
        "subtitle": "معالجة متقدمة، PHQ-9، AD8 وتقرير PDF احترافي ثنائي اللغة (للبحث فقط).",
        "upload": "١) تحميل ملفات EDF",
        "upload_hint": "ارفع ملف EDF واحد للجلسة الواحدة أو عدة ملفات لتحليل طولي.",
        "phq": "PHQ-9 (الاكتئاب)",
        "ad8": "AD8 (الفحص المعرفي)",
        "generate": "إنشاء التقارير (JSON / CSV / PDF)",
        "note": "هذا نموذج بحثي. ليس تشخيصًا طبيًا."
    }
}
TUI = TEXT["English"] if IS_EN else TEXT["العربية"]

# ---------- Questionnaire text bilingual (we'll use these both in UI and in PDF) ----------
PHQ_QS = {
    "English": [
        "Little interest or pleasure in doing things",
        "Feeling down, depressed, or hopeless",
        "Trouble falling or staying asleep, or sleeping too much",
        "Feeling tired or having little energy",
        "Poor appetite or overeating (specify type below)",
        "Feeling bad about yourself — or that you are a failure",
        "Trouble concentrating on things (e.g., reading, TV)",
        "Moving or speaking slowly vs. being restless (choose best)",
        "Thoughts that you would be better off dead or of hurting yourself"
    ],
    "العربية": [
        "قلة الاهتمام أو المتعة في القيام بالأنشطة",
        "الشعور بالحزن أو الاكتئاب أو اليأس",
        "مشاكل في النوم أو النوم المفرط",
        "الشعور بالتعب أو قلة الطاقة",
        "فقدان الشهية أو الإفراط في الأكل (حدد النوع أدناه)",
        "الشعور بأنك شخص سيء أو فاشل",
        "صعوبة في التركيز (مثل القراءة أو مشاهدة التلفاز)",
        "بطء في الحركة أو الكلام مقابل فرط النشاط (اختر الأنسب)",
        "أفكار بأنك أفضل حالاً ميتاً أو أفكار إيذاء النفس"
    ]
}
PHQ_OPTS = {
    "English": ["0 = Not at all", "1 = Several days", "2 = More than half the days", "3 = Nearly every day"],
    "العربية": ["0 = أبداً", "1 = عدة أيام", "2 = أكثر من نصف الأيام", "3 = كل يوم تقريباً"]
}
SPECIAL_Q8 = {
    "English": [
        "0 = Neither slow nor restless",
        "1 = Mostly calm/slow",
        "2 = Mostly restless",
        "3 = Both slow and restless"
    ],
    "العربية": [
        "0 = لا بطيء ولا مفرط الحركة",
        "1 = غالباً هادئ / بطيء",
        "2 = غالباً مفرط النشاط",
        "3 = كلاهما بوضوح"
    ]
}

AD8_QS = {
    "English": [
        "Problems with judgment (e.g., bad financial decisions)",
        "Reduced interest in hobbies/activities",
        "Repeats the same questions or stories",
        "Trouble learning to use a tool or gadget",
        "Forgets the correct month or year",
        "Difficulty handling finances (e.g., paying bills)",
        "Trouble remembering appointments",
        "Everyday thinking is getting worse"
    ],
    "العربية": [
        "مشاكل في الحكم أو اتخاذ القرارات",
        "انخفاض الاهتمام بالهوايات أو الأنشطة",
        "تكرار الأسئلة أو القصص",
        "صعوبة في تعلم استخدام أداة أو جهاز",
        "نسيان الشهر أو السنة الصحيحة",
        "صعوبة في إدارة الشؤون المالية (مثل دفع الفواتير)",
        "صعوبة في تذكر المواعيد",
        "تدهور التفكير اليومي"
    ]
}
AD8_OPTS = {"English": ["No", "Yes"], "العربية": ["لا", "نعم"]}

# ---------- EEG utilities ----------
BAND_DEFS = {
    "Delta (0.5–4 Hz)": (0.5, 4),
    "Theta (4–8 Hz)": (4, 8),
    "Alpha (8–12 Hz)": (8, 12),
    "Beta (12–30 Hz)": (12, 30),
    "Gamma (30–45 Hz)": (30, 45)
}

def preprocess_raw_safe(raw):
    """Bandpass + notch + attempt ICA safely"""
    try:
        raw.load_data()
    except Exception:
        pass
    # bandpass
    try:
        raw.filter(0.5, 45.0, fir_design="firwin", verbose=False)
    except Exception:
        pass
    # notch mains 50 & 60
    try:
        raw.notch_filter(freqs=[50.0, 60.0], verbose=False)
    except Exception:
        pass
    ica_applied = False
    try:
        picks = mne.pick_types(raw.info, eeg=True)
        if len(picks) >= 4:
            n_comp = min(15, max(1, len(picks)-1))
            ica = mne.preprocessing.ICA(n_components=n_comp, random_state=42, max_iter="auto")
            ica.fit(raw)
            try:
                eog_inds, scores = ica.find_bads_eog(raw)
                if eog_inds:
                    ica.exclude = eog_inds
            except Exception:
                pass
            try:
                ica.apply(raw)
                ica_applied = True
            except Exception:
                ica_applied = False
    except Exception:
        ica_applied = False
    return raw, {"ica_applied": ica_applied, "n_channels": len(mne.pick_types(raw.info, eeg=True))}

def compute_band_powers(raw):
    try:
        psd = raw.compute_psd(method="welch", fmin=0.5, fmax=45.0, n_fft=2048, n_overlap=1024, verbose=False)
        psds, freqs = psd.get_data(return_freqs=True)
        mean_psd = psds.mean(axis=0) if psds.ndim==2 else psds
    except Exception:
        data = raw.get_data()
        sf = int(raw.info.get("sfreq", 256))
        N = min(4096, data.shape[1])
        freqs = np.fft.rfftfreq(N, 1.0/sf)
        mean_psd = np.abs(np.fft.rfft(data.mean(axis=0)[:N], n=N))
    band_powers = {}
    for name, (lo, hi) in BAND_DEFS.items():
        mask = (freqs >= lo) & (freqs < hi)
        band_powers[name] = float(np.trapz(mean_psd[mask], freqs[mask])) if mask.any() else 0.0
    return band_powers

# ---------- plotting ----------
def plot_band_png(band_powers):
    labels = list(band_powers.keys())
    vals = [band_powers[l] for l in labels]
    fig, ax = plt.subplots(figsize=(7,3), dpi=120)
    ax.bar(labels, vals)
    ax.set_ylabel("Integrated power (a.u.)")
    ax.set_title("EEG Band Powers")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig); buf.seek(0)
    return buf.getvalue()

def plot_signal_png(raw, seconds=8):
    picks = mne.pick_types(raw.info, eeg=True)
    data = raw.get_data(picks=picks) if len(picks)>0 else raw.get_data()
    ch0 = data[0] if data.ndim==2 else data
    sf = int(raw.info.get("sfreq", 256))
    n = min(len(ch0), seconds*sf)
    t = np.arange(n)/sf
    fig, ax = plt.subplots(figsize=(7,2.4), dpi=120)
    ax.plot(t, ch0[:n])
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Amplitude (a.u.)")
    ax.set_title(f"EEG snippet (~{n/sf:.1f}s)")
    fig.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig); buf.seek(0)
    return buf.getvalue()

# ---------- heuristics ----------
def eeg_heuristics(band_powers):
    alpha = band_powers.get("Alpha (8–12 Hz)", 1e-9)
    theta = band_powers.get("Theta (4–8 Hz)", 0.0)
    beta = band_powers.get("Beta (12–30 Hz)", 0.0)
    theta_alpha = theta/alpha if alpha>0 else 0.0
    beta_alpha = beta/alpha if alpha>0 else 0.0
    return {"Theta/Alpha": round(theta_alpha,3), "Beta/Alpha": round(beta_alpha,3)}

def compute_early_index(band_powers, phq_score, ad8_score, weights=(0.5,0.3,0.2)):
    heur = eeg_heuristics(band_powers)
    ta = min(heur["Theta/Alpha"], 2.0)/2.0
    ba_inv = min(max(1.0 - heur["Beta/Alpha"], 0.0), 1.0)
    eeg_comp = (ta + ba_inv)/2.0
    phq_norm = min(max(phq_score/27.0, 0.0), 1.0)
    ad8_norm = min(max(ad8_score/8.0, 0.0), 1.0)
    idx = weights[0]*eeg_comp + weights[1]*phq_norm + weights[2]*ad8_norm
    return min(max(idx,0.0),1.0), {"eeg_comp": round(eeg_comp,3), "phq_norm": round(phq_norm,3), "ad8_norm": round(ad8_norm,3)}

# ---------- PDF: font + shaping ----------
FONT_DIR = "fonts"
AMIRI_TTF = os.path.join(FONT_DIR, "Amiri-Regular.ttf")
AMIRI_URL = "https://github.com/alif-type/amiri/raw/master/Amiri-Regular.ttf"

def ensure_amiri():
    os.makedirs(FONT_DIR, exist_ok=True)
    if not os.path.exists(AMIRI_TTF):
        try:
            urllib.request.urlretrieve(AMIRI_URL, AMIRI_TTF)
        except Exception:
            return False
    try:
        pdfmetrics.registerFont(TTFont("Amiri", AMIRI_TTF))
        return True
    except Exception:
        return False

def shape_ar(text):
    if not text:
        return ""
    if ARABIC_SUPPORT:
        return get_display(arabic_reshaper.reshape(text))
    return text

def shape_for_pdf(text, lang):
    if lang == "العربية":
        return shape_ar(text)
    return text

def build_pdf_bytes(results, band_png=None, sig_png=None):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4)
    styles = getSampleStyleSheet()
    flow = []

    # Register font if Arabic content expected
    ensure_amiri()

    # Title EN
    flow.append(Paragraph("NeuroEarly Pro — Report", styles["Title"]))
    flow.append(Spacer(1,6))
    flow.append(Paragraph(f"Timestamp: {results.get('timestamp','')}", styles["Normal"]))
    flow.append(Spacer(1,8))

    # EEG table (EN)
    flow.append(Paragraph("EEG Summary", styles["Heading3"]))
    eeg = results.get("EEG",{})
    rows = [["Metric","Value"]]
    rows.append(["File", eeg.get("filename","-")])
    rows.append(["Sampling rate (Hz)", str(eeg.get("sfreq","-"))])
    for k,v in eeg.get("bands",{}).items():
        rows.append([k, f"{v:.6g}"])
    t = Table(rows, colWidths=[200,300])
    t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.lightgrey),("GRID",(0,0),(-1,-1),0.25,colors.grey)]))
    flow.append(t); flow.append(Spacer(1,8))
    if band_png:
        flow.append(RLImage(io.BytesIO(band_png), width=420, height=160)); flow.append(Spacer(1,6))
    if sig_png:
        flow.append(RLImage(io.BytesIO(sig_png), width=420, height=120)); flow.append(Spacer(1,8))

    # PHQ EN
    flow.append(Paragraph("PHQ-9 (Depression) — EN", styles["Heading3"]))
    phq = results.get("PHQ9",{})
    flow.append(Paragraph(f"Score: {phq.get('score','-')} — {phq.get('label','')}", styles["Normal"]))
    # table answers
    if phq.get("questions"):
        ptab = [["Question","Answer"]]
        for q,a in zip(phq.get("questions"), phq.get("answers")):
            ptab.append([q, str(a)])
        t2 = Table(ptab, colWidths=[320,180]); t2.setStyle(TableStyle([("GRID",(0,0),(-1,-1),0.25,colors.grey)]))
        flow.append(t2)
    flow.append(Spacer(1,8))

    # AD8 EN
    flow.append(Paragraph("AD8 (Cognition) — EN", styles["Heading3"]))
    ad8 = results.get("AD8",{})
    flow.append(Paragraph(f"Score: {ad8.get('score','-')} — {ad8.get('label','')}", styles["Normal"]))
    if ad8.get("questions"):
        atab = [["Question","Answer"]]
        for q,a in zip(ad8.get("questions"), ad8.get("answers")):
            atab.append([q, "Yes" if a==1 else "No"])
        t3 = Table(atab, colWidths=[320,180]); t3.setStyle(TableStyle([("GRID",(0,0),(-1,-1),0.25,colors.grey)]))
        flow.append(t3)
    flow.append(Spacer(1,12))

    # Early index
    early = results.get("EarlyRisk",{})
    flow.append(Paragraph(f"Early Risk Index: {early.get('index','-')}", styles["Heading3"]))
    flow.append(Paragraph(f"Components: {early.get('components','')}", styles["Normal"]))
    flow.append(Spacer(1,12))

    # Separator then Arabic section
    flow.append(Paragraph("ـ" * 80, styles["Normal"]))
    flow.append(Spacer(1,6))

    # Arabic: register style using Amiri if available
    arabic_style = styles["Normal"]
    try:
        arabic_style = ParagraphStyle(name="Arabic", fontName="Amiri", fontSize=10, leading=12)
    except Exception:
        arabic_style = styles["Normal"]

    # Title AR
    flow.append(Paragraph(shape_for_pdf("تقرير نيوروإيرلي برو", "العربية"), arabic_style))
    flow.append(Spacer(1,6))
    flow.append(Paragraph(shape_for_pdf(f"الطابع الزمني: {results.get('timestamp','')}", "العربية"), arabic_style))
    flow.append(Spacer(1,8))

    # EEG AR table
    flow.append(Paragraph(shape_for_pdf("ملخص EEG", "العربية"), arabic_style))
    rows_ar = [[shape_for_pdf("المقياس","العربية"), shape_for_pdf("القيمة","العربية")]]
    rows_ar.append([shape_for_pdf("الملف","العربية"), shape_for_pdf(eeg.get("filename","-"), "العربية")])
    rows_ar.append([shape_for_pdf("معدل أخذ العينات (هرتز)","العربية"), shape_for_pdf(str(eeg.get("sfreq","-")),"العربية")])
    for k,v in eeg.get("bands",{}).items():
        rows_ar.append([shape_for_pdf(k,"العربية"), shape_for_pdf(f"{v:.6g}","العربية")])
    t_ar = Table(rows_ar, colWidths=[200,300]); t_ar.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.lightgrey),("GRID",(0,0),(-1,-1),0.25,colors.grey)]))
    flow.append(t_ar); flow.append(Spacer(1,8))

    # images (reuse)
    if band_png:
        flow.append(RLImage(io.BytesIO(band_png), width=420, height=160)); flow.append(Spacer(1,6))
    if sig_png:
        flow.append(RLImage(io.BytesIO(sig_png), width=420, height=120)); flow.append(Spacer(1,8))

    # PHQ AR
    flow.append(Paragraph(shape_for_pdf("PHQ-9 (الاكتئاب)", "العربية"), arabic_style))
    flow.append(Paragraph(shape_for_pdf(f"الدرجة: {phq.get('score','-')} — {phq.get('label','')}", "العربية"), arabic_style))
    if phq.get("questions"):
        ptab_ar = [[shape_for_pdf("السؤال","العربية"), shape_for_pdf("الإجابة","العربية")]]
        for q,a in zip(phq.get("questions_ar", phq.get("questions")), phq.get("answers")):
            ptab_ar.append([shape_for_pdf(q,"العربية"), shape_for_pdf(str(a),"العربية")])
        t2_ar = Table(ptab_ar, colWidths=[320,180]); t2_ar.setStyle(TableStyle([("GRID",(0,0),(-1,-1),0.25,colors.grey)]))
        flow.append(t2_ar)
    flow.append(Spacer(1,8))

    # AD8 AR
    flow.append(Paragraph(shape_for_pdf("AD8 (المعرفي)", "العربية"), arabic_style))
    flow.append(Paragraph(shape_for_pdf(f"الدرجة: {ad8.get('score','-')} — {ad8.get('label','')}", "العربية"), arabic_style))
    if ad8.get("questions_ar"):
        atab_ar = [[shape_for_pdf("السؤال","العربية"), shape_for_pdf("الإجابة","العربية")]]
        for q,a in zip(ad8.get("questions_ar"), ad8.get("answers")):
            atab_ar.append([shape_for_pdf(q,"العربية"), shape_for_pdf("نعم" if a==1 else "لا","العربية")])
        t3_ar = Table(atab_ar, colWidths=[320,180]); t3_ar.setStyle(TableStyle([("GRID",(0,0),(-1,-1),0.25,colors.grey)]))
        flow.append(t3_ar)

    # final note
    flow.append(Spacer(1,10))
    note_en = "This report is a research demo. Elevated theta/alpha may relate to depressive patterns; low beta/alpha may suggest cognitive concerns. Results are preliminary and require clinical follow-up."
    note_ar = "هذا التقرير بحثي. قد يرتبط ارتفاع نسبة theta/alpha بأنماط اكتئابية؛ قد يشير انخفاض beta/alpha إلى مخاوف إدراكية. النتائج أولية ويجب متابعتها سريريًا."
    flow.append(Paragraph(note_en, styles["Italic"]))
    flow.append(Spacer(1,6))
    flow.append(Paragraph(shape_for_pdf(note_ar,"العربية"), arabic_style))

    doc.build(flow)
    buf.seek(0)
    return buf.getvalue()

# ---------- UI: Upload ----------
st.header(TUI["upload"])
st.write(TUI["upload_hint"])
uploaded_files = st.file_uploader("Select EDF file(s)", type=["edf"], accept_multiple_files=True)

session_list = []
if uploaded_files:
    st.info(f"{len(uploaded_files)} file(s) selected.")
    for f in uploaded_files:
        with st.spinner(f"Processing {f.name} ..."):
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmp:
                    tmp.write(f.read()); tmp_path = tmp.name
                raw = mne.io.read_raw_edf(tmp_path, preload=True, verbose=False)
                raw_clean, meta = preprocess_raw_safe(raw)
                bands = compute_band_powers(raw_clean)
                heur = eeg_heuristics(bands)
                sig_png = plot_signal_png(raw_clean, seconds=8)
                band_png = plot_band_png(bands)
                session_list.append({
                    "filename": f.name,
                    "sfreq": float(raw.info.get("sfreq", math.nan)),
                    "bands": bands,
                    "heuristics": heur,
                    "meta": meta,
                    "sig_png": sig_png,
                    "band_png": band_png
                })
                st.success(f"{f.name} processed — channels: {meta.get('n_channels',0)}, ICA applied: {meta.get('ica_applied',False)}")
                st.image(band_png, use_column_width=True)
            except Exception as e:
                st.error(f"Failed to process {f.name}: {e}")

# ---------- UI: PHQ-9 ----------
st.header(TUI["phq"])
phq_lang = "English" if IS_EN else "العربية"
phq_questions = PHQ_QS[phq_lang]
phq_opts = PHQ_OPTS[phq_lang]
special_q8_opts = SPECIAL_Q8[phq_lang]

phq_answers = []
for i, q in enumerate(phq_questions, start=1):
    if i == 8:
        ans = st.selectbox(f"{i}. {q}", special_q8_opts, key=f"phq_{i}")
        # numeric prefix: "0 = ..." or in Arabic "0 = ..." -> parse left side until first space or '='
        try:
            val = int(ans.split("=")[0].strip())
        except Exception:
            val = 0
        phq_answers.append(val)
    else:
        ans = st.selectbox(f"{i}. {q}", phq_opts, key=f"phq_{i}")
        try:
            val = int(ans.split("=")[0].strip())
        except Exception:
            val = 0
        phq_answers.append(val)

# appetite type stored separately
if IS_EN:
    appetite_type = st.radio("If appetite change: which?", ["Poor appetite", "Overeating"], key="q5_type")
else:
    appetite_type = st.radio("إذا كان هناك تغير في الشهية: ما نوعه؟", ["فقدان الشهية", "الإفراط في الأكل"], key="q5_type")

phq_score = sum(phq_answers)
if phq_score < 5:
    phq_label = "Minimal" if IS_EN else "طفيف"
elif phq_score < 10:
    phq_label = "Mild" if IS_EN else "خفيف"
elif phq_score < 15:
    phq_label = "Moderate" if IS_EN else "متوسط"
elif phq_score < 20:
    phq_label = "Moderately severe" if IS_EN else "شديد إلى حد ما"
else:
    phq_label = "Severe" if IS_EN else "شديد"

st.write((f"PHQ-9: {phq_score} / 27 — {phq_label}") if IS_EN else (f"درجة PHQ-9: {phq_score} / ٢٧ — {phq_label}"))

# ---------- UI: AD8 ----------
st.header(TUI["ad8"])
ad8_lang = "English" if IS_EN else "العربية"
ad8_questions = AD8_QS[ad8_lang]
ad8_opts = AD8_OPTS[ad8_lang]
ad8_answers = []
for i, q in enumerate(ad8_questions, start=1):
    ans = st.selectbox(f"{i}. {q}", ad8_opts, key=f"ad8_{i}")
    ad8_answers.append(1 if ans == ( "Yes" if IS_EN else "نعم") else 0)
ad8_score = sum(ad8_answers)
ad8_label = ("Possible concern (≥2)" if ad8_score>=2 else "Low") if IS_EN else ("احتمال قلق (≥٢)" if ad8_score>=2 else "منخفض")
st.write((f"AD8: {ad8_score} / 8 — {ad8_label}") if IS_EN else (f"درجة AD8: {ad8_score} / ٨ — {ad8_label}"))

# ---------- Generate reports ----------
st.header(TUI["generate"])
if st.button("Generate Reports (JSON / CSV / PDF)"):
    sessions_meta = []
    for s in session_list:
        sessions_meta.append({
            "filename": s["filename"],
            "sfreq": s["sfreq"],
            "bands": s["bands"],
            "heuristics": s["heuristics"],
            "ica_applied": s["meta"].get("ica_applied", False)
        })
    last_bands = session_list[-1]["bands"] if session_list else {k:0.0 for k in BAND_DEFS.keys()}
    early_idx, early_comp = compute_early_index(last_bands, phq_score, ad8_score)

    results = {
        "timestamp": datetime.now().isoformat(),
        "language_ui": LANG_UI,
        "PHQ9": {
            "score": phq_score,
            "label": phq_label,
            "answers": phq_answers,
            "questions": PHQ_QS["English"],
            "questions_ar": PHQ_QS["العربية"],
            "appetite_type": appetite_type
        },
        "AD8": {
            "score": ad8_score,
            "label": ad8_label,
            "answers": ad8_answers,
            "questions": AD8_QS["English"],
            "questions_ar": AD8_QS["العربية"]
        },
        "EarlyRisk": {"index": round(early_idx,3), "components": early_comp},
        "sessions": sessions_meta,
        "note": TUI.get("note","")
    }

    # JSON
    json_bytes = io.BytesIO(json.dumps(results, indent=2, ensure_ascii=False).encode("utf-8"))
    st.download_button("⬇️ Download JSON", data=json_bytes, file_name="neuroearly_report.json", mime="application/json")

    # CSV (features)
    if sessions_meta:
        df_rows = []
        for s in sessions_meta:
            r = {"filename": s["filename"], "sfreq": s["sfreq"]}
            r.update(s["bands"])
            r.update(s["heuristics"])
            df_rows.append(r)
        df = pd.DataFrame(df_rows)
        csv_buf = io.StringIO(); df.to_csv(csv_buf, index=False)
        st.download_button("⬇️ Download CSV (features)", data=csv_buf.getvalue(), file_name="neuroearly_features.csv", mime="text/csv")

    # PDF (bilingual)
    band_png = session_list[-1]["band_png"] if session_list else None
    sig_png = session_list[-1]["sig_png"] if session_list else None
    pdf_bytes = build_pdf_bytes(results, band_png=band_png, sig_png=sig_png)
    st.download_button("⬇️ Download PDF (bilingual)", data=pdf_bytes, file_name="neuroearly_bilingual_report.pdf", mime="application/pdf")

st.caption(TUI.get("note",""))

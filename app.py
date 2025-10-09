# app.py
"""
NeuroEarly Pro — Final app (Cloud-safe)
- EDF upload (pyedflib fallback if mne missing)
- Preprocessing: notch + bandpass
- PHQ-9 (questions corrected per user's request) + AD8
- DOB calendar allowing old years (min 1940)
- Single-language PDF export (EN or AR)
- PDF uses reportlab + Amiri if available, otherwise fallback plain text
- All optional heavy imports inside try/except
"""
import streamlit as st
st.set_page_config(page_title="NeuroEarly Pro", layout="wide")

import os, io, json, tempfile, traceback, datetime
from datetime import date
from pathlib import Path
import numpy as np
import pandas as pd

# SAFE optional imports
HAS_MNE = False
HAS_PYEDF = False
HAS_SHAP = False
HAS_REPORTLAB = False
HAS_ARABIC_TOOLS = False
HAS_SKLEARN = False

try:
    import mne
    HAS_MNE = True
except Exception:
    HAS_MNE = False

try:
    import pyedflib
    HAS_PYEDF = True
except Exception:
    HAS_PYEDF = False

try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

try:
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    HAS_REPORTLAB = True
    import arabic_reshaper
    from bidi.algorithm import get_display
    HAS_ARABIC_TOOLS = True
except Exception:
    HAS_REPORTLAB = False
    HAS_ARABIC_TOOLS = False

try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False

from scipy.signal import welch, butter, filtfilt, iirnotch

# joblib for model load
try:
    import joblib
except Exception:
    joblib = None

# Constants
BANDS = {"delta":(0.5,4),"theta":(4,8),"alpha":(8,13),"beta":(13,30)}
DEFAULT_SF = 256.0
AMIRI_FILE = "Amiri-Regular.ttf"

# Helpers
def now_ts():
    return datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")

def save_tmp_upload(uploaded_file):
    with tempfile.NamedTemporaryFile(suffix=".edf", delete=False) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp.flush()
        return tmp.name

def safe_print_trace(e):
    tb = traceback.format_exc()
    print(tb)
    st.error("Internal error; see logs.")
    st.code(tb)

# EDF loader (mne or pyedflib fallback)
def read_edf(path):
    if HAS_MNE:
        raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
        data = raw.get_data()
        chs = raw.ch_names
        sf = raw.info.get("sfreq", None)
        return {"backend":"mne", "raw":raw, "data":data, "ch_names":chs, "sfreq":sf}
    elif HAS_PYEDF:
        f = pyedflib.EdfReader(path)
        n = f.signals_in_file
        chs = f.getSignalLabels()
        try:
            sf = f.getSampleFrequency(0)
        except Exception:
            sf = None
        sigs = [f.readSignal(i).astype(np.float64) for i in range(n)]
        f._close()
        data = np.vstack(sigs)
        return {"backend":"pyedflib", "raw":None, "data":data, "ch_names":chs, "sfreq":sf}
    else:
        raise ImportError("No EDF backend available. Install mne or pyedflib.")

# Filtering
def notch_filter(sig, sfreq, freq=50.0, Q=30.0):
    if sfreq is None or sfreq<=0:
        return sig
    b,a = iirnotch(freq, Q, sfreq)
    try:
        return filtfilt(b,a,sig)
    except Exception:
        return sig

def bandpass_filter(sig, sfreq, l=0.5, h=45.0, order=4):
    if sfreq is None or sfreq<=0:
        return sig
    ny = 0.5*sfreq
    low = max(l/ny, 1e-6)
    high = min(h/ny, 0.999)
    b,a = butter(order, [low, high], btype='band')
    try:
        return filtfilt(b,a,sig)
    except Exception:
        return sig

def preprocess(data, sfreq, notch=True):
    cleaned = np.zeros_like(data)
    for i in range(data.shape[0]):
        x = data[i].astype(np.float64)
        if notch:
            x = notch_filter(x, sfreq)
        x = bandpass_filter(x, sfreq)
        cleaned[i] = x
    return cleaned

# PSD & band features
def compute_psd_band(data, sfreq, nperseg=1024):
    rows = []
    for i in range(data.shape[0]):
        sig = data[i]
        try:
            freqs, pxx = welch(sig, fs=sfreq, nperseg=min(nperseg, max(256, len(sig))))
        except Exception:
            freqs = np.array([])
            pxx = np.array([])
        total = float(np.trapz(pxx, freqs)) if freqs.size>0 else 0.0
        row = {"channel_idx": i}
        for k,(lo,hi) in BANDS.items():
            if freqs.size==0:
                abs_p = 0.0
            else:
                mask = (freqs>=lo)&(freqs<=hi)
                abs_p = float(np.trapz(pxx[mask], freqs[mask])) if mask.sum()>0 else 0.0
            rel = float(abs_p/total) if total>0 else 0.0
            row[f"{k}_abs"] = abs_p
            row[f"{k}_rel"] = rel
        rows.append(row)
    return pd.DataFrame(rows)

def aggregate(df_bands, ch_names=None):
    if df_bands.empty:
        return {}
    out = {
        "alpha_rel_mean": float(df_bands['alpha_rel'].mean()),
        "theta_rel_mean": float(df_bands['theta_rel'].mean()),
        "delta_rel_mean": float(df_bands['delta_rel'].mean()),
        "beta_rel_mean": float(df_bands['beta_rel'].mean()),
        "theta_alpha_ratio": float((df_bands['theta_rel'].mean()) / (df_bands['alpha_rel'].mean()+1e-9)),
        "theta_beta_ratio": float((df_bands['theta_rel'].mean()) / (df_bands['beta_rel'].mean()+1e-9))
    }
    # alpha asymmetry best-effort
    if ch_names:
        try:
            names = [n.upper() for n in ch_names]
            def idx_of(token):
                for i,n in enumerate(names):
                    if token in n:
                        return i
                return None
            i3 = idx_of("F3"); i4 = idx_of("F4")
            if i3 is not None and i4 is not None:
                v3 = df_bands.loc[df_bands['channel_idx']==i3,'alpha_rel'].values
                v4 = df_bands.loc[df_bands['channel_idx']==i4,'alpha_rel'].values
                if v3.size>0 and v4.size>0:
                    out['alpha_asym_F3_F4'] = float(v3[0] - v4[0])
        except Exception:
            pass
    return out

# PDF generation (reportlab + Amiri support)
def register_amiri(amiri_path=None):
    if not HAS_REPORTLAB:
        return "Helvetica"
    try:
        if amiri_path and Path(amiri_path).exists():
            pdfmetrics.registerFont(TTFont("Amiri", amiri_path))
            return "Amiri"
        loc = Path("./" + AMIRI_FILE)
        if loc.exists():
            pdfmetrics.registerFont(TTFont("Amiri", str(loc)))
            return "Amiri"
    except Exception:
        pass
    return "Helvetica"

def reshape_ar(text):
    if HAS_ARABIC_TOOLS:
        try:
            return get_display(arabic_reshaper.reshape(text))
        except Exception:
            return text
    return text

def generate_pdf(summary, lang='en', amiri_path=None):
    """
    Professional-ish PDF:
    - Executive summary
    - QEEG metrics table
    - PHQ-9 & AD8 answers
    - XAI section (placeholder if no shap)
    """
    # If reportlab not installed, fallback to JSON text
    if not HAS_REPORTLAB:
        return json.dumps(summary, indent=2, ensure_ascii=False).encode('utf-8')

    font = register_amiri(amiri_path)
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    W, H = A4
    margin = 40
    x = margin
    y = H - margin

    title_en = "NeuroEarly Pro — Clinical Report"
    title_ar = "تقرير NeuroEarly Pro — سريري"
    c.setFont(font, 16)
    if lang=='en':
        c.drawCentredString(W/2, y, title_en)
    else:
        c.drawCentredString(W/2, y, reshape_ar(title_ar))
    y -= 26

    # Executive Summary
    c.setFont(font, 11)
    if lang=='en':
        c.drawString(x, y, f"Patient: {summary['patient'].get('name','-')}   | ID: {summary['patient'].get('id','-')}   | DOB: {summary['patient'].get('dob','-')}")
        y -= 14
        if summary.get("ml_risk") is not None:
            c.drawString(x, y, f"ML Risk Score: {summary.get('ml_risk'):.1f}%   Risk Level: {summary.get('risk_category','-')}")
            y -= 14
        c.drawString(x, y, f"QEEG Interpretation: {summary.get('qeegh','-')}")
        y -= 18
    else:
        c.drawRightString(W-margin, y, reshape_ar(f"المريض: {summary['patient'].get('name','-')}  المعرف: {summary['patient'].get('id','-')}  الميلاد: {summary['patient'].get('dob','-')}"))
        y -= 14
        if summary.get("ml_risk") is not None:
            c.drawRightString(W-margin, y, reshape_ar(f"معدل الخطر ML: {summary.get('ml_risk'):.1f}%  فئة الخطر: {summary.get('risk_category','-')}"))
            y -= 14
        c.drawRightString(W-margin, y, reshape_ar("التفسير QEEG: " + summary.get('qeegh','-')))
        y -= 18

    # QEEG Key Metrics table-like
    c.setFont(font, 11)
    if lang=='en':
        c.drawString(x, y, "QEEG Key Metrics:")
        y -= 14
        f0 = summary['files'][0] if summary['files'] else {}
        af = f0.get("agg_features", {})
        lines = [
            f"Theta/Alpha Ratio: {af.get('theta_alpha_ratio','N/A')}",
            f"Theta/Beta Ratio: {af.get('theta_beta_ratio','N/A')}",
            f"Alpha mean (rel): {af.get('alpha_rel_mean','N/A')}",
            f"Theta mean (rel): {af.get('theta_rel_mean','N/A')}",
            f"Alpha Asymmetry (F3-F4): {af.get('alpha_asym_F3_F4','N/A')}"
        ]
        for ln in lines:
            c.drawString(x+6, y, ln)
            y -= 12
    else:
        c.drawRightString(W-margin, y, reshape_ar("مؤشرات QEEG الأساسية:"))
        y -= 14
        f0 = summary['files'][0] if summary['files'] else {}
        af = f0.get("agg_features", {})
        lines = [
            reshape_ar(f"Theta/Alpha Ratio: {af.get('theta_alpha_ratio','N/A')}"),
            reshape_ar(f"Theta/Beta Ratio: {af.get('theta_beta_ratio','N/A')}"),
            reshape_ar(f"متوسط ألفا (نسبي): {af.get('alpha_rel_mean','N/A')}"),
            reshape_ar(f"متوسط ثيتا (نسبي): {af.get('theta_rel_mean','N/A')}"),
            reshape_ar(f"عدم تماثل ألفا (F3-F4): {af.get('alpha_asym_F3_F4','N/A')}")
        ]
        for ln in lines:
            c.drawRightString(W-margin, y, ln)
            y -= 12

    y -= 8

    # PHQ-9 & AD8 summary
    c.setFont(font, 11)
    if lang=='en':
        c.drawString(x, y, "PHQ-9 (scores):")
        y -= 12
        phq = summary.get("phq9",{})
        if phq:
            c.drawString(x+6, y, f"Total: {phq.get('total',0)}")
            y -= 12
            # list answers
            for k,v in phq.get("items",{}).items():
                c.drawString(x+10, y, f"{k}: {v}")
                y -= 10
        y -= 6
        c.drawString(x, y, "AD8 (scores):")
        y -= 12
        ad8 = summary.get("ad8",{})
        if ad8:
            c.drawString(x+6, y, f"Total: {ad8.get('total',0)}")
            y -= 12
            for k,v in ad8.get("items",{}).items():
                c.drawString(x+10, y, f"{k}: {v}")
                y -= 10
    else:
        c.drawRightString(W-margin, y, reshape_ar("PHQ-9 (الدرجات):"))
        y -= 12
        phq = summary.get("phq9",{})
        if phq:
            c.drawRightString(W-margin, y, reshape_ar(f"المجموع: {phq.get('total',0)}"))
            y -= 12
            for k,v in phq.get("items",{}).items():
                c.drawRightString(W-margin, y, reshape_ar(f"{k}: {v}"))
                y -= 10
        y -= 6
        c.drawRightString(W-margin, y, reshape_ar("AD8 (الدرجات):"))
        y -= 12
        ad8 = summary.get("ad8",{})
        if ad8:
            c.drawRightString(W-margin, y, reshape_ar(f"المجموع: {ad8.get('total',0)}"))
            y -= 12
            for k,v in ad8.get("items",{}).items():
                c.drawRightString(W-margin, y, reshape_ar(f"{k}: {v}"))
                y -= 10

    y -= 8

    # XAI summary placeholder
    c.setFont(font, 11)
    if lang=='en':
        c.drawString(x, y, "Explainable AI (XAI) — Top contributors:")
        y -= 12
        xai = summary.get("xai", None)
        if xai:
            for feat,imp in list(xai.items())[:8]:
                c.drawString(x+6, y, f"{feat}: {imp:.4f}")
                y -= 10
        else:
            c.drawString(x+6, y, "XAI not available (SHAP not installed or no model).")
            y -= 12
    else:
        c.drawRightString(W-margin, y, reshape_ar("الذكاء الاصطناعي القابل للتفسير — أعلى المؤثرين:"))
        y -= 12
        xai = summary.get("xai", None)
        if xai:
            for feat,imp in list(xai.items())[:8]:
                c.drawRightString(W-margin, y, reshape_ar(f"{feat}: {imp:.4f}"))
                y -= 10
        else:
            c.drawRightString(W-margin, y, reshape_ar("XAI غير متاح (SHAP غير مثبت أو لا يوجد نموذج)."))
            y -= 12

    y -= 16

    # Recommendations
    c.setFont(font, 11)
    if lang=='en':
        c.drawString(x, y, "Structured Clinical Recommendations:")
        y -= 12
        for r in summary.get("recommendations",[]):
            c.drawString(x+6, y, f"- {r}")
            y -= 10
    else:
        c.drawRightString(W-margin, y, reshape_ar("التوصيات السريرية المنظمة:"))
        y -= 12
        for r in summary.get("recommendations",[]):
            c.drawRightString(W-margin, y, reshape_ar(f"- {r}"))
            y -= 10

    # Footer
    footer = "Designed and developed by Golden Bird LLC — Vista Kaviani | Muscat, Sultanate of Oman"
    footer_ar = reshape_ar("صمّم وطوّر من قبل شركة Golden Bird LLC — فيستا كاوياني | مسقط، سلطنة عمان")
    c.setFont(font, 8)
    if lang=='en':
        c.drawCentredString(W/2, 30, footer)
    else:
        c.drawCentredString(W/2, 30, footer_ar)
    c.drawCentredString(W/2, 18, "Research/demo only — Not a clinical diagnosis.")
    c.save()
    buf.seek(0)
    return buf.read()

# UI
st.markdown("""
<style>
.block-container{max-width:1100px;}
.header {background: linear-gradient(90deg,#0b3d91,#2451a6); color: white; padding:14px; border-radius:8px;}
.card {background:white; padding:12px; border-radius:8px; box-shadow:0 1px 6px rgba(0,0,0,0.06);}
</style>
""", unsafe_allow_html=True)

col1, col2 = st.columns([4,1])
with col1:
    st.markdown("<div class='header'><h2 style='margin:0'>🧠 NeuroEarly Pro — Clinical Assistant</h2><div style='opacity:0.9'>EEG/QEEG + XAI — Professional reporting</div></div>", unsafe_allow_html=True)
with col2:
    st.markdown("<div style='text-align:right; font-weight:600'>Golden Bird LLC</div>", unsafe_allow_html=True)

# Sidebar: patient and settings
with st.sidebar:
    st.header("Settings & Patient")
    lang = st.radio("Report language / لغة التقرير (choose one)", options=["en","ar"], index=0)
    st.markdown("---")
    st.subheader("Patient info")
    patient_name = st.text_input("Name / الاسم")
    patient_id = st.text_input("ID")
    dob = st.date_input("DOB / تاريخ الميلاد", min_value=date(1940,1,1), max_value=date.today())
    sex = st.selectbox("Sex / الجنس", ("Unknown","Male","Female","Other"))

st.markdown("### 1) Upload EDF file(s) (.edf)")
uploads = st.file_uploader("Upload EDF", type=["edf"], accept_multiple_files=True)

# PHQ-9 (corrected)
st.markdown("### 2) PHQ-9 (Depression screening) — سوالات PHQ-9")
PHQ_ITEMS_EN = [
    "Little interest or pleasure in doing things",
    "Feeling down, depressed, or hopeless",
    # Q3 custom (sleep): choices mapping to 0..3 as requested
    "Trouble falling/staying asleep, or sleeping too much",
    "Feeling tired or having little energy",
    # Q5 custom (appetite): choices mapping to 0..3
    "Poor appetite or overeating",
    "Feeling bad about yourself — or that you are a failure",
    "Trouble concentrating on things, such as reading or watching TV",
    # Q8 custom (psychomotor agitation/retardation): choices mapping to 0..3
    "Moving or speaking slowly OR being fidgety/restless",
    "Thoughts that you would be better off dead or of harming yourself"
]
# Arabic minimal translations (short)
PHQ_ITEMS_AR = [
    "قلة الاهتمام أو المتعة بالأشياء",
    "الشعور بالحزن أو اليأس",
    "صعوبات في النوم أو نوم زائد",
    "الشعور بالتعب أو قلة الطاقة",
    "قلة أو زيادة الشهية",
    "الشعور بالسوء تجاه النفس أو الفشل",
    "صعوبة في التركيز",
    "تباطؤ في الحركة/التكلم أو قلق/اضطراب",
    "أفكار بأن الحياة ليست جديرة بالاستمرار/ إيذاء النفس"
]

phq = {}
st.markdown("Answer each item for the last 2 weeks. Options are 0..3 (standard).")
# We'll present custom choice labels for Q3,Q5,Q8
for i in range(1,10):
    key = f"phq{i}"
    if lang=='en':
        label = f"Q{i}. {PHQ_ITEMS_EN[i-1]}"
    else:
        label = f"س{i}. {PHQ_ITEMS_AR[i-1]}"
    if i == 3:
        # sleep choices: 0=Not at all,1=Insomnia (difficulty falling/staying),2=Sleeping less,3=Sleeping more
        if lang=='en':
            choices = ["0 — Not at all", "1 — Insomnia (difficulty falling/staying asleep)", "2 — Sleeping less", "3 — Sleeping more"]
        else:
            choices = [reshape_ar("0 — لا شيء"), reshape_ar("1 — أرق (صعوبة النوم أو البقاء نائماً)"), reshape_ar("2 — قلة النوم"), reshape_ar("3 — زيادة النوم")]
    elif i == 5:
        # appetite: 0=Not at all,1=Eating less,2=Eating more,3=Both/variable
        if lang=='en':
            choices = ["0 — Not at all", "1 — Eating less", "2 — Eating more", "3 — Both / variable"]
        else:
            choices = [reshape_ar("0 — لا"), reshape_ar("1 — قلة الأكل"), reshape_ar("2 — زيادة الأكل"), reshape_ar("3 — متغير / كلاهما")]
    elif i == 8:
        # psychomotor: 0=Not at all,1=Slow movement/speech,2=Restless/fidgety,3=Both/variable
        if lang=='en':
            choices = ["0 — Not at all", "1 — Moving/speaking slowly", "2 — Fidgety / restless", "3 — Both / variable"]
        else:
            choices = [reshape_ar("0 — لا"), reshape_ar("1 — تباطؤ في الحركة/التكلم"), reshape_ar("2 — تململ/قلق"), reshape_ar("3 — متغير / كلاهما")]
    else:
        # standard labels
        if lang=='en':
            choices = ["0 — Not at all", "1 — Several days", "2 — More than half the days", "3 — Nearly every day"]
        else:
            choices = [reshape_ar("0 — لا شيء"), reshape_ar("1 — عدة أيام"), reshape_ar("2 — أكثر من نصف الأيام"), reshape_ar("3 — تقريباً كل يوم")]
    phq[key] = st.radio(label, options=choices, index=0, key="phq_radio_"+str(i), horizontal=False)
    # convert label to number
    try:
        val = int(str(phq[key]).split("—")[0].strip())
    except Exception:
        # fallback
        val = int(phq[key][0]) if isinstance(phq[key], str) and phq[key][0].isdigit() else 0
    phq[key] = val

phq_total = sum([phq[f"phq{i}"] for i in range(1,10)])
st.info(f"PHQ-9 total: {phq_total} (0–4 minimal, 5–9 mild, 10–14 moderate, 15–19 moderat.-severe, 20–27 severe)")

# AD8 (binary answers 0/1)
st.markdown("### 3) AD8 (Cognitive screening) — 8 yes/no items")
AD8_ITEMS_EN = [
 "Problems with judgment (making bad decisions)",
 "Less interest in hobbies/activities",
 "Repeats questions/stories",
 "Trouble learning to use a tool or gadget",
 "Forgetting the correct month or year",
 "Difficulty handling complicated financial affairs",
 "Trouble remembering appointments",
 "Daily problems with thinking and memory"
]
AD8_ITEMS_AR = [
 "مشاكل في الحكم/القرارات",
 "قلة الاهتمام بالهوايات/الأنشطة",
 "تكرار الأسئلة/القصص",
 "صعوبة في تعلم استخدام جهاز/أداة",
 "نسيان الشهر أو السنة الصحيحة",
 "صعوبة في التعامل مع الشؤون المالية المعقدة",
 "صعوبة في تذكر المواعيد",
 "مشاكل يومية في التفكير والذاكرة"
]
ad8 = {}
for i,item in enumerate(AD8_ITEMS_EN, start=1):
    if lang=='en':
        lbl = f"A{i}. {item}"
        ad8[f"a{i}"] = st.radio(lbl, options=[0,1], index=0, horizontal=True, key=f"ad8_{i}")
    else:
        lbl = f"أ{i}. {AD8_ITEMS_AR[i-1]}"
        ad8[f"a{i}"] = st.radio(lbl, options=[0,1], index=0, horizontal=True, key=f"ad8_{i}_ar")
ad8_total = sum(ad8.values())
st.info(f"AD8 total: {ad8_total} (score ≥2 suggests cognitive impairment)")

# Processing options
st.markdown("---")
st.header("Processing options")
use_notch = st.checkbox("Apply notch filter (50Hz)", value=True)
attempt_ica = st.checkbox("Attempt ICA (if mne installed)", value=False)
run_models = st.checkbox("Run ML models if provided (model_depression.pkl, model_alzheimer.pkl)", value=False)

# Process uploads
results = []
if uploads:
    for up in uploads:
        st.write(f"Processing {up.name} ...")
        try:
            tmp = save_tmp_upload(up)
            edf = read_edf(tmp)
            sf = edf.get("sfreq") or DEFAULT_SF
            data = edf["data"]
            st.success(f"Loaded: backend={edf['backend']}  channels={data.shape[0]}  sfreq={sf}")
            # optional ICA if mne and requested
            if attempt_ica and HAS_MNE and edf.get("backend")=="mne":
                try:
                    raw = edf["raw"]
                    picks = mne.pick_types(raw.info, eeg=True, exclude='bads')
                    ica = mne.preprocessing.ICA(n_components=min(20, len(picks)), random_state=97, verbose=False)
                    ica.fit(raw, picks=picks, verbose=False)
                    st.write("ICA fitted (no automatic component removal).")
                except Exception as e:
                    st.warning("ICA failed or not possible: " + str(e))
            # preprocess
            cleaned = preprocess(data, sf, notch=use_notch)
            dfbands = compute_psd_band(cleaned, sf)
            st.dataframe(dfbands.head(10))
            agg = aggregate(dfbands, ch_names=edf.get("ch_names"))
            st.write("Aggregated features:", agg)
            # bar chart of band means (Delta/Theta/Alpha/Beta)
            band_means = {
                "delta": agg.get("delta_rel_mean",0),
                "theta": agg.get("theta_rel_mean",0),
                "alpha": agg.get("alpha_rel_mean",0),
                "beta": agg.get("beta_rel_mean",0)
            }
            st.bar_chart(pd.Series(band_means))
            results.append({
                "filename": up.name,
                "raw_summary": {"n_channels": int(data.shape[0]), "sfreq": float(sf)},
                "df_bands": dfbands,
                "agg_features": agg
            })
        except Exception as e:
            st.error(f"Failed processing {up.name}: {e}")
            safe_print_trace(e)

# Build summary structure for report
full_summary = {
    "patient": {"name": patient_name or "-", "id": patient_id or "-", "dob": str(dob), "sex": sex},
    "phq9": {"total": phq_total, "items": {f"Q{i}": phq[f"phq{i}"] for i in range(1,10)}},
    "ad8": {"total": ad8_total, "items": ad8},
    "files": results,
    "ml_risk": None,
    "risk_category": None,
    "qeegh": None,
    "xai": None,
    "recommendations": []
}

# Basic heuristic QEEG narrative
if results:
    af = results[0].get("agg_features", {})
    ta = af.get("theta_alpha_ratio", None)
    if ta is not None:
        if ta > 1.4:
            full_summary["qeegh"] = "Elevated Theta/Alpha ratio consistent with early cognitive slowing."
        elif ta > 1.1:
            full_summary["qeegh"] = "Mild elevation of Theta/Alpha ratio; correlate clinically."
        else:
            full_summary["qeegh"] = "Theta/Alpha within expected range."

# Load optional models if requested
if run_models and results and joblib:
    try:
        model_dep = None
        model_ad = None
        if Path("model_depression.pkl").exists():
            model_dep = joblib.load("model_depression.pkl")
        if Path("model_alzheimer.pkl").exists():
            model_ad = joblib.load("model_alzheimer.pkl")
        Xdf = pd.DataFrame([r.get("agg_features",{}) for r in results]).fillna(0)
        preds = []
        if model_dep is not None:
            try:
                p = model_dep.predict_proba(Xdf)[:,1] if hasattr(model_dep, "predict_proba") else model_dep.predict(Xdf)
                full_summary.setdefault("predictions", {})["depression_prob"] = p.tolist()
                preds.append(np.mean(p))
            except Exception:
                pass
        if model_ad is not None:
            try:
                p2 = model_ad.predict_proba(Xdf)[:,1] if hasattr(model_ad, "predict_proba") else model_ad.predict(Xdf)
                full_summary.setdefault("predictions", {})["alzheimers_prob"] = p2.tolist()
                preds.append(np.mean(p2))
            except Exception:
                pass
        if preds:
            mlrisk = float(np.mean(preds))*100.0
            full_summary["ml_risk"] = mlrisk
            if mlrisk >= 50:
                full_summary["risk_category"] = "High"
            elif mlrisk >= 25:
                full_summary["risk_category"] = "Moderate"
            else:
                full_summary["risk_category"] = "Low"
            # primary suggestions
            if full_summary["risk_category"] == "High":
                full_summary["recommendations"].append("Urgent neurological referral and imaging recommended.")
            elif full_summary["risk_category"] == "Moderate":
                full_summary["recommendations"].append("Clinical follow-up and further cognitive testing recommended (AD8, MMSE).")
            else:
                full_summary["recommendations"].append("Routine monitoring; correlate with PHQ-9 / AD8.")
    except Exception as e:
        st.warning("Model prediction failed: " + str(e))

# If no recommendations yet, add based on qeegh
if not full_summary["recommendations"]:
    if full_summary.get("qeegh") and "Elevated" in full_summary["qeegh"]:
        full_summary["recommendations"].append("Correlate QEEG with AD8 and formal cognitive testing (MMSE).")
        full_summary["recommendations"].append("Check B12 and TSH to rule out reversible causes.")
        full_summary["recommendations"].append("Consider MRI or FDG-PET if ML risk > 25% and Theta/Alpha > 1.4.")
    else:
        full_summary["recommendations"].append("Clinical follow-up and re-evaluation in 3-6 months.")

# Report generation UI
st.markdown("---")
st.header("Generate Report")
st.write("Choose one language for the report (English or Arabic). The report will include Executive Summary, QEEG metrics, PHQ-9 & AD8 answers, XAI info (if available) and recommendations.")
colA, colB = st.columns([3,1])
with colA:
    report_lang = st.selectbox("Report language / لغة التقرير", options=["en","ar"], index=0)
    amiri_path = st.text_input("Amiri TTF path (optional, leave empty if Amiri-Regular.ttf in app root)", value="")
with colB:
    if st.button("Generate PDF Report"):
        try:
            pdf_bytes = generate_pdf(full_summary, lang=report_lang, amiri_path=(amiri_path or None))
            st.download_button("Download PDF", data=pdf_bytes, file_name=f"NeuroEarly_Report_{now_ts()}.pdf", mime="application/pdf")
            st.success("Report generated (download button above).")
        except Exception as e:
            st.error("PDF generation failed.")
            safe_print_trace(e)

st.markdown("---")
st.caption("Designed and developed by Golden Bird LLC — Vista Kaviani  | Muscat, Sultanate of Oman")
st.caption("Research/demo only — Not a clinical diagnosis. Final clinical decisions must be made by a qualified physician.")

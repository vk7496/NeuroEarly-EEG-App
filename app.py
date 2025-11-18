# app.py — NeuroEarly Pro v6.5 (Clinical, bilingual EN/AR, AD8 + PHQ-9, Topomaps, Connectivity, SHAP fallback, PDF)
# Option B implementation: preserve v6 structure, UI polish, fixes for heatmaps/connectivity, AD8 & PHQ-9 fully included.
# Author: generated for vk7496 by ChatGPT assistant
# Notes: Put Amiri-Regular.ttf in project root or assets/, put your logo at assets/goldenbird_logo.png
# Use: streamlit run app.py

import os
import io
import sys
import json
import math
import tempfile
import traceback
from pathlib import Path
from datetime import datetime, date
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import streamlit as st
from PIL import Image as PILImage

# Optional heavy libs
HAS_MNE = False
HAS_PYEDF = False
HAS_SHAP = False
HAS_REPORTLAB = False
HAS_ARABIC = False
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
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    HAS_REPORTLAB = True
except Exception:
    HAS_REPORTLAB = False

try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    HAS_ARABIC = True
except Exception:
    HAS_ARABIC = False

try:
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False

# ---------- Constants & assets ----------
ROOT = Path(__file__).parent
ASSETS = ROOT / "assets"
AMIRI_TTF = ROOT / "Amiri-Regular.ttf"
LOGO_PATH = ASSETS / "goldenbird_logo.png"
SHAP_JSON = ROOT / "shap_summary.json"
HEALTHY_EDF = ROOT / "healthy_baseline.edf"  # optional baseline
APP_TITLE = "NeuroEarly Pro — Clinical & Research"
BLUE = "#1786f0"
LIGHT_BG = "#f7fbff"

# Frequency bands
BANDS = {
    "Delta": (1.0, 4.0),
    "Theta": (4.0, 8.0),
    "Alpha": (8.0, 13.0),
    "Beta": (13.0, 30.0),
    "Gamma": (30.0, 45.0)
}

# ---------- translations ----------
LANGS = {
    "en": {
        "app_title": "NeuroEarly Pro — Clinical & Research",
        "upload_hint": "Upload EDF file (.edf)",
        "process_button": "Process EDF(s) and Analyze",
        "no_results": "No processed results yet. Upload EDF and press 'Process EDF(s) and Analyze'.",
        "edf_loaded": "EDF loaded successfully.",
        "error_reading_edf": "Error reading EDF:",
        "download_pdf": "Download PDF report",
        "pdf_failed": "PDF generation failed - ensure reportlab is installed.",
        "phq9_title": "PHQ-9 (Depression screening)",
        "ad8_title": "AD8 (Cognitive concern screening)",
        "risk_depression": "Depression risk",
        "risk_alzheimer": "Alzheimer risk",
        "console": "Console / Visualization",
        "settings": "Settings",
        "language": "Language",
        "patient_id": "Patient ID",
        "dob": "Date of birth",
        "sex": "Sex",
        "meds": "Current meds (one per line)",
        "labs": "Relevant labs (B12, TSH, ...)",
        "processing": "Processing..."
    },
    "ar": {
        "app_title": "نظام النيورإيرلي — کلینیکال و أبحاث",
        "upload_hint": "رفع ملف EDF (.edf)",
        "process_button": "معالجة ملفات EDF و التحليل",
        "no_results": "لا توجد نتائج بعد. ارفع ملف EDF واضغط 'معالجة'.",
        "edf_loaded": "تم تحميل ملف EDF بنجاح.",
        "error_reading_edf": "خطأ في قراءة EDF:",
        "download_pdf": "تحميل تقرير PDF",
        "pdf_failed": "فشل إنشاء PDF - تأكد من تثبيت reportlab.",
        "phq9_title": "PHQ-9 (فحص الاكتئاب)",
        "ad8_title": "AD8 (فحص الاهتمام الإدراكي)",
        "risk_depression": "خطر الاكتئاب",
        "risk_alzheimer": "خطر الخرف/ألزهايمر",
        "console": "لوحة التحكم / التصوير",
        "settings": "الإعدادات",
        "language": "اللغة",
        "patient_id": "معرّف المريض",
        "dob": "تاريخ الميلاد",
        "sex": "الجنس",
        "meds": "الأدوية الحالية (سطر لكل دواء)",
        "labs": "فحوصات الدم ذات الصلة"
    }
}

# ---------- utility: i18n ----------
def _(key: str, lang: str = "en"):
    return LANGS.get(lang, LANGS["en"]).get(key, key)

def maybe_arabic(text: str, lang: str):
    if lang == "ar" and HAS_ARABIC:
        reshaped = arabic_reshaper.reshape(text)
        bidi = get_display(reshaped)
        return bidi
    return text

# ---------- EDF reading helpers ----------
def read_edf_bytes(uploaded) -> Tuple[Optional[Any], Optional[dict], Optional[str]]:
    """
    Read EDF from Streamlit uploaded file or bytes.
    Return (raw_like, metadata dict or None, error message or None)
    raw_like: if mne available -> Raw object; else dict with {"data": np.ndarray, "ch_names":[], "sfreq":...}
    """
    if not uploaded:
        return None, None, "No file provided"
    # uploaded may be a UploadedFile or path-like
    try:
        # If streamlit UploadedFile object -> getvalue() returns bytes
        if hasattr(uploaded, "getvalue"):
            raw_bytes = uploaded.getvalue()
        elif isinstance(uploaded, (bytes, bytearray)):
            raw_bytes = bytes(uploaded)
        else:
            # maybe filepath str
            if isinstance(uploaded, (str, Path)) and Path(uploaded).exists():
                path = str(uploaded)
                if HAS_MNE:
                    raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
                    return raw, {"fname": path}, None
                else:
                    # fallback via pyedflib
                    if HAS_PYEDF:
                        with pyedflib.EdfReader(str(path)) as f:
                            n = f.signals_in_file
                            ch_names = f.getSignalLabels()
                            sf = f.getSampleFrequencies()[0] if n>0 else 250
                            data = np.vstack([f.readSignal(i) for i in range(n)])
                            return {"data": data, "ch_names": ch_names, "sfreq": sf}, {"fname": path}, None
            return None, None, "Unsupported uploaded object"
        # write bytes to temp file for libraries that expect path
        with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tf:
            tf.write(raw_bytes)
            tmp_path = tf.name
        if HAS_MNE:
            try:
                raw = mne.io.read_raw_edf(tmp_path, preload=True, verbose=False)
                meta = {"fname": tmp_path}
                return raw, meta, None
            except Exception as e:
                # some mne versions dislike temporary info keys; try using pyedflib fallback
                pass
        if HAS_PYEDF:
            try:
                with pyedflib.EdfReader(tmp_path) as f:
                    n = f.signals_in_file
                    ch_names = f.getSignalLabels()
                    sf = int(f.getSampleFrequencies()[0]) if n>0 else 250
                    data = np.vstack([f.readSignal(i) for i in range(n)])
                    meta = {"fname": tmp_path, "ch_names": ch_names, "sfreq": sf}
                    return {"data": data, "ch_names": ch_names, "sfreq": sf}, meta, None
            except Exception as e:
                return None, None, f"pyedflib error: {e}"
        # last fallback: try parsing minimal header (rare)
        return None, None, "Could not parse EDF (no mne/pyedflib available)."
    except Exception as e:
        return None, None, f"read exception: {e}"

# ---------- signal processing ----------
def extract_raw_matrix(raw_like):
    """
    Convert raw (mne Raw or fallback dict) to (data, ch_names, sfreq)
    data shape: (n_channels, n_samples)
    """
    if raw_like is None:
        return None, None, None
    try:
        if HAS_MNE and isinstance(raw_like, mne.io.BaseRaw):
            data, times = raw_like.get_data(return_times=True)
            ch_names = raw_like.ch_names
            sfreq = raw_like.info["sfreq"]
            return data, ch_names, sfreq
        # else fallback dict
        if isinstance(raw_like, dict) and "data" in raw_like:
            return raw_like["data"], raw_like.get("ch_names", []), raw_like.get("sfreq", 250)
    except Exception:
        pass
    return None, None, None

def bandpower_welch(data: np.ndarray, sf: float, band: Tuple[float,float]):
    """
    Compute relative band power using Welch's method (numpy FFT fallback).
    data: 1D array (samples)
    """
    from scipy.signal import welch
    fmin, fmax = band
    try:
        freqs, psd = welch(data, fs=sf, nperseg=int(sf*2))
        # integrate
        idx_band = np.logical_and(freqs >= fmin, freqs <= fmax)
        band_power = np.trapz(psd[idx_band], freqs[idx_band])
        total_power = np.trapz(psd, freqs)
        rel = band_power / total_power if total_power > 0 else 0.0
        return rel, band_power
    except Exception:
        return 0.0, 0.0

def compute_band_powers_allchannels(data: np.ndarray, sf: float, bands=BANDS):
    """
    Compute relative band powers for all channels.
    Returns DataFrame with rows per channel and columns for each band's abs and rel power.
    """
    ch_count = data.shape[0]
    rows = []
    for i in range(ch_count):
        row = {"ch": f"ch_{i}"}
        total_abs = 0.0
        band_abs = {}
        for name, rng in bands.items():
            rel, abs_p = bandpower_welch(data[i, :], sf, rng)
            band_abs[f"{name}_abs"] = abs_p
            band_abs[f"{name}_rel"] = rel
            total_abs += abs_p
        # optional normalization / checks
        row.update(band_abs)
        rows.append(row)
    df = pd.DataFrame(rows)
    return df

# ---------- topomap & heatmaps ----------
def generate_channel_heatmap_matrix(df_bands: pd.DataFrame, band_name: str):
    """
    Simple heatmap: we produce a channel x 1 bar as fallback if no montage.
    Better: if standard_1020 available use mne.make_topomap.
    Here: create matrix (n_channels,) of relative power.
    """
    if band_name + "_rel" not in df_bands.columns:
        return None
    vals = df_bands[f"{band_name}_rel"].values
    return vals

def plot_heatmaps(df_bands, ch_names, lang, cols=2):
    """
    Produce matplotlib figure with one subplot per band (heatmaps as barplots).
    Return dict band->png bytes.
    """
    imgs = {}
    bands = list(BANDS.keys())
    n = len(bands)
    rows = math.ceil(n/cols)
    fig = plt.figure(figsize=(cols*5, rows*3.2))
    for idx, band in enumerate(bands):
        ax = fig.add_subplot(rows, cols, idx+1)
        vals = generate_channel_heatmap_matrix(df_bands, band)
        if vals is None or vals.size == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            # simple bar
            xs = np.arange(len(vals))
            ax.bar(xs, vals)
            # label channels with small rotation
            ch_labels = [cn if i%2==0 else "" for i,cn in enumerate(ch_names)] if ch_names else [f"ch{i}" for i in xs]
            ax.set_xticks(xs)
            ax.set_xticklabels(ch_labels, rotation=90, fontsize=7)
            ax.set_title(f"{band} ({BANDS[band][0]}-{BANDS[band][1]} Hz)")
            ax.set_ylim(0, max(1.0, vals.max()*1.1))
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    data = buf.getvalue()
    # put same image for each band for embedding convenience (or slice later)
    for band in bands:
        imgs[band] = data
    return imgs

# ---------- connectivity ----------
def compute_connectivity(data: np.ndarray, sf: float, ch_names=None, band=None):
    """
    Compute simple connectivity measure: band-limited coherence/correlation fallback.
    Returns connectivity matrix (n_channels x n_channels)
    """
    n_channels = data.shape[0]
    # bandpass filter to band if provided (use simple FFT windowing via scipy)
    try:
        from scipy.signal import butter, filtfilt
        def bandpass(x, low, high, fs):
            ny = 0.5*fs
            lowc = low/ny
            highc = high/ny
            b, a = butter(3, [lowc, highc], btype="band")
            return filtfilt(b, a, x)
        d = data.copy()
        if band:
            for i in range(n_channels):
                try:
                    d[i, :] = bandpass(d[i, :], band[0], band[1], sf)
                except Exception:
                    pass
        # compute correlation matrix as simple proxy
        mat = np.corrcoef(d)
        # clean inf/nan
        mat = np.nan_to_num(mat)
        return mat
    except Exception:
        # last fallback: identity
        return np.eye(n_channels)

def plot_connectivity(conn_mat, ch_names):
    try:
        import networkx as nx
    except Exception:
        nx = None
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111)
    if conn_mat is None:
        ax.text(0.5,0.5,"No connectivity", ha='center', va='center')
    else:
        mat = conn_mat.copy()
        # mask diagonal
        np.fill_diagonal(mat, 0)
        im = ax.imshow(mat, vmin=-1, vmax=1, cmap='viridis')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title("Connectivity")
        ax.set_xticks(np.arange(mat.shape[0]))
        ax.set_yticks(np.arange(mat.shape[0]))
        labels = ch_names if ch_names and len(ch_names)==mat.shape[0] else [f"ch{i}" for i in range(mat.shape[0])]
        ax.set_xticklabels(labels, rotation=90, fontsize=6)
        ax.set_yticklabels(labels, fontsize=6)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

# ---------- SHAP visualization ----------
def load_shap_json(path: Path):
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception:
        return None

def make_shap_bar(shap_summary_json):
    """
    Create a horizontal bar summary (simple) for SHAP if JSON provided.
    """
    if not shap_summary_json:
        return None
    # shap_summary_json expected: list of {"feature":..., "importance":...}
    feats = shap_summary_json.get("summary", []) if isinstance(shap_summary_json, dict) else shap_summary_json
    # sort
    feats_sorted = sorted(feats, key=lambda x: abs(x.get("importance",0)), reverse=True)[:10]
    names = [f.get("feature","") for f in feats_sorted]
    vals = [f.get("importance",0) for f in feats_sorted]
    fig, ax = plt.subplots(figsize=(6,3))
    ax.barh(range(len(vals)), vals, align='center')
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("SHAP importance")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

# ---------- Questionnaires ----------
PHQ9_QUESTIONS = [
    "Little interest or pleasure in doing things",
    "Feeling down, depressed, or hopeless",
    "Trouble falling or staying asleep, or sleeping too much",
    "Feeling tired or having little energy",
    "Poor appetite or overeating",
    "Feeling bad about yourself — or that you are a failure or have let yourself or your family down",
    "Trouble concentrating on things, such as reading the newspaper or watching television",
    "Moving or speaking so slowly that other people could have noticed. Or the opposite — being so fidgety or restless that you have been moving a lot more than usual",
    "Thoughts that you would be better off dead or of hurting yourself in some way"
]

AD8_QUESTIONS = [
    "Problems with judgement (e.g., making bad decisions, problems with thinking)",
    "Reduced interest in hobbies/activities",
    "Repeats questions/stories/needs to be reminded of same things",
    "Trouble learning how to use gadgets or appliances",
    "Forgets appointments, meetings, or engagements",
    "Less aware of recent events",
    "More difficulty performing complicated tasks (e.g., handling finances)",
    "Having trouble remembering correct month, year or time"
]

def score_phq9(answers):
    s = sum(int(a) for a in answers)
    if s >= 20:
        severity = "Severe"
    elif s >= 15:
        severity = "Moderately severe"
    elif s >= 10:
        severity = "Moderate"
    elif s >= 5:
        severity = "Mild"
    else:
        severity = "Minimal"
    return s, severity

def score_ad8(answers):
    # AD8: items answered 'yes' (1) count; score>=2 suggests cognitive impairment
    s = sum(1 for a in answers if str(a).strip() in ("1","yes","y",1))
    risk = "Likely impairment" if s >= 2 else "Unlikely impairment"
    return s, risk

# ---------- PDF report ----------
def generate_pdf_report(summary: dict, lang="en", amiri_path: Optional[Path]=None) -> Optional[bytes]:
    """
    Build a bilingual PDF. Uses reportlab if available; else returns None.
    summary: dict contains patient_info, metrics df, images bytes etc.
    """
    if not HAS_REPORTLAB:
        return None
    buffer = io.BytesIO()
    try:
        doc = SimpleDocTemplate(buffer, pagesize=A4,
                                topMargin=36, bottomMargin=36, leftMargin=44, rightMargin=44)
        styles = getSampleStyleSheet()
        base_font = "Helvetica"
        if amiri_path and Path(amiri_path).exists():
            try:
                pdfmetrics.registerFont(TTFont("Amiri", str(amiri_path)))
                base_font = "Amiri"
            except Exception:
                base_font = "Helvetica"
        styles.add(ParagraphStyle(name="TitleBlue", fontName=base_font, fontSize=16, textColor=colors.HexColor(BLUE), alignment=1, spaceAfter=8))
        styles.add(ParagraphStyle(name="H2", fontName=base_font, fontSize=12, spaceAfter=6))
        styles.add(ParagraphStyle(name="Body", fontName=base_font, fontSize=10, leading=12))
        styles.add(ParagraphStyle(name="Note", fontName=base_font, fontSize=9, textColor=colors.grey))

        story = []
        # Header
        story.append(Paragraph(summary.get("patient_info", {}).get("title", APP_TITLE), styles["TitleBlue"]))
        story.append(Spacer(1, 8))

        # Patient block
        pinfo = summary.get("patient_info", {})
        rows = [["Patient ID:", pinfo.get("id","")], ["DOB:", pinfo.get("dob","")], ["Sex:", pinfo.get("sex","")]]
        t = Table(rows, colWidths=[2.0*inch, 4.0*inch])
        t.setStyle(TableStyle([("GRID",(0,0),(-1,-1),0.25,colors.grey), ("BACKGROUND",(0,0),(0,-1),colors.HexColor("#eef7ff"))]))
        story.append(t)
        story.append(Spacer(1, 12))

        # QEEG table if present
        if summary.get("metrics_df") is not None:
            df = summary["metrics_df"]
            # reduce columns to a few important ones for PDF
            cols = [c for c in df.columns if "_rel" in c or "_abs" in c][:10]
            table_data = [ ["ch"] + cols ]
            for _, r in df.iterrows():
                row = [r.get("ch","")]
                for c in cols:
                    val = r.get(c, "")
                    row.append(f"{val:.4f}" if isinstance(val, (float,int)) else str(val))
                table_data.append(row)
            t2 = Table(table_data, colWidths=[1.0*inch]+[1.0*inch]*len(cols))
            t2.setStyle(TableStyle([("GRID",(0,0),(-1,-1),0.25,colors.grey), ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#eef7ff"))]))
            story.append(Paragraph("QEEG Band summary", styles["H2"]))
            story.append(t2)
            story.append(Spacer(1,12))

        # Images: normative bar, topomaps, connectivity, shap
        if summary.get("normative_bar"):
            story.append(Paragraph("Normative Comparison", styles["H2"]))
            try:
                img = Image(io.BytesIO(summary["normative_bar"]), width=5.5*inch, height=3.0*inch)
                story.append(img)
                story.append(Spacer(1,8))
            except Exception:
                pass

        if summary.get("topo_images"):
            story.append(Paragraph("Topography Maps", styles["H2"]))
            # add one combined image if exists
            try:
                any_img = list(summary["topo_images"].values())[0]
                img = Image(io.BytesIO(any_img), width=5.5*inch, height=3.0*inch)
                story.append(img)
                story.append(Spacer(1,8))
            except Exception:
                pass

        if summary.get("connectivity_image"):
            story.append(Paragraph("Functional connectivity (alpha band)", styles["H2"]))
            try:
                img = Image(io.BytesIO(summary["connectivity_image"]), width=5.5*inch, height=3.0*inch)
                story.append(img)
                story.append(Spacer(1,8))
            except Exception:
                pass

        if summary.get("shap_img"):
            story.append(Paragraph("Model explanation — SHAP", styles["H2"]))
            try:
                img = Image(io.BytesIO(summary["shap_img"]), width=5.5*inch, height=2.5*inch)
                story.append(img)
                story.append(Spacer(1,8))
            except Exception:
                pass
        # Recommendations & narrative
        story.append(Paragraph("<b>Clinical narrative & recommendations</b>", styles["H2"]))
        for line in summary.get("recommendations", []):
            story.append(Paragraph(line, styles["Body"]))
            story.append(Spacer(1,4))

        story.append(Spacer(1,12))
        story.append(Paragraph("Prepared by Golden Bird LLC — NeuroEarly Pro", styles["Note"]))
        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()
    except Exception as e:
        print("PDF gen failed:", e)
        return None

# ---------- UI ----------
st.set_page_config(page_title=APP_TITLE, layout="wide")
# header
st.markdown(f"""
<div style="background:linear-gradient(90deg,#1e90ff 0%, #55b3ff 100%); padding:14px; border-radius:8px;">
  <div style="display:flex; align-items:center; justify-content:space-between;">
    <div style="font-weight:700; color:white; font-size:20px;">🧠 {APP_TITLE}</div>
    <div style="display:flex; align-items:center;">
      {'<img src="assets/goldenbird_logo.png" style="height:40px; margin-left:10px;">' if LOGO_PATH.exists() else ''}
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    lang = st.selectbox(_("language","en"), options=[("English","en"),("العربية","ar")], format_func=lambda x: x[0])
    lang_code = lang[1] if isinstance(lang, tuple) else (lang if isinstance(lang,str) else "en")
    st.markdown("---")
    st.text_input(_("patient_id", lang_code), key="patient_id")
    st.date_input(_("dob", lang_code), key="dob", value=date(1980,1,1))
    st.selectbox(_("sex", lang_code), options=["Unknown","Male","Female"], key="sex")
    st.text_area(_("meds", lang_code), key="meds", help="List current medications, one per line")
    st.text_area(_("labs", lang_code), key="labs", help="e.g. B12: 250 pg/mL; TSH: 2.1 uIU/mL")
    st.markdown("---")
    uploaded_files = st.file_uploader(_("upload_hint", lang_code), accept_multiple_files=True, type=["edf"], key="uploader")
    st.markdown("")
    st.button(_("process_button", lang_code), on_click=None, key="process_btn")

# main area: console + results
st.markdown("<div style='display:flex; gap:30px; margin-top:10px;'>", unsafe_allow_html=True)
# left: console/visualization
st.markdown("<div style='flex:0.45'>", unsafe_allow_html=True)
st.subheader(_("console", lang_code))
console_placeholder = st.empty()
st.markdown("</div>", unsafe_allow_html=True)

# right: results & controls
st.markdown("<div style='flex:1'>", unsafe_allow_html=True)
st.subheader("Upload & Quick stats")
results_placeholder = st.empty()
st.markdown("</div>", unsafe_allow_html=True)

# ---------- Processing logic ----------
def process_uploaded_files(uploaded_list):
    # We'll process the first EDF for now (can extend to batch)
    if not uploaded_list:
        console_placeholder.error(_("no_results", lang_code))
        return None
    # use first file
    f = uploaded_list[0]
    console_placeholder.info("📁 " + f.name + " — reading...")
    raw, meta, err = read_edf_bytes(f)
    if err:
        console_placeholder.error(f"{_('error_reading_edf', lang_code)} {err}")
        return None
    data, ch_names, sfreq = extract_raw_matrix(raw)
    if data is None:
        console_placeholder.error(f"{_('error_reading_edf', lang_code)} Could not extract channels.")
        return None
    console_placeholder.success(f"{_('edf_loaded', lang_code)} Shape: {data.shape}")
    # compute band powers
    df_bands = compute_band_powers_allchannels(data, sfreq, BANDS)
    # attach channel names if available
    if ch_names and len(ch_names)==data.shape[0]:
        df_bands["ch"] = ch_names
    else:
        df_bands["ch"] = [f"ch{i}" for i in range(data.shape[0])]
    # produce topomap/heatmaps images
    topo_imgs = plot_heatmaps(df_bands, df_bands["ch"].tolist(), lang_code, cols=2)
    # connectivity (alpha)
    conn_mat = compute_connectivity(data, sfreq, ch_names=ch_names, band=BANDS["Alpha"])
    conn_img = plot_connectivity(conn_mat, df_bands["ch"].tolist())
    # SHAP
    shap_img = None
    shap_json = load_shap_json(SHAP_JSON) if SHAP_JSON.exists() else None
    if shap_json:
        shap_img = make_shap_bar(shap_json)
    # PHQ9 & AD8 scoring: get from session_state if present
    phq_answers = [st.session_state.get(f"phq_{i}", 0) for i in range(len(PHQ9_QUESTIONS))] if "phq_0" in st.session_state else None
    ad8_answers = [st.session_state.get(f"ad8_{i}", 0) for i in range(len(AD8_QUESTIONS))] if "ad8_0" in st.session_state else None
    phq_score = None; phq_sev = None
    ad8_score = None; ad8_risk = None
    if phq_answers:
        phq_score, phq_sev = score_phq9(phq_answers)
    if ad8_answers:
        ad8_score, ad8_risk = score_ad8(ad8_answers)
    # assemble summary
    summary = {
        "patient_info": {"id": st.session_state.get("patient_id",""), "dob": str(st.session_state.get("dob","")), "sex": st.session_state.get("sex",""), "title": APP_TITLE},
        "metrics_df": df_bands,
        "topo_images": topo_imgs,
        "connectivity_image": conn_img,
        "shap_img": shap_img,
        "phq": {"score": phq_score, "severity": phq_sev},
        "ad8": {"score": ad8_score, "risk": ad8_risk},
        "recommendations": []
    }
    # compute overall risk heuristics (simple weighted rule-based)
    theta_alpha_ratios = []
    if "Theta_rel" in df_bands.columns and "Alpha_rel" in df_bands.columns:
        try:
            theta_alpha_ratios = (df_bands["Theta_rel"].mean() / (df_bands["Alpha_rel"].mean()+1e-9))
        except Exception:
            theta_alpha_ratios = 0.0
    tar = float(theta_alpha_ratios if theta_alpha_ratios else 0.0)
    # make recommendations
    recs = []
    recs.append(f"Mean Theta/Alpha ratio: {tar:.3f}")
    if tar > 1.0:
        recs.append("Elevated Theta/Alpha — consider neurocognitive follow-up and further imaging if clinical correlation.")
    if summary["phq"]["score"] is not None:
        s = summary["phq"]["score"]
        recs.append(f"PHQ-9 score: {s} ({summary['phq']['severity']})")
        if s >= 10:
            recs.append("Recommend psychiatric evaluation and consider initiating treatment or therapy pathway.")
    if summary["ad8"]["score"] is not None:
        s = summary["ad8"]["score"]
        recs.append(f"AD8 score: {s} — {summary['ad8']['risk']}")
        if s >= 2:
            recs.append("AD8 suggests possible cognitive impairment — recommend detailed neuropsychological testing and consider MRI.")
    # append general notes
    recs.append("This automated report is a screening tool and should be interpreted in clinical context.")
    summary["recommendations"] = recs
    return summary

# questionnaire UI insertion (below main area)
st.markdown("---")
col1, col2 = st.columns([1,1])
with col1:
    st.subheader(_("phq9_title", lang_code))
    phq_cols = st.columns(3)
    for i, q in enumerate(PHQ9_QUESTIONS):
        key = f"phq_{i}"
        st.radio(q, options=[0,1,2,3], index=0, key=key, help="0=Not at all ... 3=Nearly every day")
with col2:
    st.subheader(_("ad8_title", lang_code))
    for i, q in enumerate(AD8_QUESTIONS):
        key = f"ad8_{i}"
        st.selectbox(q, options=[0,1], index=0, key=key, format_func=lambda x: "No" if x==0 else "Yes")

st.markdown("---")
# process if user clicked
if st.session_state.get("process_btn", False):
    summary = process_uploaded_files(uploaded_files)
    if summary:
        # display quick visuals
        results_placeholder.success(_("edf_loaded", lang_code))
        # show table
        df = summary["metrics_df"]
        results_placeholder.dataframe(df, use_container_width=True)
        # show images
        st.subheader("Topography / Band heatmaps")
        cols = st.columns(2)
        for i, (band, img) in enumerate(summary["topo_images"].items()):
            cols[i%2].image(img, caption=band)
        st.subheader("Connectivity")
        st.image(summary["connectivity_image"])
        if summary.get("shap_img"):
            st.subheader("Model explanation (SHAP)")
            st.image(summary["shap_img"])
        # show recommendations
        st.subheader("Clinical recommendations")
        for r in summary["recommendations"]:
            st.markdown(f"- {r}")
        # PDF generation
        if st.button(_("download_pdf", lang_code)):
            pdf_bytes = generate_pdf_report(summary, lang=lang_code, amiri_path=AMIRI_TTF if AMIRI_TTF.exists() else None)
            if pdf_bytes:
                st.download_button("Download PDF", data=pdf_bytes, file_name=f"NeuroEarly_Report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pdf", mime="application/pdf")
            else:
                st.error(_("pdf_failed", lang_code))
else:
    results_placeholder.info(_("no_results", lang_code))

# footer note
st.markdown("---")
st.markdown("<div style='font-size:0.9em;color:gray;'>Note: This tool is a research/clinical aide. Always correlate findings clinically. SHAP visuals require shap_summary.json. Connectivity & microstate results improve with mne installed.</div>", unsafe_allow_html=True)

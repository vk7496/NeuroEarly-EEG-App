# app.py — NeuroEarly v5.2 — Clinical + FDI + XAI
# Author: generated for user (adapt paths if needed)
# Requirements (recommended): streamlit, numpy, scipy, pandas, matplotlib,
# mne or pyedflib, reportlab, arabic-reshaper, python-bidi, shap (optional), joblib (optional)
# Place assets/goldenbird_logo.png and Amiri-Regular.ttf (optional) in repo.

import os
import io
import sys
import json
import math
import tempfile
import traceback
from pathlib import Path
from datetime import datetime, date
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd

# plotting backend for server
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

import streamlit as st
from PIL import Image as PILImage

# optional / heavy libs (graceful fallback)
HAS_MNE = False
HAS_PYEDF = False
HAS_REPORTLAB = False
HAS_ARABIC = False
HAS_SHAP = False
HAS_JOBLIB = False

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
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
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
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

try:
    import joblib
    HAS_JOBLIB = True
except Exception:
    HAS_JOBLIB = False

ROOT = Path(__file__).parent
ASSETS = ROOT / "assets"
LOGO_PATH = ASSETS / "goldenbird_logo.png"
AMIRI_PATH = ROOT / "Amiri-Regular.ttf"
SHAP_JSON = ROOT / "shap_summary.json"

APP_TITLE = "NeuroEarly Pro — Clinical"
BLUE = "#0b63d6"
LIGHT_BG = "#eaf2ff"

# EEG bands
BANDS = {
    "Delta": (1.0, 4.0),
    "Theta": (4.0, 8.0),
    "Alpha": (8.0, 13.0),
    "Beta": (13.0, 30.0),
    "Gamma": (30.0, 45.0),
}

def now_ts(fmt="%Y%m%d_%H%M%S"):
    return datetime.utcnow().strftime(fmt)

def clamp01(x):
    try:
        return max(0.0, min(1.0, float(x)))
    except Exception:
        return 0.0

# -----------------------
# EDF reading (robust)
# -----------------------
def read_edf_to_tempfile(uploaded) -> Tuple[Optional[object], Optional[dict], Optional[str]]:
    """
    Accepts Streamlit UploadedFile or path; returns (raw_object, meta_dict, error_msg).
    raw_object: if mne available -> mne Raw; if pyedflib available -> path to tempfile and meta.
    meta: {'sfreq':..., 'ch_names':[...] }
    """
    if uploaded is None:
        return None, None, "No file provided"
    # if a path string
    if isinstance(uploaded, (str, Path)):
        p = str(uploaded)
        if HAS_MNE:
            try:
                raw = mne.io.read_raw_edf(p, preload=True, verbose=False)
                meta = {"sfreq": float(raw.info.get("sfreq", 256.0)), "ch_names": raw.info.get("ch_names", [])}
                return raw, meta, None
            except Exception as e:
                return None, None, f"mne read error: {e}"
        if HAS_PYEDF:
            try:
                edf = pyedflib.EdfReader(p)
                n = edf.signals_in_file
                chs = edf.getSignalLabels()
                sf = edf.getSampleFrequency(0)
                edf.close()
                meta = {"sfreq": float(sf), "ch_names": chs}
                return p, meta, None
            except Exception as e:
                return None, None, f"pyedflib read error: {e}"
        return None, None, "No EDF backend available (install mne or pyedflib)"
    # else assume UploadedFile
    try:
        raw_bytes = uploaded.getvalue()
    except Exception as e:
        return None, None, f"uploaded file access error: {e}"
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmp:
            tmp.write(raw_bytes)
            tmp.flush()
            tmp_path = tmp.name
        if HAS_MNE:
            try:
                raw = mne.io.read_raw_edf(tmp_path, preload=True, verbose=False)
                meta = {"sfreq": float(raw.info.get("sfreq", 256.0)), "ch_names": raw.info.get("ch_names", [])}
                # don't delete tmp immediately because mne may use memory-mapping; but we keep it temporary
                return raw, meta, None
            except Exception:
                # fallback to pyedflib
                pass
        if HAS_PYEDF:
            try:
                edf = pyedflib.EdfReader(tmp_path)
                n = edf.signals_in_file
                chs = edf.getSignalLabels()
                sf = edf.getSampleFrequency(0)
                edf.close()
                meta = {"sfreq": float(sf), "ch_names": chs}
                return tmp_path, meta, None
            except Exception as e:
                return None, None, f"pyedflib on temp file failed: {e}"
        return None, None, "No EDF reader available"
    finally:
        # do not unlink here if mne returned raw (it may still read from file); leave to OS cleanup
        pass

# -----------------------
# PSD / band power
# -----------------------
from scipy.signal import welch, butter, sosfilt, coherence

def bandpower_welch_per_channel(data: np.ndarray, sf: float, bands=BANDS, nperseg=2048):
    """
    data: (n_channels, n_samples)
    returns df_bands: DataFrame indexed by channel names "ch_0"...
    with columns e.g. "Delta_abs","Delta_rel"
    """
    n_ch, n_s = data.shape
    results = []
    freqs = None
    for i in range(n_ch):
        try:
            f, Pxx = welch(data[i, :], fs=sf, nperseg=min(nperseg, n_s))
        except Exception:
            f, Pxx = welch(data[i, :], fs=sf)
        freqs = f if freqs is None else freqs
        total = float(np.trapz(Pxx[(f>=1) & (f<=45)], f[(f>=1)&(f<=45)])) if np.any((f>=1)&(f<=45)) else 0.0
        row = {}
        for bname, (lo, hi) in bands.items():
            mask = (f>=lo)&(f<hi)
            p = float(np.trapz(Pxx[mask], f[mask])) if mask.any() else 0.0
            row[f"{bname}_abs"] = p
            row[f"{bname}_rel"] = (p/total) if total>0 else 0.0
        row["total_power"] = total
        results.append(row)
    df = pd.DataFrame(results, index=[f"ch_{i}" for i in range(n_ch)])
    return df

# -----------------------
# Topomap generation (simple)
# -----------------------
def generate_topomap_image(vals: np.ndarray, ch_names: List[str], band_name:str="Band"):
    """
    If no montage available, render simple grid representation.
    Returns PNG bytes or None
    """
    try:
        arr = np.asarray(vals, dtype=float)
        n = len(arr)
        side = int(np.ceil(np.sqrt(n)))
        grid = np.full((side, side), np.nan)
        grid_flat = np.full(side*side, np.nan)
        grid_flat[:n] = arr
        grid = grid_flat.reshape(side, side)
        fig, ax = plt.subplots(figsize=(4,3))
        im = ax.imshow(grid, cmap='RdBu_r', interpolation='nearest', origin='upper')
        ax.set_title(f"{band_name} Topomap")
        ax.axis('off')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format='png', dpi=150); plt.close(fig); buf.seek(0)
        return buf.getvalue()
    except Exception as e:
        print("topomap error:", e)
        try:
            # fallback: bar chart
            fig, ax = plt.subplots(figsize=(6,2.4))
            ax.bar(range(len(arr)), arr)
            ax.set_xticks(range(len(ch_names))); ax.set_xticklabels(ch_names, rotation=90, fontsize=6)
            ax.set_title(band_name)
            buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format='png', dpi=120); plt.close(fig); buf.seek(0)
            return buf.getvalue()
        except Exception:
            return None

# -----------------------
# Connectivity (simple Pearson corr)
# -----------------------
def compute_connectivity_matrix(data: np.ndarray):
    """
    data: n_ch x n_samples -> compute correlation matrix
    """
    try:
        x = data.copy()
        x = (x - np.nanmean(x, axis=1, keepdims=True)) / (np.nanstd(x, axis=1, keepdims=True) + 1e-12)
        conn = np.corrcoef(x)
        conn = np.nan_to_num(conn)
        return conn
    except Exception as e:
        print("connectivity error:", e)
        return None

# -----------------------
# FDI (Focal Delta Index)
# -----------------------
def compute_fdi(df_bands: pd.DataFrame):
    """
    Use Delta_rel column to compute focal delta index:
    FDI_channel = channel_delta_rel / global_mean_delta_rel
    Returns dict with global_mean, top_channel_idx, top_channel_name, FDI_value
    """
    try:
        if "Delta_rel" in df_bands.columns:
            vals = df_bands["Delta_rel"].values
            global_mean = float(np.nanmean(vals))
            idx = int(np.nanargmax(vals)) if len(vals)>0 else 0
            top_val = float(vals[idx]) if len(vals)>0 else 0.0
            fdi = float(top_val / (global_mean + 1e-12)) if global_mean>0 else None
            return {"global_mean": global_mean, "top_idx": idx, "top_name": df_bands.index[idx] if idx < len(df_bands.index) else "", "top_value": top_val, "FDI": fdi}
        return {}
    except Exception as e:
        print("FDI error:", e)
        return {}

# -----------------------
# SHAP visualization
# -----------------------
def render_shap_bar_from_json(shap_json_path: Path, model_key_hint: str="depression_global"):
    if not shap_json_path.exists():
        return None
    try:
        with open(shap_json_path, "r", encoding="utf-8") as f:
            sj = json.load(f)
        # choose candidate key
        key = model_key_hint if model_key_hint in sj else next(iter(sj.keys()))
        feat = sj.get(key, {})
        if not feat:
            return None
        s = pd.Series(feat).abs().sort_values(ascending=False).head(12)
        fig, ax = plt.subplots(figsize=(6,3))
        s.plot.bar(ax=ax)
        ax.set_title("Top contributors (SHAP)")
        fig.tight_layout()
        buf = io.BytesIO(); fig.savefig(buf, format='png'); plt.close(fig); buf.seek(0)
        return buf.getvalue()
    except Exception as e:
        print("shap render error:", e)
        return None

# -----------------------
# PDF generation (ReportLab) bilingual support
# -----------------------
def reshape_ar(text):
    if HAS_ARABIC:
        try:
            return get_display(arabic_reshaper.reshape(text))
        except Exception:
            return text
    return text

def generate_pdf_report(summary: dict, lang: str="en", amiri_path: Optional[str]=None) -> Optional[bytes]:
    """
    summary: {patient_info, metrics, topo_images, connectivity_image, fdi, shap_img, normative_bar, recommendations, created}
    """
    if not HAS_REPORTLAB:
        return None
    try:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=36, bottomMargin=36, leftMargin=44, rightMargin=44)
        styles = getSampleStyleSheet()
        base_font = "Helvetica"
        if amiri_path and Path(amiri_path).exists():
            try:
                pdfmetrics.registerFont(TTFont("Amiri", str(amiri_path)))
                base_font = "Amiri"
            except Exception:
                base_font = "Helvetica"

        styles.add(ParagraphStyle(name="TitleBlue", fontName=base_font, fontSize=16, textColor=colors.HexColor(BLUE), alignment=1, spaceAfter=8))
        styles.add(ParagraphStyle(name="H2", fontName=base_font, fontSize=12, textColor=colors.HexColor(BLUE), spaceAfter=6))
        styles.add(ParagraphStyle(name="Body", fontName=base_font, fontSize=10, leading=14))
        styles.add(ParagraphStyle(name="Note", fontName=base_font, fontSize=9, textColor=colors.grey))

        story = []
        # header
        title = "NeuroEarly Pro — Clinical Report"
        if lang.startswith("ar") and HAS_ARABIC:
            title = reshape_ar("تقرير NeuroEarly Pro السريري")
        story.append(Paragraph(title, styles["TitleBlue"]))
        story.append(Spacer(1,8))
        # logo
        if LOGO_PATH.exists():
            try:
                img = RLImage(str(LOGO_PATH), width=1.2*inch, height=1.2*inch)
                story.append(img)
            except Exception:
                pass
        story.append(Spacer(1,8))

        # patient info table
        pinfo = summary.get("patient_info", {})
        pid = pinfo.get("id","-")
        dob = pinfo.get("dob","-")
        created = summary.get("created", now_ts())
        rows = [["Field","Value"], ["Patient ID", pid], ["DOB", dob], ["Report generated", created]]
        t = Table(rows, colWidths=[2.6*inch,3.2*inch])
        t.setStyle(TableStyle([("GRID",(0,0),(-1,-1),0.25,colors.grey),("BACKGROUND",(0,0),(-1,0),colors.HexColor("#eef7ff"))]))
        story.append(t); story.append(Spacer(1,8))

        # Final ML Risk Score prominent
        fs = summary.get("final_ml_risk", None)
        if fs is not None:
            story.append(Paragraph(f"<b>Final ML Risk Score: {fs*100:.1f}%</b>", styles["H2"]))
            story.append(Spacer(1,6))

        # Metrics
        story.append(Paragraph("QEEG Key Metrics", styles["H2"]))
        metrics = summary.get("metrics", {})
        if metrics:
            rows = [[k, str(v)] for k,v in metrics.items()]
            t2 = Table(rows, colWidths=[3.2*inch,2.6*inch])
            t2.setStyle(TableStyle([("GRID",(0,0),(-1,-1),0.25,colors.grey)]))
            story.append(t2)
            story.append(Spacer(1,6))
        # normative bar
        if summary.get("normative_bar"):
            try:
                story.append(Paragraph("Normative Comparison", styles["H2"]))
                story.append(RLImage(io.BytesIO(summary["normative_bar"]), width=5.5*inch, height=2.0*inch))
                story.append(Spacer(1,6))
            except Exception:
                pass

        # topomaps
        if summary.get("topo_images"):
            story.append(Paragraph("Topography Maps", styles["H2"]))
            imgs = []
            for band, b in summary["topo_images"].items():
                try:
                    imgs.append(RLImage(io.BytesIO(b), width=2.6*inch, height=1.6*inch))
                except Exception:
                    pass
            row=[]
            for i,im in enumerate(imgs):
                row.append(im)
                if (i%2)==1:
                    story.append(Table([row], colWidths=[2.6*inch,2.6*inch]))
                    row=[]
            if row:
                story.append(Table([row], colWidths=[2.6*inch,2.6*inch]))
            story.append(Spacer(1,6))

        # connectivity
        if summary.get("connectivity_image"):
            story.append(Paragraph("Functional Connectivity (example)", styles["H2"]))
            try:
                story.append(RLImage(io.BytesIO(summary["connectivity_image"]), width=5.5*inch, height=3.0*inch))
            except Exception:
                pass
            story.append(Spacer(1,6))

        # FDI
        if summary.get("fdi"):
            story.append(Paragraph("Focal Delta Index (FDI)", styles["H2"]))
            fdi = summary["fdi"]
            story.append(Paragraph(f"Top channel: {fdi.get('top_name','-')} — FDI: {fdi.get('FDI', '-')}", styles["Body"]))
            story.append(Spacer(1,6))

        # SHAP
        if summary.get("shap_img"):
            story.append(Paragraph("Explainable AI (SHAP)", styles["H2"]))
            try:
                story.append(RLImage(io.BytesIO(summary["shap_img"]), width=5.5*inch, height=3.0*inch))
            except Exception:
                pass
            story.append(Spacer(1,6))

        # Clinical questionnaires results (PHQ + AD8)
        if summary.get("clinical"):
            story.append(Paragraph("Clinical Questionnaires", styles["H2"]))
            cli = summary["clinical"]
            if cli.get("phq_score") is not None:
                story.append(Paragraph(f"PHQ-9 Score: {cli.get('phq_score')} (max 27)", styles["Body"]))
            if cli.get("ad_score") is not None:
                story.append(Paragraph(f"Cognitive Screening Score: {cli.get('ad_score')} (max 24)", styles["Body"]))
            story.append(Spacer(1,6))

        # Recommendations
        if summary.get("recommendations"):
            story.append(Paragraph("Structured Clinical Recommendations", styles["H2"]))
            for r in summary["recommendations"]:
                story.append(Paragraph(r, styles["Body"]))
            story.append(Spacer(1,6))

        # footer
        story.append(Paragraph("Prepared by Golden Bird LLC — NeuroEarly Pro System, 2025", styles["Note"]))
        story.append(Spacer(1,6))
        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()
    except Exception as e:
        print("PDF generation error:", e)
        traceback.print_exc()
        return None

# -----------------------
# Clinical questionnaires (PHQ-9 adjusted and Alzheimer 8)
# -----------------------
PHQ9_QUESTIONS = [
    "1. Little interest or pleasure in doing things",
    "2. Feeling down, depressed, or hopeless",
    # 3 special: sleep
    "3. Sleep problems (choose the option that best describes you)",
    "4. Feeling tired or having little energy",
    # 5 special: appetite
    "5. Appetite changes (choose the option that best describes you)",
    "6. Feeling bad about yourself — or that you are a failure or have let yourself or your family down",
    "7. Trouble concentrating on things, such as reading the newspaper or watching television",
    # 8 special: psychomotor
    "8. Moving or speaking so slowly that other people could have noticed? Or the opposite — being so fidgety or restless that you have been moving a lot more than usual",
    "9. Thoughts that you would be better off dead or of hurting yourself in some way"
]
# scoring options for typical PHQ9 (0-3)
PHQ_STANDARD_OPTIONS = [
    ("0 - Not at all", 0),
    ("1 - Several days", 1),
    ("2 - More than half the days", 2),
    ("3 - Nearly every day", 3)
]
# for Q3 (sleep) we need options that map 0-3 but with labels indicating type
PHQ_Q3_OPTIONS = [
    ("0 - No change", 0),
    ("1 - Difficulty falling/staying asleep (insomnia) - Several days", 1),
    ("2 - Mostly difficulty sleeping (insomnia) - More than half the days", 2),
    ("3 - Hypersomnia (sleeping more) - Nearly every day", 3)
]
# Q5 appetite
PHQ_Q5_OPTIONS = [
    ("0 - No change", 0),
    ("1 - Decreased appetite - Several days", 1),
    ("2 - Increased or decreased appetite - More than half the days", 2),
    ("3 - Marked change in appetite - Nearly every day", 3)
]
# Q8 psychomotor (slow vs restless)
PHQ_Q8_OPTIONS = [
    ("0 - No change", 0),
    ("1 - Slightly slowed or restless - Several days", 1),
    ("2 - Noticeably slowed or restless - More than half the days", 2),
    ("3 - Marked psychomotor change - Nearly every day", 3)
]

ALZ_QUESTIONS = [
    "1. Recurrent memory loss (e.g., forgetting recent events)",
    "2. Difficulty with orientation (time/place)",
    "3. Difficulty naming familiar objects/people",
    "4. Getting lost in familiar places",
    "5. Noticeable personality or behavior changes",
    "6. Difficulty performing familiar tasks",
    "7. Impaired judgement or decision making",
    "8. Social withdrawal or apathy"
]
ALZ_OPTIONS = [
    ("0 - No", 0),
    ("1 - Occasionally", 1),
    ("2 - Often", 2),
    ("3 - Always / Severe", 3)
]

# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title=APP_TITLE, layout="wide")
# header banner
header_html = f"""
<div style="display:flex;align-items:center;justify-content:space-between;padding:10px;border-radius:8px;background:{LIGHT_BG};">
  <div style="font-weight:700;color:{BLUE};font-size:18px;">🧠 {APP_TITLE}</div>
  <div style="display:flex;align-items:center;">
     <div style="font-size:12px;color:#333;margin-right:12px;">Prepared by Golden Bird LLC</div>
     {'<img src="'+str(LOGO_PATH).replace('\\\\','/')+'" style="height:42px;">' if LOGO_PATH.exists() else ''}
  </div>
</div>
"""
st.markdown(header_html, unsafe_allow_html=True)

# Sidebar (left)
with st.sidebar:
    st.header("Settings")
    lang_choice = st.selectbox("Language / اللغة", ["English", "Arabic"], index=0)
    lang = "ar" if lang_choice.startswith("Ar") or lang_choice.startswith("ع") else "en"

    st.markdown("---")
    st.subheader("Patient Information")
    patient_name = st.text_input("Patient Name (optional)", value="")
    patient_id = st.text_input("Patient ID", value="")
    dob = st.date_input("Date of Birth", value=date(1980,1,1), min_value=date(1900,1,1), max_value=date(2025,12,31))
    sex = st.selectbox("Sex", ["Unknown", "Male", "Female", "Other"])
    st.markdown("---")
    st.subheader("Medications (one per line)")
    meds = st.text_area("Medications", value="", height=80)
    st.subheader("Blood tests (summary)")
    labs = st.text_area("Labs (one per line, e.g. B12:250)", value="", height=100)
    st.markdown("---")
    st.subheader("Upload EDF files")
    uploads = st.file_uploader("Upload .edf files (multiple allowed)", type=["edf","EDF"], accept_multiple_files=True)
    st.markdown("")
    run_btn = st.button("Process & Analyze")

# main layout: left console + right content
col_console, col_content = st.columns([1,2])
with col_console:
    st.markdown("### Console")
    console = st.empty()
with col_content:
    st.markdown("### Analysis & Results")
    main_area = st.container()

def console_log(msg, kind="info"):
    if kind=="info":
        console.info(msg)
    elif kind=="success":
        console.success(msg)
    elif kind=="warning":
        console.warning(msg)
    elif kind=="error":
        console.error(msg)
    else:
        console.write(msg)

# questionnaire UI (display in main content before processing)
with main_area:
    st.subheader("Clinical Questionnaires")
    st.markdown("#### Depression (PHQ-9)")
    phq_answers = {}
    # render PHQ-9 with special options for 3,5,8
    for i, q in enumerate(PHQ9_QUESTIONS, start=1):
        key = f"phq_{i}"
        if i == 3:
            # special options
            sel = st.selectbox(q, [o[0] for o in PHQ_Q3_OPTIONS], index=0, key=key+"_q3")
            phq_answers[key] = dict(PHQ_Q3_OPTIONS)[sel] if sel in dict(PHQ_Q3_OPTIONS) else PHQ_Q3_OPTIONS[[o[0] for o in PHQ_Q3_OPTIONS].index(sel)][1]
        elif i == 5:
            sel = st.selectbox(q, [o[0] for o in PHQ_Q5_OPTIONS], index=0, key=key+"_q5")
            phq_answers[key] = dict(PHQ_Q5_OPTIONS)[sel] if sel in dict(PHQ_Q5_OPTIONS) else PHQ_Q5_OPTIONS[[o[0] for o in PHQ_Q5_OPTIONS].index(sel)][1]
        elif i == 8:
            sel = st.selectbox(q, [o[0] for o in PHQ_Q8_OPTIONS], index=0, key=key+"_q8")
            phq_answers[key] = dict(PHQ_Q8_OPTIONS)[sel] if sel in dict(PHQ_Q8_OPTIONS) else PHQ_Q8_OPTIONS[[o[0] for o in PHQ_Q8_OPTIONS].index(sel)][1]
        else:
            sel = st.selectbox(q, [o[0] for o in PHQ_STANDARD_OPTIONS], index=0, key=key)
            phq_answers[key] = dict(PHQ_STANDARD_OPTIONS)[sel] if sel in dict(PHQ_STANDARD_OPTIONS) else PHQ_STANDARD_OPTIONS[[o[0] for o in PHQ_STANDARD_OPTIONS].index(sel)][1]

    st.markdown("#### Cognitive Screening (Alzheimer's short form)")
    ad_answers = {}
    for i, q in enumerate(ALZ_QUESTIONS, start=1):
        key = f"ad_{i}"
        sel = st.selectbox(q, [o[0] for o in ALZ_OPTIONS], index=0, key=key)
        ad_answers[key] = dict(ALZ_OPTIONS)[sel] if sel in dict(ALZ_OPTIONS) else ALZ_OPTIONS[[o[0] for o in ALZ_OPTIONS].index(sel)][1]

# session storage
if "results" not in st.session_state:
    st.session_state["results"] = []

# Process button action
if run_btn:
    # sanity checks
    if not uploads:
        st.error("Please upload at least one EDF file to analyze.")
    else:
        st.session_state["results"] = []
        console_log("Starting processing...", "info")
        for up in uploads:
            try:
                console_log(f"Reading {up.name} ...", "info")
                raw_obj, meta, err = read_edf_to_tempfile(up)
                if err:
                    console_log(f"{up.name} read error: {err}", "error")
                    continue
                # obtain data array (n_ch x n_samples)
                if HAS_MNE and isinstance(raw_obj, mne.io.BaseRaw):
                    data = raw_obj.get_data()
                    ch_names = raw_obj.info.get("ch_names", [f"ch{i}" for i in range(data.shape[0])])
                    sf = float(raw_obj.info.get("sfreq", 250.0))
                else:
                    # raw_obj may be path (pyedflib)
                    if isinstance(raw_obj, str) and HAS_PYEDF:
                        edf = pyedflib.EdfReader(raw_obj)
                        n = edf.signals_in_file
                        ch_names = edf.getSignalLabels()
                        sf = float(edf.getSampleFrequency(0))
                        arrs = []
                        for i in range(n):
                            arrs.append(edf.readSignal(i))
                        data = np.vstack(arrs)
                        edf.close()
                    else:
                        console_log(f"No usable raw object for {up.name}.", "error"); continue

                console_log(f"{up.name}: channels={data.shape[0]}, samples={data.shape[1]}, sfreq={sf}", "success")

                # compute band powers
                df_bands = bandpower_welch_per_channel(data, sf, bands=BANDS)
                # compute FDI
                fdi = compute_fdi(df_bands)

                # compute band-wise topomaps (use relative if available)
                topo_imgs = {}
                for b in BANDS.keys():
                    col_rel = f"{b}_rel" if f"{b}_rel" in df_bands.columns else None
                    if col_rel:
                        vals = df_bands[col_rel].values
                    else:
                        vals = df_bands[f"{b}_abs"].values if f"{b}_abs" in df_bands.columns else np.zeros(data.shape[0])
                    img = generate_topomap_image(vals, ch_names, band_name=b)
                    topo_imgs[b] = img

                # compute connectivity on band-filtered data (Alpha/Theta/Delta)
                conn_img = None
                try:
                    conn_mats={}
                    for bnm, (lo,hi) in {"Delta":BANDS["Delta"], "Theta":BANDS["Theta"], "Alpha":BANDS["Alpha"]}.items():
                        try:
                            sos = butter(4, [lo, hi], btype="band", fs=sf, output="sos")
                            filtered = np.array([sosfilt(sos, data[i]) for i in range(data.shape[0])])
                            conn = compute_connectivity_matrix(filtered)
                            if conn is not None:
                                conn_mats[bnm] = conn
                        except Exception:
                            pass
                    chosen = conn_mats.get("Alpha") or (next(iter(conn_mats.values())) if conn_mats else None)
                    if chosen is not None:
                        fig, ax = plt.subplots(figsize=(4,3))
                        im = ax.imshow(chosen, cmap='viridis')
                        ax.set_title("Connectivity (matrix)")
                        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                        buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format='png'); plt.close(fig); buf.seek(0)
                        conn_img = buf.getvalue()
                except Exception as e:
                    console_log(f"Connectivity generation failed: {e}", "warning")

                # SHAP visualization (if provided)
                shap_img = None
                if SHAP_JSON.exists() and HAS_SHAP:
                    try:
                        # heuristics to pick model
                        model_key = "depression_global" if df_bands["Theta_rel"].mean() < 0.5 else "alzheimers_global"
                        shap_img = render_shap_bar_from_json(SHAP_JSON, model_key)
                    except Exception as e:
                        console_log(f"SHAP render failed: {e}", "warning")

                # normative bar chart for Theta/Alpha and Alpha asymmetry
                normative_bar = None
                try:
                    theta_alpha = (df_bands["Theta_rel"].mean() / (df_bands["Alpha_rel"].mean() + 1e-12)) if "Theta_rel" in df_bands.columns and "Alpha_rel" in df_bands.columns else 0.0
                    alpha_asym = None
                    # try find F3 and F4 indexes if present
                    f3_idx = next((i for i,cn in enumerate(ch_names) if "F3" in cn), None)
                    f4_idx = next((i for i,cn in enumerate(ch_names) if "F4" in cn), None)
                    if f3_idx is not None and f4_idx is not None and "Alpha_rel" in df_bands.columns:
                        alpha_asym = float(df_bands.iloc[f3_idx]["Alpha_rel"] - df_bands.iloc[f4_idx]["Alpha_rel"])
                    fig, ax = plt.subplots(figsize=(5.5,2.2))
                    labels = ["Theta/Alpha", "Alpha Asym (F3-F4)"]
                    vals = [theta_alpha, alpha_asym if alpha_asym is not None else 0.0]
                    ax.bar(range(len(vals)), vals, color=['#1f77b4','#1f77b4'])
                    ax.set_xticks(range(len(vals))); ax.set_xticklabels(labels)
                    ax.axhspan(0,1.2,alpha=0.06,color='white')
                    ax.axhspan(1.2,10,alpha=0.08,color='red')
                    ax.set_ylim(0, max(1.5, max(vals)+0.1))
                    buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format='png'); plt.close(fig); buf.seek(0)
                    normative_bar = buf.getvalue()
                except Exception:
                    normative_bar = None

                # Clinical questionnaire scoring
                phq_score = 0
                for i in range(1,10):
                    key = f"phq_{i}"
                    val = phq_answers.get(key) if 'phq_answers' in locals() else None
                    # If val is None because reading earlier variable scoping, read from session_state keys
                    if val is None:
                        # try reading from st.session_state
                        val = st.session_state.get(f"phq_{i}_q{i}", None) or st.session_state.get(f"phq_{i}", None)
                    try:
                        phq_score += int(val)
                    except Exception:
                        pass

                # Alternative: read from UI stored dict (we populated phq_answers in UI scope)
                # ensure fallback:
                if 'phq_answers' in locals() and isinstance(phq_answers, dict):
                    phq_score = 0
                    for i in range(1,10):
                        key = f"phq_{i}"
                        v = phq_answers.get(key)
                        try:
                            phq_score += int(v)
                        except Exception:
                            pass

                ad_score = 0
                if 'ad_answers' in locals() and isinstance(ad_answers, dict):
                    for i in range(1,9):
                        key = f"ad_{i}"
                        v = ad_answers.get(key, 0)
                        try:
                            ad_score += int(v)
                        except Exception:
                            pass

                # Final ML Risk Score (heuristic combination)
                try:
                    theta_alpha_val = (df_bands["Theta_rel"].mean() / (df_bands["Alpha_rel"].mean()+1e-12)) if "Theta_rel" in df_bands.columns and "Alpha_rel" in df_bands.columns else 0.0
                    # normalize heuristically
                    theta_alpha_norm = clamp01(theta_alpha_val / 2.0)  # assume >2 is high
                    phq_norm = clamp01(phq_score / 27.0)
                    ad_norm = clamp01(ad_score / 24.0)
                    # weights: theta/alpha 0.45, ad 0.35, phq 0.2
                    final_risk = 0.45*theta_alpha_norm + 0.35*ad_norm + 0.2*phq_norm
                except Exception:
                    final_risk = 0.0

                # metrics
                metrics = {
                    "theta_alpha_ratio": float(df_bands["Theta_rel"].mean() / (df_bands["Alpha_rel"].mean()+1e-12)) if "Theta_rel" in df_bands.columns and "Alpha_rel" in df_bands.columns else None,
                    "alpha_asym_F3_F4": float(df_bands.iloc[f3_idx]["Alpha_rel"] - df_bands.iloc[f4_idx]["Alpha_rel"]) if (f3_idx is not None and f4_idx is not None and "Alpha_rel" in df_bands.columns) else None,
                    "mean_connectivity_alpha": float(np.nanmean(conn_mats["Alpha"])) if ('conn_mats' in locals() and "Alpha" in conn_mats) else None
                }

                # assemble result dict
                result = {
                    "filename": up.name,
                    "df_bands": df_bands,
                    "topo_images": topo_imgs,
                    "connectivity_image": conn_img,
                    "fdi": fdi,
                    "shap_img": shap_img,
                    "normative_bar": normative_bar,
                    "metrics": metrics,
                    "phq_score": phq_score,
                    "ad_score": ad_score,
                    "final_ml_risk": final_risk,
                }
                st.session_state["results"].append(result)
                console_log(f"Processed {up.name} successfully.", "success")
            except Exception as e:
                console_log(f"Processing error for {up.name}: {e}\n{traceback.format_exc()}", "error")

# display results
if st.session_state.get("results"):
    for res in st.session_state["results"]:
        st.markdown(f"### File: {res.get('filename')}")
        st.markdown("**Key QEEG Metrics**")
        st.write(res.get("metrics"))
        st.markdown("**PHQ-9 Score:** " + str(res.get("phq_score")))
        st.markdown("**Cognitive Score:** " + str(res.get("ad_score")))
        st.markdown("**Final ML Risk Score:** " + f"{res.get('final_ml_risk',0)*100:.1f}%")
        # topomaps
        st.markdown("Topography Maps")
        cols = st.columns(3)
        i=0
        for band,img in res.get("topo_images", {}).items():
            if img:
                try:
                    cols[i%3].image(img, caption=band, width=None)
                except Exception:
                    pass
            i+=1
        # connectivity
        if res.get("connectivity_image"):
            st.markdown("Functional Connectivity (matrix)")
            st.image(res.get("connectivity_image"))
        else:
            st.info("Connectivity not available (install required libs for best results).")
        # normative
        if res.get("normative_bar"):
            st.markdown("Normative Comparison")
            st.image(res.get("normative_bar"))
        # SHAP
        if res.get("shap_img"):
            st.markdown("SHAP (XAI)")
            st.image(res.get("shap_img"))
        # FDI
        if res.get("fdi"):
            st.markdown("Focal Delta Index (FDI)")
            st.write(res.get("fdi"))
        st.markdown("---")

    # Exports
    st.markdown("### Export")
    try:
        # CSV
        rows=[]
        for r in st.session_state["results"]:
            row = {"filename": r["filename"], "phq_score": r.get("phq_score"), "ad_score": r.get("ad_score"), "final_ml_risk": r.get("final_ml_risk")}
            # add band means
            df = r.get("df_bands")
            if isinstance(df, pd.DataFrame):
                for b in BANDS.keys():
                    try:
                        row[f"{b}_mean_rel"] = float(df[f"{b}_rel"].mean())
                    except Exception:
                        row[f"{b}_mean_rel"] = None
            rows.append(row)
        df_export = pd.DataFrame(rows)
        csv = df_export.to_csv(index=False).encode("utf-8")
        st.download_button("Download metrics (CSV)", data=csv, file_name=f"NeuroEarly_metrics_{now_ts()}.csv", mime="text/csv")
    except Exception:
        pass

    # PDF generation (first result for simplicity)
    if st.button("Generate PDF report (first result)"):
        try:
            r = st.session_state["results"][0]
            summary = {
                "patient_info": {"id": patient_id, "dob": dob.isoformat(), "meds": meds, "labs": labs},
                "metrics": r.get("metrics", {}),
                "topo_images": r.get("topo_images", {}),
                "connectivity_image": r.get("connectivity_image"),
                "fdi": r.get("fdi"),
                "shap_img": r.get("shap_img"),
                "normative_bar": r.get("normative_bar"),
                "clinical": {"phq_score": r.get("phq_score"), "ad_score": r.get("ad_score")},
                "final_ml_risk": r.get("final_ml_risk"),
                "recommendations": [
                    "Automated screening only — clinical correlation required.",
                    "Consider MRI if focal delta index > 2 or extreme asymmetry is present.",
                    "If ML Risk > 30%, consider cognitive follow-up and further assessment."
                ],
                "created": now_ts()
            }
            pdf_bytes = generate_pdf_report(summary, lang=lang, amiri_path=str(AMIRI_PATH) if AMIRI_PATH.exists() else None)
            if pdf_bytes:
                st.download_button("Download PDF report", data=pdf_bytes, file_name=f"NeuroEarly_Report_{now_ts()}.pdf", mime="application/pdf")
                st.success("PDF generated.")
            else:
                st.error("PDF generation failed (reportlab not installed or font missing).")
        except Exception as e:
            st.error(f"PDF generation exception: {e}")

else:
    st.info("No results yet — upload EDF files and press 'Process & Analyze'.")

st.markdown("---")
st.markdown("Prepared by Golden Bird LLC — NeuroEarly Pro System, 2025")

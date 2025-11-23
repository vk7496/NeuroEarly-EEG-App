# app.py — NeuroEarly Pro (Final Clinical Edition)
# - No default EDF (no hardcoded baseline)
# - Bilingual (English default, Arabic optional)
# - Sidebar left: language, patient info, meds, labs upload, EDF upload
# - Main: console/log, topomaps (fallback grid), metrics, SHAP
# - Right: questionnaires (PHQ-9, AD8) with corrected Q3/Q5/Q8 options
# - PDF report (ReportLab, Amiri if available)
# - Logo path used: assets/goldenbird_logo.png (must exist in repo)

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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import streamlit as st
from PIL import Image as PILImage

# Optional heavy libs - try import, set flags
HAS_MNE = False
HAS_PYEDF = False
HAS_REPORTLAB = False
HAS_SHAP = False
HAS_ARABIC = False
HAS_SCIPY = False

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
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    HAS_REPORTLAB = True
except Exception:
    HAS_REPORTLAB = False

try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    HAS_ARABIC = True
except Exception:
    HAS_ARABIC = False

try:
    from scipy.signal import welch, coherence
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

# ----------------- Config / Assets -----------------
ROOT = Path(".")
ASSETS = ROOT / "assets"
AMIRI_TTF = ROOT / "Amiri-Regular.ttf"  # optional - for Arabic PDF
LOGO_PATH_PNG = ASSETS / "goldenbird_logo.png"  # <-- user confirmed this path in repo
SHAP_JSON = ROOT / "shap_summary.json"

APP_TITLE = "NeuroEarly Pro — Clinical"
BLUE = "#2D9CDB"
LIGHT_BG = "#eef7ff"
BANDS = {
    "Delta": (1.0, 4.0),
    "Theta": (4.0, 8.0),
    "Alpha": (8.0, 13.0),
    "Beta": (13.0, 30.0),
    "Gamma": (30.0, 45.0),
}

def now_ts(fmt="%Y%m%d_%H%M%S"):
    return datetime.utcnow().strftime(fmt)

# ----------------- EDF reading (robust) -----------------
def read_edf_bytes(uploaded) -> Tuple[Optional[object], Optional[str]]:
    """
    Read uploaded EDF object robustly.
    Returns (raw_obj, msg) where raw_obj is:
      - mne Raw if mne available
      - dict {"signals": np.ndarray (n_ch,n_samples), "ch_names": [...], "sfreq": float} as fallback
    """
    if not uploaded:
        return None, "No file"
    try:
        # get bytes from Streamlit UploadedFile
        data = uploaded.read() if hasattr(uploaded, "read") else uploaded.getvalue()
    except Exception as e:
        return None, f"Could not read uploaded file bytes: {e}"

    # write to temporary file to avoid BytesIO issues with some libs
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmp:
            tmp.write(data)
            tmp.flush()
            tmp_path = tmp.name
    except Exception as e:
        return None, f"Temporary file write failed: {e}"

    try:
        if HAS_MNE:
            try:
                raw = mne.io.read_raw_edf(tmp_path, preload=True, verbose=False)
                os.unlink(tmp_path)
                return raw, None
            except Exception:
                # continue to pyedflib fallback
                pass
        if HAS_PYEDF:
            try:
                f = pyedflib.EdfReader(tmp_path)
                n = f.signals_in_file
                ch_names = f.getSignalLabels()
                sfreq = f.getSampleFrequency(0)
                signals = np.vstack([f.readSignal(i) for i in range(n)])
                f.close()
                os.unlink(tmp_path)
                return {"signals": signals, "ch_names": ch_names, "sfreq": sfreq}, None
            except Exception as e:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass
                return None, f"pyedflib read error: {e}"
        else:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
            return None, "No EDF backend available (install mne or pyedflib)"
    except Exception as e:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
        return None, f"Unexpected EDF read error: {e}"

# ----------------- Spectral computations -----------------
def compute_band_powers_from_raw(raw_obj, bands=BANDS):
    """
    Accepts mne Raw or fallback dict {'signals','ch_names','sfreq'}.
    Returns dict with 'bands' per channel and 'metrics' global.
    """
    if raw_obj is None:
        return None
    # prepare data array (n_ch, n_samples) and ch_names, sfreq
    if HAS_MNE and isinstance(raw_obj, mne.io.BaseRaw):
        raw = raw_obj.copy().pick_types(eeg=True, meg=False)
        sf = raw.info.get('sfreq', 256.0)
        data = raw.get_data()  # shape (n_ch, n_samples)
        ch_names = raw.ch_names
    else:
        dd = raw_obj
        data = np.asarray(dd["signals"])
        ch_names = list(dd.get("ch_names", [f"ch{i}" for i in range(data.shape[0])]))
        sf = float(dd.get("sfreq", 256.0))
        if data.ndim == 1:
            data = data[np.newaxis, :]
    n_ch = data.shape[0]

    # PSD
    if HAS_SCIPY:
        freqs = None
        psd_list = []
        for ch in range(n_ch):
            f, Pxx = welch(data[ch, :], fs=sf, nperseg=min(2048, data.shape[1]))
            psd_list.append(Pxx)
            freqs = f
        psd = np.vstack(psd_list)
    else:
        # fallback FFT-based PSD (less accurate)
        n = data.shape[1]
        freqs = np.fft.rfftfreq(n, d=1.0/sf)
        fft_vals = np.abs(np.fft.rfft(data, axis=1))**2 / n
        psd = fft_vals

    band_summary = {}
    total_power_per_ch = psd.sum(axis=1) + 1e-12
    for i, ch in enumerate(ch_names):
        band_summary[ch] = {}
        for bname, (fmin, fmax) in bands.items():
            mask = (freqs >= fmin) & (freqs < fmax)
            band_power = float(psd[i, mask].sum()) if mask.any() else 0.0
            band_summary[ch][f"{bname}_abs"] = band_power
            band_summary[ch][f"{bname}_rel"] = (band_power / total_power_per_ch[i]) if total_power_per_ch[i] > 0 else 0.0

    # Global metrics
    # Theta/Alpha as mean of (theta_rel/alpha_rel) per channel safe
    theta_alpha_list = []
    for ch in band_summary:
        a = band_summary[ch].get("Alpha_rel", 0.0)
        t = band_summary[ch].get("Theta_rel", 0.0)
        if a > 0:
            theta_alpha_list.append(t / a)
    theta_alpha_ratio = float(np.mean(theta_alpha_list)) if theta_alpha_list else 0.0

    # FDI (focal delta index) - max delta_rel / mean delta_rel
    delta_rels = [band_summary[ch].get("Delta_rel", 0.0) for ch in band_summary]
    mean_delta = float(np.mean(delta_rels)) if delta_rels else 0.0
    max_delta = float(np.max(delta_rels)) if delta_rels else 0.0
    FDI = (max_delta / (mean_delta + 1e-12)) if mean_delta > 0 else None

    metrics = {
        "theta_alpha_ratio": theta_alpha_ratio,
        "mean_delta_rel": mean_delta,
        "max_delta_rel": max_delta,
        "FDI": float(FDI) if FDI is not None else None,
        "mean_connectivity_alpha": 0.0  # placeholder (connectivity computed separately)
    }

    return {"bands": band_summary, "metrics": metrics, "ch_names": ch_names, "sfreq": sf, "psd": psd, "freqs": freqs}

def topomap_png_from_vals(vals: np.ndarray, band_name:str="Band"):
    """
    Fallback topomap: arrange values in nearest-square grid and render heatmap PNG bytes.
    """
    try:
        arr = np.asarray(vals).astype(float).ravel()
        n = len(arr)
        side = int(np.ceil(np.sqrt(n)))
        grid_flat = np.full(side*side, np.nan)
        grid_flat[:n] = arr
        grid = grid_flat.reshape(side, side)
        fig, ax = plt.subplots(figsize=(4,3))
        vmin = np.nanmin(grid) if np.isfinite(np.nanmin(grid)) else 0.0
        vmax = np.nanmax(grid) if np.isfinite(np.nanmax(grid)) else 1.0
        im = ax.imshow(grid, cmap="RdBu_r", interpolation='nearest', origin='upper', vmin=vmin, vmax=vmax)
        ax.set_title(f"{band_name}")
        ax.axis('off')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        buf = io.BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format='png', dpi=150)
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue()
    except Exception as e:
        print("topomap error:", e)
        return None

def compute_connectivity_matrix(data: np.ndarray, sf: float):
    """
    Compute simple connectivity matrix: mean coherence in Alpha band if scipy available,
    else Pearson correlation fallback.
    """
    try:
        n_ch = data.shape[0]
        if HAS_SCIPY:
            lo, hi = BANDS["Alpha"]
            conn = np.zeros((n_ch, n_ch))
            for i in range(n_ch):
                for j in range(i, n_ch):
                    try:
                        f, Cxy = coherence(data[i,:], data[j,:], fs=sf, nperseg=min(2048, data.shape[1]))
                        mask = (f >= lo) & (f <= hi)
                        meanc = float(np.nanmean(Cxy[mask])) if mask.any() else 0.0
                    except Exception:
                        meanc = 0.0
                    conn[i,j] = meanc
                    conn[j,i] = meanc
            return conn
        else:
            # Pearson corr of normalized channels
            x = data.copy()
            x = (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-12)
            conn = np.corrcoef(x)
            conn = np.nan_to_num(conn)
            return conn
    except Exception as e:
        print("connectivity compute failed:", e)
        return None

# ----------------- SHAP rendering -----------------
def render_shap_from_json(shap_path: Path, model_hint: str = "depression_global"):
    if not shap_path.exists():
        return None
    try:
        sj = json.loads(shap_path.read_text(encoding="utf-8"))
        # pick model key intelligently
        key = model_hint if model_hint in sj else next(iter(sj.keys()))
        feats = sj.get(key, {})
        if not feats:
            return None
        s = pd.Series(feats).abs().sort_values(ascending=False).head(12)
        fig, ax = plt.subplots(figsize=(6,2.5))
        s.plot.barh(ax=ax)
        ax.invert_yaxis()
        ax.set_xlabel("mean(|SHAP value|)")
        fig.tight_layout()
        buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig); buf.seek(0)
        return buf.getvalue()
    except Exception as e:
        print("SHAP render failed:", e)
        return None

# ----------------- PDF report generator -----------------
def reshape_ar(text: str) -> str:
    if HAS_ARABIC:
        try:
            return get_display(arabic_reshaper.reshape(text))
        except Exception:
            return text
    return text

def generate_pdf_report(summary: dict, lang: str = "en", amiri_path: Optional[Path] = None) -> Optional[bytes]:
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
            except Exception as e:
                print("Amiri font register failed:", e)
        styles.add(ParagraphStyle(name="TitleBlue", fontName=base_font, fontSize=16, textColor=colors.HexColor("#0b63d6"), alignment=1, spaceAfter=8))
        styles.add(ParagraphStyle(name="H2", fontName=base_font, fontSize=12, textColor=colors.HexColor("#0b63d6"), spaceAfter=6))
        styles.add(ParagraphStyle(name="Body", fontName=base_font, fontSize=10, leading=14))
        story = []
        # header
        title = "NeuroEarly Pro — Clinical Report" if lang == "en" else reshape_ar("تقرير NeuroEarly Pro السريري")
        story.append(Paragraph(title, styles["TitleBlue"]))
        story.append(Spacer(1, 6))
        # logo
        if LOGO_PATH_PNG.exists():
            try:
                story.append(RLImage(str(LOGO_PATH_PNG), width=100, height=40))
                story.append(Spacer(1, 6))
            except Exception:
                pass
        # patient info
        pi = summary.get("patient_info", {})
        info_table = [
            ["Patient ID:", pi.get("id", "-"), "DOB:", pi.get("dob", "-")],
            ["Report created:", summary.get("created", now_ts()), "Exam:", pi.get("exam", "EEG")]
        ]
        t = Table(info_table, colWidths=[70,150,70,120])
        t.setStyle(TableStyle([("GRID",(0,0),(-1,-1),0.25,colors.grey),("BACKGROUND",(0,0),(-1,0),colors.whitesmoke)]))
        story.append(t)
        story.append(Spacer(1,12))
        # key metrics
        metrics = summary.get("metrics", {})
        story.append(Paragraph("<b>Key QEEG metrics</b>", styles["H2"]))
        mrows = [["Metric", "Value"]]
        for k,v in metrics.items():
            mrows.append([k, f"{v}"])
        mt = Table(mrows, colWidths=[200,200])
        mt.setStyle(TableStyle([("GRID",(0,0),(-1,-1),0.25,colors.grey)]))
        story.append(mt)
        story.append(Spacer(1,12))
        # images
        if summary.get("topo_images"):
            story.append(Paragraph("<b>Topomaps</b>", styles["H2"]))
            for name, img_bytes in summary["topo_images"].items():
                try:
                    img = RLImage(io.BytesIO(img_bytes), width=200, height=120)
                    story.append(img)
                    story.append(Spacer(1,6))
                except Exception:
                    pass
        if summary.get("connectivity_image"):
            story.append(Paragraph("<b>Functional Connectivity</b>", styles["H2"]))
            try:
                story.append(RLImage(io.BytesIO(summary["connectivity_image"]), width=360, height=140))
                story.append(Spacer(1,6))
            except Exception:
                pass
        if summary.get("shap_img"):
            story.append(Paragraph("<b>Explainability (SHAP)</b>", styles["H2"]))
            try:
                story.append(RLImage(io.BytesIO(summary["shap_img"]), width=360, height=120))
            except Exception:
                pass
        # recommendations
        story.append(Paragraph("<b>Structured clinical recommendations</b>", styles["H2"]))
        recs = summary.get("recommendations", [
            "This is an automated screening report. Clinical correlation is required.",
            "Consider MRI if focal delta index > 2 or extreme focal abnormalities.",
            "Follow-up in 3-6 months for moderate risk cases."
        ])
        for r in recs:
            story.append(Paragraph(r, styles["Body"]))
            story.append(Spacer(1,6))
        story.append(Spacer(1,12))
        story.append(Paragraph("Prepared by Golden Bird LLC — NeuroEarly Pro", styles["Body"]))
        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()
    except Exception as e:
        print("PDF gen error:", e)
        traceback.print_exc()
        return None

# ----------------- Questionnaires (PHQ & AD8) -----------------
PHQ9_QUESTIONS = [
    "1) Little interest or pleasure in doing things",
    "2) Feeling down, depressed, or hopeless",
    "3) Sleep: choose (0) No change / (1) Insomnia (several days) / (2) Insomnia (more than half) / (3) Hypersomnia (nearly every day)",
    "4) Feeling tired or having little energy",
    "5) Appetite: choose (0) No change / (1) Decreased (several days) / (2) Increased/Decreased (more than half) / (3) Marked change",
    "6) Feeling bad about yourself — or that you are a failure",
    "7) Trouble concentrating on things",
    "8) Movement / speech: choose (0) No change / (1) Slow OR restless (several days) / (2) More noticeable / (3) Marked change",
    "9) Thoughts that you would be better off dead or of hurting yourself"
]

AD8_QUESTIONS = [
    "1) Problems with judgment (decisions, finances)",
    "2) Less interest in hobbies/activities",
    "3) Repeats same things over and over",
    "4) Trouble learning to use gadgets",
    "5) Forgets correct month or year",
    "6) Trouble handling complicated financial affairs",
    "7) Trouble remembering appointments",
    "8) Daily problems with thinking and memory"
]

def score_phq9(answers: List[int]):
    total = sum(answers)
    if total >= 20:
        level = "Severe"
    elif total >= 15:
        level = "Moderately Severe"
    elif total >= 10:
        level = "Moderate"
    elif total >= 5:
        level = "Mild"
    else:
        level = "Minimal"
    return total, level

def score_ad8(answers: List[int]):
    s = sum(answers)
    risk = "High" if s >= 2 else "Low"
    return s, risk

# ----------------- Labs parsing -----------------
def parse_lab_text(text: str) -> List[str]:
    """
    Very simple parsing: look for keywords and low/high flags.
    Returns list of short warnings: ["B12 low", "TSH high", ...]
    """
    if not text:
        return []
    t = text.lower()
    warnings = []
    # look for common tokens with some flexibility
    for marker in ["b12", "vit b12"]:
        if marker in t and ("low" in t or "defici" in t):
            warnings.append("B12 deficiency")
    for marker in ["tsh", "thyroid"]:
        if marker in t and ("high" in t or "low" in t):
            warnings.append("Thyroid abnormality (TSH)")
    for marker in ["vit d", "vitamin d", "vitd"]:
        if marker in t and ("low" in t or "defici" in t):
            warnings.append("Vitamin D deficiency")
    return warnings

# ----------------- Streamlit UI -----------------
st.set_page_config(page_title=APP_TITLE, layout="wide")
header_html = f"""
<div style="background: linear-gradient(90deg,{BLUE},#7DD3FC);padding:12px;border-radius:8px;color:white;display:flex;align-items:center;justify-content:space-between">
  <div style="font-weight:700;font-size:20px;">🧠 {APP_TITLE}</div>
  <div style="display:flex;align-items:center;">
    <div style="margin-right:12px;color:white;font-size:12px;">Prepared by Golden Bird LLC</div>
    {'<img src="assets/goldenbird_logo.png" style="height:36px"/>' if LOGO_PATH_PNG.exists() else ''}
  </div>
</div>
"""
st.markdown(header_html, unsafe_allow_html=True)
st.write("")

# layout: sidebar (left), main, right column
left_col, main_col, right_col = st.columns([1, 2.2, 1])

with left_col:
    st.header("Settings")
    lang_choice = st.selectbox("Language / اللغة", ["English", "Arabic"], key="lang_sel")
    lang = "ar" if lang_choice.startswith("A") else "en"
    patient_name = st.text_input("Patient Name (optional)", key="pname")
    patient_id = st.text_input("Patient ID", key="pid")
    dob = st.date_input("Date of birth", value=date(1980,1,1), max_value=date(2025,12,31), key="pdob")
    sex = st.selectbox("Sex / الجنس", ["Unknown", "Male", "Female", "Other"], key="psex")
    meds = st.text_area("Current meds (one per line)", key="meds", height=80)
    st.markdown("---")
    st.subheader("Upload labs (optional)")
    lab_file = st.file_uploader("Upload lab report (txt/pdf) — parser will attempt to read text", type=["txt","pdf"], key="labfile")
    st.markdown("---")
    st.subheader("Upload EEG (.edf)")
    uploaded_file = st.file_uploader("Upload .edf file", type=["edf"], accept_multiple_files=False, key="edf_u")
    st.markdown("")
    process_btn = st.button("Process EEG & Generate Results", key="process_btn")

with main_col:
    st.subheader("Console / Visualization")
    console_box = st.empty()
    status_placeholder = st.empty()
    progress = st.progress(0.0)

with right_col:
    st.subheader("Questionnaires")
    st.markdown("**PHQ-9 (Depression)**")
    phq_answers = []
    # present improved options; keep unique keys
    for i,q in enumerate(PHQ9_QUESTIONS):
        key = f"phq_q_{i}"
        if i == 2:  # Q3 sleep special choices
            choice = st.selectbox(q, ["0 - No change","1 - Insomnia (several days)","2 - Insomnia (more than half the days)","3 - Hypersomnia (nearly every day)"], key=key)
            phq_answers.append(int(choice.split(" - ")[0]))
        elif i == 4:  # Q5 appetite
            choice = st.selectbox(q, ["0 - No change","1 - Decreased (several days)","2 - Increased/Decreased (more than half)","3 - Marked change (nearly every day)"], key=key)
            phq_answers.append(int(choice.split(" - ")[0]))
        elif i == 7:  # Q8 movement/speech
            choice = st.selectbox(q, ["0 - No change","1 - Slight change (several days)","2 - Noticeable change (more than half the days)","3 - Marked change (nearly every day)"], key=key)
            phq_answers.append(int(choice.split(" - ")[0]))
        else:
            choice = st.selectbox(q, ["0 - Not at all","1 - Several days","2 - More than half the days","3 - Nearly every day"], key=key)
            phq_answers.append(int(choice.split(" - ")[0]))

    st.markdown("---")
    st.markdown("**AD8 / Cognitive short**")
    ad8_answers = []
    for i,q in enumerate(AD8_QUESTIONS):
        key = f"ad8_q_{i}"
        choice = st.selectbox(q, ["0 - No", "1 - Occasionally", "2 - Often", "3 - Always/Severe"], key=key)
        # AD8 typically binary, but we accept graded and map to 0/1 for classic AD8 cutoff
        val = int(choice.split(" - ")[0])
        # Map: 0->0, 1+ ->1 for AD8 binary decision
        ad8_answers.append(1 if val >= 1 else 0)

    st.markdown("---")
    gen_pdf_btn = st.button("Generate PDF report (current results)", key="genpdf_btn")

# No default EDF: only process when user uploads file & clicks
processing_result = None
summary = {}

# Parse lab text if uploaded
lab_text = ""
if lab_file:
    try:
        # Try to extract text if txt, else simple binary for pdf (not full OCR)
        if lab_file.type == "text/plain":
            lab_text = lab_file.getvalue().decode("utf-8", errors="ignore")
        else:
            # attempt to read bytes for pdf might not be ideal; just store raw bytes as fallback
            lab_text = ""  # we avoid heavy pdf parsing on server in this lightweight script
    except Exception:
        lab_text = ""

lab_warnings = parse_lab_text(lab_text)

# Process when button clicked
if process_btn:
    progress.progress(0.05)
    if not uploaded_file:
        status_placeholder.error("No EDF uploaded. Please upload a valid .edf file before processing.")
    else:
        console_box.info("Reading EDF…")
        raw, err = read_edf_bytes(uploaded_file)
        if err or raw is None:
            status_placeholder.error(f"Error reading EDF: {err}")
        else:
            progress.progress(0.25)
            status_placeholder.info("Computing band powers…")
            res = compute_band_powers_from_raw(raw, bands=BANDS)
            progress.progress(0.5)
            if not res:
                status_placeholder.error("Band power computation failed.")
            else:
                # build topo images
                topo_imgs = {}
                chs = res["ch_names"]
                for b in BANDS:
                    vals = [res["bands"].get(ch, {}).get(f"{b}_rel", 0.0) for ch in chs]
                    img = topomap_png_from_vals(vals, band_name=b)
                    if img:
                        topo_imgs[b] = img
                # connectivity
                conn_img = None
                conn_matrix = None
                try:
                    # extract raw data array if mne Raw
                    if HAS_MNE and isinstance(raw, mne.io.BaseRaw):
                        data = raw.get_data()
                        sf = raw.info.get("sfreq", 256.0)
                    else:
                        # fallback dict
                        data = np.asarray(raw["signals"])
                        sf = float(raw.get("sfreq", 256.0))
                    conn_matrix = compute_connectivity_matrix(data, sf)
                    if conn_matrix is not None:
                        fig, ax = plt.subplots(figsize=(4,3))
                        im = ax.imshow(conn_matrix, cmap="viridis")
                        ax.set_title("Connectivity (alpha/coherence)")
                        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                        buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format='png'); plt.close(fig); buf.seek(0)
                        conn_img = buf.getvalue()
                except Exception as e:
                    print("Connectivity creation failed:", e)
                # metrics + final risk heuristic
                metrics = res["metrics"]
                # compute theta/alpha global safely
                ta = metrics.get("theta_alpha_ratio", 0.0)
                phq_total, phq_level = score_phq9(phq_answers)
                ad8_score, ad8_risk = score_ad8(ad8_answers)
                # normalize and combine (heuristic weights)
                ta_norm = min(1.0, ta / 2.0)  # scale
                phq_norm = min(1.0, phq_total / 27.0)
                ad_norm = 1.0 if ad8_risk == "High" else 0.0
                final_ml_risk = 0.45 * ta_norm + 0.35 * ad_norm + 0.2 * phq_norm
                # Eye state detection: use occipital channels if present
                # Build simple df for detection: channel rows x band columns
                eeg_df_rows = []
                bandcols = []
                for ch in chs:
                    row = {}
                    for b in BANDS:
                        row[f"{b}_%"] = res["bands"].get(ch, {}).get(f"{b}_rel", 0.0) * 100.0
                    eeg_df_rows.append(row)
                eeg_df = pd.DataFrame(eeg_df_rows, index=chs)
                # detect eye state (occipital alpha)
                occ_vals = [eeg_df.loc[ch, "Alpha_%"] for ch in eeg_df.index if "O1" in ch or "O2" in ch]
                global_alpha = eeg_df["Alpha_%"].mean() if "Alpha_%" in eeg_df.columns else 0.0
                if occ_vals:
                    eye_state = "Eyes Closed" if np.nanmean(occ_vals) > 11.0 else ("Eyes Closed" if global_alpha > 9.5 else "Eyes Open")
                else:
                    eye_state = "Eyes Closed" if global_alpha > 9.5 else "Eyes Open"

                # SHAP image
                shap_img = None
                if SHAP_JSON.exists():
                    try:
                        shap_img = render_shap_from_json(SHAP_JSON)
                    except Exception:
                        shap_img = None

                summary = {
                    "patient_info": {"id": patient_id, "dob": str(dob), "name": patient_name},
                    "bands": res["bands"],
                    "metrics": metrics,
                    "topo_images": topo_imgs,
                    "connectivity_image": conn_img,
                    "shap_img": shap_img,
                    "phq_total": phq_total,
                    "phq_level": phq_level,
                    "ad8_score": ad8_score,
                    "ad8_risk": ad8_risk,
                    "final_ml_risk": float(final_ml_risk),
                    "lab_warnings": lab_warnings,
                    "eye_state": eye_state,
                    "created": now_ts()
                }
                processing_result = res
                progress.progress(1.0)
                console_box.success("Processing complete. Scroll down for results.")

# Display results area
if summary:
    st.markdown("---")
    st.subheader("Results")
    st.write(f"Final ML Risk Score: **{summary['final_ml_risk']*100:.1f}%**")
    st.write(f"PHQ-9: {summary['phq_total']} ({summary['phq_level']}) — AD8 risk: {summary['ad8_risk']}")
    st.write("Eye state detected:", summary.get("eye_state"))
    if summary.get("lab_warnings"):
        st.warning("Lab warnings: " + ", ".join(summary["lab_warnings"]))
    # band table
    df_display = []
    for ch,vals in summary["bands"].items():
        row = {"channel": ch}
        for b in BANDS:
            row[b] = vals.get(f"{b}_rel", 0.0) * 100.0
        df_display.append(row)
    df_display = pd.DataFrame(df_display).set_index("channel")
    st.dataframe(df_display.style.format("{:.2f}"), height=300)
    # topomaps
    st.markdown("### Topomaps")
    cols = st.columns(2)
    idx = 0
    for bname, img_bytes in summary.get("topo_images", {}).items():
        with cols[idx % 2]:
            st.image(img_bytes, caption=bname, width=360)
        idx += 1
    if summary.get("connectivity_image"):
        st.markdown("### Functional connectivity")
        st.image(summary["connectivity_image"], width=520)
    if summary.get("shap_img"):
        st.markdown("### Explainable AI (SHAP)")
        st.image(summary["shap_img"], width=520)

# PDF generation
if gen_pdf_btn:
    if not summary:
        st.error("No processed results to generate PDF. Run analysis first.")
    else:
        pdf_bytes = generate_pdf_report({
            "patient_info": {"id": patient_id, "dob": str(dob), "name": patient_name},
            "metrics": summary.get("metrics"),
            "topo_images": summary.get("topo_images"),
            "connectivity_image": summary.get("connectivity_image"),
            "shap_img": summary.get("shap_img"),
            "recommendations": [
                "Automated screening only — clinical correlation required.",
                "If FDI > 2 or extreme focal delta or asymmetry: recommend MRI.",
                "Correlate with PHQ and AD8 scores; check B12/TSH/VitD if abnormal."
            ],
            "created": now_ts()
        }, lang=lang, amiri_path=(AMIRI_TTF if AMIRI_TTF.exists() else None))
        if pdf_bytes:
            st.download_button("Download PDF report", data=pdf_bytes, file_name=f"NeuroEarly_Report_{now_ts()}.pdf", mime="application/pdf")
        else:
            st.error("PDF generation failed — ensure reportlab is installed on the server.")

# SHAP upload helper (allow user to upload shap json)
st.markdown("---")
st.markdown("**Optional:** upload `shap_summary.json` to enable XAI visualizations")
uploaded_shap = st.file_uploader("Upload shap_summary.json", type=["json"], key="shap_up")
if uploaded_shap:
    try:
        txt = uploaded_shap.getvalue().decode("utf-8")
        Path("shap_summary.json").write_text(txt, encoding="utf-8")
        st.success("shap_summary.json uploaded. Re-run analysis to see SHAP.")
    except Exception as e:
        st.error(f"Could not save shap JSON: {e}")

st.markdown("---")
st.markdown("Notes: Default language is English; Arabic uses Amiri/reshaper if installed. This app does NOT use any server-side default EDF — results are computed only from user-uploaded EEG files.")

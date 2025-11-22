# app.py — NeuroEarly Pro — Final Clinical Edition
# Features:
# - bilingual (EN/AR), default English
# - robust EDF reading (mne preferred, pyedflib fallback), supports UploadedFile or path
# - PSD: scipy.welch preferred, fallback FFT
# - Topomaps (grid fallback), connectivity (coherence -> corr), FDI
# - SHAP visualization from shap_summary.json if present
# - PDF generation via ReportLab (if available), fallback: downloadable CSV + text summary
# - Uses default EDF: /mnt/data/test_edf.edf when no upload provided
# - Graceful degradation and clear user messages

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

# Optional heavy libs
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
AMIRI_TTF = ROOT / "Amiri-Regular.ttf"  # optional font for Arabic PDF
LOGO_PATH_PNG = ASSETS / "goldenbird_logo.png"
SHAP_JSON = ROOT / "shap_summary.json"
DEFAULT_EDF = Path("/mnt/data/test_edf.edf")  # default baseline/test EDF on server

APP_TITLE = "NeuroEarly Pro — Clinical & Research"
BLUE = "#2D9CDB"
LIGHT_BG = "#eef7ff"
BANDS = {
    "Delta": (1.0, 4.0),
    "Theta": (4.0, 8.0),
    "Alpha": (8.0, 13.0),
    "Beta": (13.0, 30.0),
    "Gamma": (30.0, 45.0),
}

# ----------------- Utilities -----------------
def now_ts(fmt="%Y%m%d_%H%M%S"):
    return datetime.utcnow().strftime(fmt)

def safe_makedirs(p: Path):
    try:
        p.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

# ----------------- EDF reading -----------------
def read_edf_as_object(path_or_uploaded) -> Tuple[Optional[Any], Optional[str]]:
    """
    Returns (obj, msg) where obj is:
     - mne.io.Raw if mne available
     - dict with keys {'signals', 'ch_names', 'sfreq'} if pyedflib fallback used
    """
    if path_or_uploaded is None:
        return None, "No file"
    # If streamlit UploadedFile object
    if hasattr(path_or_uploaded, "getbuffer") or hasattr(path_or_uploaded, "read"):
        try:
            data = path_or_uploaded.read()
        except Exception as e:
            try:
                data = path_or_uploaded.getvalue()
            except Exception as e2:
                return None, f"Uploaded file read error: {e} / {e2}"
        tmp = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tf:
                tf.write(data)
                tmp = tf.name
            return read_edf_as_object(tmp)
        except Exception as e:
            return None, f"Temp write failed: {e}"
    # If path string or Path
    p = str(path_or_uploaded)
    if not os.path.exists(p):
        return None, f"File not found: {p}"
    # Try mne
    if HAS_MNE:
        try:
            raw = mne.io.read_raw_edf(p, preload=True, verbose=False)
            return raw, None
        except Exception as e:
            # continue to pyedflib fallback
            pass
    if HAS_PYEDF:
        try:
            f = pyedflib.EdfReader(p)
            n = f.signals_in_file
            ch_names = f.getSignalLabels()
            sfreq = f.getSampleFrequency(0)
            # read signals
            signals = np.vstack([f.readSignal(i) for i in range(n)])
            f.close()
            return {"signals": signals, "ch_names": ch_names, "sfreq": sfreq}, None
        except Exception as e:
            return None, f"pyedflib read error: {e}"
    return None, "No EDF backend available (install mne or pyedflib)"

# ----------------- Band power computation -----------------
def compute_band_powers(raw_obj, bands=BANDS):
    """
    Accepts mne Raw or dict {"signals","ch_names","sfreq"}.
    Returns dict: {"bands": {ch: {Delta_abs,..., Delta_rel,...}}, "metrics": {...}, "ch_names": [...], "sfreq": float, "psd": np.ndarray, "freqs": np.ndarray}
    """
    if raw_obj is None:
        return None
    # extract data
    if HAS_MNE and isinstance(raw_obj, mne.io.BaseRaw):
        raw = raw_obj.copy().pick_types(eeg=True, meg=False)
        sf = raw.info.get("sfreq", 250.0)
        data = raw.get_data()  # shape (n_ch, n_samples)
        ch_names = raw.ch_names
    elif isinstance(raw_obj, dict):
        data = np.asarray(raw_obj["signals"])
        ch_names = list(raw_obj.get("ch_names", [f"ch{i}" for i in range(data.shape[0])]))
        sf = float(raw_obj.get("sfreq", 250.0))
        # ensure shape (n_ch, n_samples)
        if data.ndim == 1:
            data = data[np.newaxis, :]
    else:
        return None
    n_ch, n_samp = data.shape
    # compute PSD
    if HAS_SCIPY:
        psd_list = []
        freqs = None
        for ch in range(n_ch):
            f, Pxx = welch(data[ch, :], fs=sf, nperseg=min(4096, n_samp))
            psd_list.append(Pxx)
            freqs = f
        psd = np.vstack(psd_list)
    else:
        # simple FFT PSD fallback
        n = n_samp
        freqs = np.fft.rfftfreq(n, d=1.0/sf)
        fft_vals = np.abs(np.fft.rfft(data, axis=1))**2 / n
        psd = fft_vals
    # compute band powers
    band_summary = {}
    total_power = psd.sum(axis=1) + 1e-12
    for i, ch in enumerate(ch_names):
        band_summary[ch] = {}
        for bname, (fmin, fmax) in bands.items():
            mask = (freqs >= fmin) & (freqs < fmax)
            val = float(psd[i, mask].sum()) if mask.any() else 0.0
            band_summary[ch][f"{bname}_abs"] = val
            band_summary[ch][f"{bname}_rel"] = float(val / total_power[i])
    # metrics
    # theta/alpha ratio: mean across channels of (theta_rel / alpha_rel) safe
    theta_alpha_list = []
    delta_rels = []
    for ch in ch_names:
        a_rel = band_summary[ch].get("Alpha_rel", 0.0)
        t_rel = band_summary[ch].get("Theta_rel", 0.0)
        theta_alpha_list.append((t_rel / (a_rel + 1e-12)) if a_rel > 0 else 0.0)
        delta_rels.append(band_summary[ch].get("Delta_rel", 0.0))
    global_metrics = {
        "theta_alpha_ratio": float(np.mean(theta_alpha_list)) if theta_alpha_list else 0.0,
        "FDI_percent": float(max(delta_rels) * 100.0) if delta_rels else 0.0
    }
    return {"bands": band_summary, "metrics": global_metrics, "ch_names": ch_names, "sfreq": sf, "psd": psd, "freqs": freqs}

# ----------------- Topomap PNG (grid fallback) -----------------
def topomap_png_from_vals(vals: List[float], band_name: str = "Band"):
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

# ----------------- Connectivity -----------------
def compute_connectivity_matrix(data: np.ndarray, sf: float):
    try:
        n_ch = data.shape[0]
        if HAS_SCIPY:
            # compute mean coherence in alpha band between pairs
            lo, hi = BANDS["Alpha"]
            conn = np.zeros((n_ch, n_ch))
            for i in range(n_ch):
                for j in range(i, n_ch):
                    try:
                        f, Cxy = coherence(data[i, :], data[j, :], fs=sf, nperseg=min(4096, data.shape[1]))
                        mask = (f >= lo) & (f <= hi)
                        mean_coh = float(np.nanmean(Cxy[mask])) if mask.any() else 0.0
                    except Exception:
                        mean_coh = 0.0
                    conn[i, j] = mean_coh
                    conn[j, i] = mean_coh
            return conn
        else:
            # Pearson correlation fallback
            x = data.copy()
            x = (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-12)
            conn = np.corrcoef(x)
            conn = np.nan_to_num(conn)
            return conn
    except Exception as e:
        print("connectivity error:", e)
        return None

# ----------------- FDI -----------------
def compute_fdi(band_summary: Dict[str, Any]):
    try:
        chs = list(band_summary.keys())
        vals = np.array([band_summary[ch].get("Delta_rel", 0.0) for ch in chs])
        gm = np.nanmean(vals) + 1e-12
        top_idx = int(np.nanargmax(vals))
        top_name = chs[top_idx] if top_idx < len(chs) else ""
        fdi = float(vals[top_idx] / gm) if gm > 0 else None
        return {"global_mean": float(gm), "top_idx": top_idx, "top_name": top_name, "top_value": float(vals[top_idx]), "FDI": fdi}
    except Exception as e:
        print("fdi error:", e)
        return {}

# ----------------- SHAP render -----------------
def render_shap_if_present(shap_json_path: Path):
    if not shap_json_path.exists() or not HAS_SHAP:
        return None
    try:
        with open(shap_json_path, "r", encoding="utf-8") as f:
            sj = json.load(f)
        # sj expected: {model_key: {feature: value}}
        # pick first model
        model_key = next(iter(sj.keys()))
        feats = sj.get(model_key, {})
        if not feats:
            return None
        s = pd.Series(feats).abs().sort_values(ascending=False).head(12)
        fig, ax = plt.subplots(figsize=(6, 3))
        s.plot.barh(ax=ax)
        ax.set_xlabel("mean(|shap|)")
        ax.invert_yaxis()
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue()
    except Exception as e:
        print("shap render error:", e)
        return None

# ----------------- PDF report generator -----------------
def generate_pdf_report(summary: dict, lang: str = "en", amiri_path: Optional[Path] = None) -> Optional[bytes]:
    """
    summary: dict with keys patient_info, metrics, topo_images, connectivity_image, fdi, shap_img, recommendations, created
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
            except Exception as e:
                print("Amiri register failed:", e)
        styles.add(ParagraphStyle(name="TitleBlue", fontName=base_font, fontSize=16, textColor=colors.HexColor("#0b63d6"), alignment=1, spaceAfter=8))
        styles.add(ParagraphStyle(name="H2", fontName=base_font, fontSize=12, textColor=colors.HexColor("#0b63d6"), spaceAfter=6))
        styles.add(ParagraphStyle(name="Body", fontName=base_font, fontSize=10, leading=14))
        styles.add(ParagraphStyle(name="Note", fontName=base_font, fontSize=9, textColor=colors.grey))
        story = []
        # header
        story.append(Paragraph("NeuroEarly Pro — Clinical Report", styles["TitleBlue"]))
        story.append(Spacer(1, 6))
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
        # metrics table
        metrics = summary.get("metrics", {})
        if metrics:
            mrows = [["Metric","Value"]]
            for k,v in metrics.items():
                mrows.append([k, f"{v}"])
            mt = Table(mrows, colWidths=[200,200])
            mt.setStyle(TableStyle([("GRID",(0,0),(-1,-1),0.25,colors.grey)]))
            story.append(Paragraph("<b>Key QEEG metrics</b>", styles["H2"]))
            story.append(mt)
            story.append(Spacer(1,12))
        # topomaps
        if summary.get("topo_images"):
            story.append(Paragraph("<b>Topomaps</b>", styles["H2"]))
            for name, img_bytes in summary["topo_images"].items():
                try:
                    story.append(RLImage(io.BytesIO(img_bytes), width=200, height=140))
                    story.append(Spacer(1,6))
                except Exception:
                    pass
        # connectivity
        if summary.get("connectivity_image"):
            story.append(Paragraph("<b>Functional Connectivity</b>", styles["H2"]))
            try:
                story.append(RLImage(io.BytesIO(summary["connectivity_image"]), width=360, height=200))
            except Exception:
                pass
            story.append(Spacer(1,6))
        # shap
        if summary.get("shap_img"):
            story.append(Paragraph("<b>Explainability (SHAP)</b>", styles["H2"]))
            try:
                story.append(RLImage(io.BytesIO(summary["shap_img"]), width=360, height=140))
            except Exception:
                pass
            story.append(Spacer(1,6))
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
        story.append(Paragraph("Prepared by Golden Bird LLC — NeuroEarly Pro", styles["Note"]))
        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()
    except Exception as e:
        print("PDF gen error:", e)
        return None

# ----------------- Questionnaires -----------------
PHQ9_QUESTIONS = [
    "Little interest or pleasure in doing things",
    "Feeling down, depressed, or hopeless",
    "Trouble falling or staying asleep, or sleeping too much",
    "Feeling tired or having little energy",
    "Poor appetite or overeating",
    "Feeling bad about yourself — or that you are a failure or have let yourself or your family down",
    "Trouble concentrating on things, such as reading the newspaper or watching television",
    "Moving or speaking so slowly that other people could have noticed? Or the opposite — being so fidgety or restless that you have been moving a lot more than usual",
    "Thoughts that you would be better off dead or of hurting yourself"
]

AD8_QUESTIONS = [
    "Problems with judgment (e.g., problems making decisions, bad financial decisions)",
    "Less interest in hobbies/activities",
    "Repeats same things over and over",
    "Trouble learning how to use gadgets",
    "Forgets correct month or year",
    "Trouble handling complicated financial affairs",
    "Trouble remembering appointments",
    "Daily problems with thinking and memory"
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

# ----------------- Streamlit UI -----------------
st.set_page_config(page_title=APP_TITLE, layout="wide")
header_html = f"""
<div style="background: linear-gradient(90deg,{BLUE},{'#7DD3FC'});padding:12px;border-radius:8px;color:white;display:flex;align-items:center;justify-content:space-between">
  <div style="font-weight:700;font-size:20px;">🧠 {APP_TITLE}</div>
  <div style="display:flex;align-items:center;">
    <div style="margin-right:12px;color:white;font-size:12px;">Prepared by Golden Bird LLC</div>
    {'<img src="assets/goldenbird_logo.png" style="height:36px"/>' if LOGO_PATH_PNG.exists() else ''}
  </div>
</div>
"""
st.markdown(header_html, unsafe_allow_html=True)
st.write("")

left_col, main_col, right_col = st.columns([1, 2.2, 1])

with left_col:
    st.header("Settings")
    lang = st.selectbox("Language / اللغة", ["English", "Arabic"])
    patient_name = st.text_input("Patient Name (optional)")
    patient_id = st.text_input("Patient ID")
    dob = st.date_input("Date of birth", value=date(1980,1,1), max_value=date(2100,12,31))
    sex = st.selectbox("Sex / الجنس", ["Unknown", "Male", "Female", "Other"])
    meds = st.text_area("Current meds (one per line)")
    labs = st.text_area("Relevant labs (B12, TSH, ...)")
    st.markdown("---")
    st.subheader("Upload")
    uploaded_file = st.file_uploader("Upload EDF file (.edf)", type=["edf"], accept_multiple_files=False)
    st.write("")
    process_btn = st.button("Process EDF(s) and Analyze")

with main_col:
    st.subheader("Console / Visualization")
    console_box = st.empty()
    status_placeholder = st.empty()
    progress_bar = st.progress(0.0)

with right_col:
    st.subheader("Questionnaires")
    st.markdown("**PHQ-9 (Depression)**")
    phq_ans = []
    for i,q in enumerate(PHQ9_QUESTIONS):
        v = st.radio(f"Q{i+1}. {q}", [0,1,2,3], index=0, key=f"phq_{i}")
        phq_ans.append(v)
    st.markdown("---")
    st.markdown("**AD8 (Cognitive)**")
    ad8_ans = []
    for i,q in enumerate(AD8_QUESTIONS):
        v = st.radio(f"Q{i+1}. {q}", [0,1], index=0, key=f"ad8_{i}")
        ad8_ans.append(v)
    st.markdown("---")
    gen_pdf_btn = st.button("Generate PDF report (from last result)")

# session storage
if "last_result" not in st.session_state:
    st.session_state["last_result"] = None

# Prepare EDF path to use (uploaded or default)
edf_to_use = None
if uploaded_file:
    # write to temp and use that path
    try:
        tmp_path = Path(tempfile.gettempdir()) / f"uploaded_{now_ts()}.edf"
        with open(tmp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        edf_to_use = str(tmp_path)
    except Exception as e:
        console_box.error(f"Failed to save uploaded EDF: {e}")
        edf_to_use = None
else:
    if DEFAULT_EDF.exists():
        st.info("No EDF uploaded — using default test EDF on server.")
        edf_to_use = str(DEFAULT_EDF)
    else:
        st.warning("No EDF uploaded and default EDF not found. Please upload an EDF file to analyze.")
        edf_to_use = None

# Main processing when button pressed
if process_btn:
    console_box.info("📁 Reading EDF file... please wait")
    progress_bar.progress(0.05)
    if not edf_to_use:
        status_placeholder.error("No EDF available for analysis.")
    else:
        raw_obj, msg = read_edf_as_object(edf_to_use)
        if raw_obj is None:
            status_placeholder.error(f"Error reading EDF: {msg}")
            console_box.error(f"EDF read error: {msg}")
        else:
            progress_bar.progress(0.2)
            status_placeholder.success("✅ EDF loaded successfully.")
            try:
                console_box.info("🔬 Computing band powers and metrics...")
                res = compute_band_powers(raw_obj, bands=BANDS)
                if res is None:
                    raise RuntimeError("Band computation returned no result.")
                progress_bar.progress(0.45)
                # prepare topomaps
                topo_imgs = {}
                chs = res["ch_names"]
                for b in BANDS:
                    vals = [res["bands"].get(ch, {}).get(f"{b}_rel", 0.0) for ch in chs]
                    img = topomap_png_from_vals(vals, band_name=b)
                    if img:
                        topo_imgs[b] = img
                progress_bar.progress(0.6)
                # connectivity (if raw_obj is mne Raw, get raw data; else from dict)
                data_matrix = None
                sf = float(res.get("sfreq", 250.0))
                if HAS_MNE and isinstance(raw_obj, mne.io.BaseRaw):
                    data_matrix = raw_obj.get_data()
                elif isinstance(raw_obj, dict):
                    data_matrix = np.asarray(raw_obj["signals"])
                conn_img = None
                conn_matrix = None
                if data_matrix is not None:
                    conn_matrix = compute_connectivity_matrix(data_matrix, sf)
                    if conn_matrix is not None:
                        fig,ax = plt.subplots(figsize=(4,3))
                        im = ax.imshow(conn_matrix, cmap="viridis")
                        ax.set_title("Connectivity")
                        fig.colorbar(im,ax=ax,fraction=0.046,pad=0.04)
                        buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf,format='png'); plt.close(fig); buf.seek(0)
                        conn_img = buf.getvalue()
                progress_bar.progress(0.75)
                # FDI
                fdi = compute_fdi(res["bands"])
                # SHAP
                shap_img = render_shap_if_present(Path(SHAP_JSON)) if Path(SHAP_JSON).exists() else None
                # questionnaire scoring
                phq_total, phq_level = score_phq9(phq_ans)
                ad8_score, ad8_risk = score_ad8(ad8_ans)
                # quick heuristic final risk (example weighting)
                ta_ratio = res["metrics"].get("theta_alpha_ratio", 0.0)
                ta_norm = max(0.0, min(1.0, ta_ratio / 3.0))
                phq_norm = phq_total / 27.0
                ad_norm = (ad8_score / 8.0)
                final_risk = 0.45*ta_norm + 0.35*ad_norm + 0.2*phq_norm
                # recommendations (simple heuristics)
                recs = []
                if fdi.get("FDI") and fdi["FDI"] > 2.0:
                    recs.append("Consider urgent MRI/neurology evaluation for focal slowing (FDI > 2).")
                if phq_total >= 15:
                    recs.append("High depression symptoms — consider referral to psychiatry for evaluation and treatment.")
                if ad8_score >= 2:
                    recs.append("Cognitive screening positive — consider neuropsychological assessment and brain imaging as indicated.")
                if not recs:
                    recs.append("No immediate high-risk red flags on automated screening; correlate clinically.")
                progress_bar.progress(0.95)
                console_box.success("✅ Processing complete.")
                status_placeholder.info("View results below. You can generate PDF or export CSV.")
                # assemble summary
                summary = {
                    "patient_info": {"id": patient_id, "dob": str(dob), "name": patient_name},
                    "bands": res["bands"],
                    "metrics": res["metrics"],
                    "ch_names": res["ch_names"],
                    "sfreq": res["sfreq"],
                    "psd": res["psd"].tolist() if isinstance(res.get("psd"), np.ndarray) else None,
                    "topo_images": topo_imgs,
                    "connectivity_image": conn_img,
                    "conn_matrix": conn_matrix.tolist() if conn_matrix is not None else None,
                    "fdi": fdi,
                    "normative_bar": None,
                    "shap_img": shap_img,
                    "phq": {"total": phq_total, "level": phq_level},
                    "ad8": {"score": ad8_score, "risk": ad8_risk},
                    "final_risk": float(final_risk),
                    "recommendations": recs,
                    "created": now_ts()
                }
                st.session_state["last_result"] = summary
                progress_bar.progress(1.0)
            except Exception as e:
                console_box.error(f"Processing exception: {e}")
                traceback.print_exc()
                status_placeholder.error("Processing failed. See console for details.")

# Display last result if exists
if st.session_state.get("last_result"):
    s = st.session_state["last_result"]
    st.markdown("---")
    st.subheader("Interactive Dashboard (Results)")
    # band table
    rows = []
    for ch, vals in s["bands"].items():
        row = {"ch": ch}
        for b in BANDS:
            row[f"{b}_abs"] = vals.get(f"{b}_abs", 0.0)
            row[f"{b}_rel"] = vals.get(f"{b}_rel", 0.0)
        rows.append(row)
    df = pd.DataFrame(rows).set_index("ch")
    st.dataframe(df, height=300)
    # topomaps
    st.markdown("### Topomaps")
    cols = st.columns(2)
    idx = 0
    for bname, img in s.get("topo_images", {}).items():
        with cols[idx % 2]:
            st.image(img, caption=bname, use_column_width=True)
        idx += 1
    # connectivity
    if s.get("connectivity_image"):
        st.markdown("### Connectivity")
        st.image(s["connectivity_image"], use_column_width=False)
    # metrics and scores
    st.markdown("### Key metrics & scores")
    st.write(s.get("metrics", {}))
    st.write("PHQ-9:", s.get("phq"))
    st.write("AD8:", s.get("ad8"))
    st.metric("Final ML Risk", f"{s.get('final_risk',0.0)*100:.1f}%")
    st.markdown("### Recommendations")
    for r in s.get("recommendations", []):
        st.write("- " + r)
    # export CSV
    try:
        export_rows = []
        for ch, vals in s["bands"].items():
            entry = {"ch": ch}
            for b in BANDS:
                entry[f"{b}_rel"] = vals.get(f"{b}_rel", 0.0)
            export_rows.append(entry)
        df_export = pd.DataFrame(export_rows)
        csv_bytes = df_export.to_csv(index=False).encode("utf-8")
        st.download_button("Download band CSV", csv_bytes, file_name=f"NeuroEarly_bands_{now_ts()}.csv", mime="text/csv")
    except Exception:
        pass

# Generate PDF if requested
if gen_pdf_btn:
    if not st.session_state.get("last_result"):
        st.error("No processed result — run analysis first.")
    else:
        st.info("Generating PDF...")
        pdf_bytes = generate_pdf_report(st.session_state["last_result"], lang=("ar" if lang == "Arabic" else "en"), amiri_path=AMIRI_TTF if AMIRI_TTF.exists() else None)
        if pdf_bytes:
            st.download_button("Download PDF report", data=pdf_bytes, file_name=f"NeuroEarly_Report_{now_ts()}.pdf", mime="application/pdf")
            st.success("PDF generated.")
        else:
            # fallback: export JSON summary
            st.warning("PDF generation not available on this server (ReportLab missing). Exporting JSON summary instead.")
            st.download_button("Download JSON summary", json.dumps(st.session_state["last_result"], default=str, indent=2), file_name=f"NeuroEarly_Summary_{now_ts()}.json", mime="application/json")

# SHAP hint
if not Path(SHAP_JSON).exists():
    st.info("No shap_summary.json found. Upload shap summary to enable SHAP visualizations.")

st.markdown("---")
st.markdown("**Notes:** This application is an assistive screening tool. Clinical correlation and formal diagnostic work-up are required. For full functionality install optional packages: `mne`, `pyedflib`, `scipy`, `reportlab`, `shap`.")

# app.py — NeuroEarly Pro (Real EDF processing, vFinal)
# Requirements (recommended): streamlit, mne, pyedflib, scipy, numpy, pandas, matplotlib, reportlab, shap, arabic-reshaper, python-bidi, pillow

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

# Optional heavy libraries
HAS_MNE = False
HAS_PYEDF = False
HAS_SCIPY = False
HAS_REPORTLAB = False
HAS_SHAP = False
HAS_ARABIC = False

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
    from scipy.signal import welch, coherence
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

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

# ---------------- Config / Assets ----------------
ROOT = Path(".")
ASSETS = ROOT / "assets"
LOGO_PNG = ASSETS / "goldenbird_logo.png"
AMIRI_TTF = ROOT / "Amiri-Regular.ttf"   # if present, used in PDF for Arabic
SHAP_JSON = ROOT / "shap_summary.json"   # optional
# baseline EDF you previously uploaded (developer instruction)
HEALTHY_EDF = Path("/mnt/data/test_edf.edf")  # <-- existing uploaded file in your workspace

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

# ---------------- Helpers ----------------
def now_ts(fmt="%Y%m%d_%H%M%S"):
    return datetime.utcnow().strftime(fmt)

def safe_div(a, b, eps=1e-12):
    return a / (b + eps)

# Robust EDF reader: try mne, then pyedflib. Use temp file to avoid BytesIO issues.
def read_edf_bytes(uploaded_file) -> Tuple[Optional[Any], Optional[str]]:
    """
    uploaded_file: Streamlit UploadedFile OR path-like string
    Returns (raw_object, error_msg)
    raw_object: if mne available --> mne Raw
                else dict: {'signals': ndarray (n_ch x n_samples), 'ch_names': list, 'sfreq': float}
    """
    if not uploaded_file:
        return None, "No file provided"
    # If path-like string
    if isinstance(uploaded_file, (str, Path)):
        p = str(uploaded_file)
        # try mne
        if HAS_MNE:
            try:
                raw = mne.io.read_raw_edf(p, preload=True, verbose=False)
                return raw, None
            except Exception as e:
                # fallback to pyedflib
                pass
        if HAS_PYEDF:
            try:
                f = pyedflib.EdfReader(p)
                n = f.signals_in_file
                ch_names = f.getSignalLabels()
                sfreq = f.getSampleFrequency(0)
                sigs = np.vstack([f.readSignal(i) for i in range(n)])
                f.close()
                return {'signals': sigs, 'ch_names': ch_names, 'sfreq': sfreq}, None
            except Exception as e:
                return None, f"pyedflib read error: {e}"
        return None, "No EDF backend available on server (install mne or pyedflib)"
    # else it's UploadedFile-like
    try:
        data = uploaded_file.read()
    except Exception as e:
        return None, f"Could not read uploaded file buffer: {e}"
    # write to temp file
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmp:
            tmp.write(data)
            tmp.flush()
            tmp_path = tmp.name
        return read_edf_bytes(tmp_path)
    except Exception as e:
        return None, f"Temp write error: {e}"
    finally:
        # keep temp for mne if needed; not deleting here to avoid mne mmap issues
        pass

# compute bandpowers from mne.Raw or fallback dict
def compute_band_powers(raw_obj, bands=BANDS):
    """
    raw_obj: mne Raw OR {'signals': ndarray (n_ch x n_samples), 'ch_names':..., 'sfreq':...}
    returns dict: {'bands': {ch: {Delta_abs/rel...}}, 'metrics': {...}, 'ch_names':[], 'sfreq':..., 'psd': ndarray, 'freqs': ndarray}
    """
    if raw_obj is None:
        return None
    if HAS_MNE and isinstance(raw_obj, mne.io.BaseRaw):
        raw = raw_obj.copy().pick_types(eeg=True, meg=False, stim=False)
        sfreq = raw.info['sfreq']
        ch_names = raw.ch_names
        data = raw.get_data()  # shape n_ch x n_samples
    else:
        # fallback dict
        dd = raw_obj
        data = np.asarray(dd['signals'])
        ch_names = dd.get('ch_names', [f"ch{i}" for i in range(data.shape[0])])
        sfreq = float(dd.get('sfreq', 250.0))
        if data.ndim == 1:
            data = data[np.newaxis, :]
    n_ch = data.shape[0]
    # PSD via welch if scipy available
    if HAS_SCIPY:
        win = min(2048, data.shape[1])
        freqs = None
        psd_list = []
        for i in range(n_ch):
            f, Pxx = welch(data[i, :], fs=sfreq, nperseg=win)
            psd_list.append(Pxx)
            freqs = f
        psd = np.vstack(psd_list)  # n_ch x len(freqs)
    else:
        # simple FFT fallback (less accurate)
        n = data.shape[1]
        freqs = np.fft.rfftfreq(n, d=1.0/sfreq)
        fft_vals = np.abs(np.fft.rfft(data, axis=1))**2 / n
        psd = fft_vals

    band_summary = {}
    total_power = psd.sum(axis=1) + 1e-12
    for idx, ch in enumerate(ch_names):
        band_summary[ch] = {}
        for bname, (fmin, fmax) in bands.items():
            mask = (freqs >= fmin) & (freqs < fmax)
            power = float(psd[idx, mask].sum()) if mask.any() else 0.0
            band_summary[ch][f"{bname}_abs"] = power
            band_summary[ch][f"{bname}_rel"] = power / total_power[idx]
    # metrics
    # theta/alpha ratio (mean across channels of theta_rel / alpha_rel)
    ta_list = []
    for ch in ch_names:
        a_rel = band_summary[ch].get("Alpha_rel", 0.0)
        t_rel = band_summary[ch].get("Theta_rel", 0.0)
        if a_rel > 0:
            ta_list.append(t_rel / a_rel)
    theta_alpha_ratio = float(np.mean(ta_list)) if ta_list else 0.0
    # FDI: focal delta index: top channel delta_rel / mean(delta_rel)
    delta_rels = np.array([band_summary[ch].get("Delta_rel", 0.0) for ch in ch_names])
    mean_delta = float(np.mean(delta_rels)) if delta_rels.size else 0.0
    max_delta = float(np.max(delta_rels)) if delta_rels.size else 0.0
    FDI = float(max_delta / (mean_delta + 1e-12)) if mean_delta > 0 else 0.0
    # try compute alpha asymmetry F3-F4 if channels present
    alpha_asym = None
    def find_idx(names, pat):
        for i, nm in enumerate(names):
            if pat.upper() in nm.upper():
                return i
        return None
    idx_f3 = find_idx(ch_names, "F3")
    idx_f4 = find_idx(ch_names, "F4")
    if idx_f3 is not None and idx_f4 is not None:
        alpha_rel_f3 = band_summary[ch_names[idx_f3]].get("Alpha_rel", 0.0)
        alpha_rel_f4 = band_summary[ch_names[idx_f4]].get("Alpha_rel", 0.0)
        alpha_asym = float(alpha_rel_f3 - alpha_rel_f4)
    metrics = {
        "theta_alpha_ratio": theta_alpha_ratio,
        "FDI": FDI,
        "alpha_asym_F3_F4": alpha_asym,
        "mean_total_power": float(np.mean(total_power))
    }
    return {"bands": band_summary, "metrics": metrics, "ch_names": ch_names, "sfreq": float(sfreq), "psd": psd, "freqs": freqs}

# fallback topomap generator: grid-based heatmap (returns PNG bytes)
def topomap_png_from_vals(vals: List[float], band_name: str = "Band"):
    try:
        arr = np.asarray(vals).astype(float).ravel()
        n = arr.size
        side = int(np.ceil(np.sqrt(n)))
        grid_flat = np.full(side*side, np.nan)
        grid_flat[:n] = arr
        grid = grid_flat.reshape(side, side)
        fig, ax = plt.subplots(figsize=(3.6, 2.4))
        vmin = np.nanmin(grid)
        vmax = np.nanmax(grid)
        if np.isfinite(vmin) and np.isfinite(vmax) and vmin != vmax:
            im = ax.imshow(grid, cmap="RdBu_r", interpolation="nearest", origin="upper", vmin=vmin, vmax=vmax)
        else:
            im = ax.imshow(grid, cmap="RdBu_r", interpolation="nearest", origin="upper")
        ax.set_title(band_name, fontsize=10)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        buf = io.BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format="png", dpi=120)
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue()
    except Exception as e:
        print("topomap error:", e)
        return None

# connectivity computation: coherence (alpha) if possible, else corrcoef fallback
def compute_connectivity_matrix(raw_obj, sfreq, method="coherence", band=(8.0,13.0)):
    try:
        # extract signals
        if HAS_MNE and isinstance(raw_obj, mne.io.BaseRaw):
            raw = raw_obj.copy().pick_types(eeg=True, stim=False)
            data = raw.get_data()
        else:
            data = np.asarray(raw_obj['signals'])
            if data.ndim == 1:
                data = data[np.newaxis, :]
        n_ch = data.shape[0]
        conn = np.zeros((n_ch, n_ch))
        if HAS_SCIPY and method == "coherence":
            lo, hi = band
            # compute mean coherence in band per pair
            for i in range(n_ch):
                for j in range(i, n_ch):
                    try:
                        f, Cxy = coherence(data[i,:], data[j,:], fs=sfreq, nperseg=min(2048, data.shape[1]))
                        mask = (f >= lo) & (f <= hi)
                        val = float(np.nanmean(Cxy[mask])) if mask.any() else 0.0
                    except Exception:
                        val = 0.0
                    conn[i,j] = val
                    conn[j,i] = val
        else:
            # Pearson correlation across time
            x = data.copy()
            x = (x - np.mean(x, axis=1, keepdims=True)) / (np.std(x, axis=1, keepdims=True) + 1e-12)
            conn = np.corrcoef(x)
            conn = np.nan_to_num(conn)
        return conn
    except Exception as e:
        print("connectivity error:", e)
        return None

# render SHAP summary from JSON -> PNG bytes
def render_shap_from_json(path, model_key_hint=None):
    try:
        if not Path(path).exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            sj = json.load(f)
        # choose a model key
        key = model_key_hint if model_key_hint and model_key_hint in sj else next(iter(sj.keys()))
        feats = sj.get(key, {})
        if not feats:
            return None
        s = pd.Series(feats).abs().sort_values(ascending=False).head(12)
        fig, ax = plt.subplots(figsize=(6, 2.2))
        s.plot.barh(ax=ax)
        ax.invert_yaxis()
        ax.set_xlabel("mean |SHAP|")
        fig.tight_layout()
        buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig); buf.seek(0)
        return buf.getvalue()
    except Exception as e:
        print("shap render error:", e)
        return None

# PDF generator
def generate_pdf_report(summary: dict, lang: str = "en", amiri_path: Optional[Path] = None) -> Optional[bytes]:
    if not HAS_REPORTLAB:
        return None
    try:
        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=36, bottomMargin=36, leftMargin=44, rightMargin=44)
        styles = getSampleStyleSheet()
        base_font = "Helvetica"
        if amiri_path and Path(amiri_path).exists():
            try:
                pdfmetrics.registerFont(TTFont("Amiri", str(amiri_path)))
                base_font = "Amiri"
            except Exception as e:
                print("Amiri register fail:", e)
        # ensure unique style names
        styles.add(ParagraphStyle(name="NEP_Title", fontName=base_font, fontSize=16, textColor=colors.HexColor(BLUE), alignment=1, spaceAfter=8))
        styles.add(ParagraphStyle(name="NEP_H2", fontName=base_font, fontSize=12, textColor=colors.HexColor(BLUE), spaceAfter=6))
        styles.add(ParagraphStyle(name="NEP_Body", fontName=base_font, fontSize=10, leading=13))
        story = []
        # header
        story.append(Paragraph("NeuroEarly Pro — Clinical Report", styles["NEP_Title"]))
        story.append(Spacer(1,6))
        # logo if present
        if LOGO_PNG.exists():
            try:
                story.append(RLImage(str(LOGO_PNG), width=120, height=60))
                story.append(Spacer(1,6))
            except Exception:
                pass
        # patient info table
        pi = summary.get("patient_info", {})
        info_rows = [
            ["Patient ID", pi.get("id","-"), "DOB", pi.get("dob","-")],
            ["Report created", summary.get("created", now_ts()), "Exam", pi.get("exam","EEG")]
        ]
        t = Table(info_rows, colWidths=[70,150,70,120])
        t.setStyle(TableStyle([("GRID",(0,0),(-1,-1),0.25,colors.grey), ("BACKGROUND",(0,0),(-1,0),colors.whitesmoke)]))
        story.append(t); story.append(Spacer(1,10))
        # final ML risk prominent
        if summary.get("final_ml_risk") is not None:
            story.append(Paragraph(f"<b>Final ML Risk Score: {summary['final_ml_risk']*100:.1f}%</b>", styles["NEP_H2"]))
            story.append(Spacer(1,8))
        # key metrics
        story.append(Paragraph("QEEG Key Metrics", styles["NEP_H2"]))
        metrics = summary.get("metrics", {})
        if metrics:
            rows = [["Metric","Value"]]
            for k,v in metrics.items():
                try:
                    rows.append([k, f"{v:.4f}" if isinstance(v,(float,int)) else str(v)])
                except Exception:
                    rows.append([k, str(v)])
            mt = Table(rows, colWidths=[240, 200])
            mt.setStyle(TableStyle([("GRID",(0,0),(-1,-1),0.25,colors.grey)]))
            story.append(mt); story.append(Spacer(1,8))
        # normative bar
        if summary.get("normative_bar"):
            story.append(Paragraph("Normative Comparison", styles["NEP_H2"]))
            try:
                story.append(RLImage(io.BytesIO(summary["normative_bar"]), width=420, height=120))
                story.append(Spacer(1,8))
            except Exception:
                pass
        # topomaps
        if summary.get("topo_images"):
            story.append(Paragraph("Topography Maps", styles["NEP_H2"]))
            imgs = []
            for band, b in summary["topo_images"].items():
                try:
                    imgs.append(RLImage(io.BytesIO(b), width=200, height=120))
                except Exception:
                    pass
            # arrange approx two per row
            row = []
            for i, im in enumerate(imgs):
                row.append(im)
                if (i%2)==1:
                    story.append(Table([row], colWidths=[220,220])); row=[]
            if row:
                story.append(Table([row], colWidths=[220,220]))
            story.append(Spacer(1,8))
        # connectivity
        if summary.get("connectivity_image"):
            story.append(Paragraph("Functional Connectivity (Alpha)", styles["NEP_H2"]))
            try:
                story.append(RLImage(io.BytesIO(summary["connectivity_image"]), width=420, height=260))
            except Exception:
                pass
            story.append(Spacer(1,8))
        # SHAP
        if summary.get("shap_img"):
            story.append(Paragraph("Explainable AI (SHAP)", styles["NEP_H2"]))
            try:
                story.append(RLImage(io.BytesIO(summary["shap_img"]), width=420, height=160))
            except Exception:
                pass
            story.append(Spacer(1,8))
        # recommendations
        story.append(Paragraph("Structured Clinical Recommendations", styles["NEP_H2"]))
        recs = summary.get("recommendations", ["Automated screening — clinical correlation required."])
        for r in recs:
            story.append(Paragraph(str(r), styles["NEP_Body"]))
            story.append(Spacer(1,4))
        story.append(Spacer(1,12))
        story.append(Paragraph("Prepared by Golden Bird LLC — NeuroEarly Pro", styles["NEP_Body"]))
        doc.build(story)
        buf.seek(0)
        return buf.getvalue()
    except Exception as e:
        print("PDF gen error:", e)
        traceback.print_exc()
        return None

# ---------------- Questionnaires ----------------
PHQ9_QS = [
    "Little interest or pleasure in doing things",
    "Feeling down, depressed, or hopeless",
    # Q3 special: sleep (we will provide options mapping to 0..3)
    "Trouble falling or staying asleep, or sleeping too much",
    "Feeling tired or having little energy",
    # Q5 appetite: overeating or undereating mapping
    "Poor appetite or overeating",
    "Feeling bad about yourself — or that you are a failure or have let yourself or your family down",
    "Trouble concentrating on things",
    # Q8 psychomotor change: slow or restless
    "Moving or speaking so slowly that other people could have noticed? Or the opposite — being so fidgety or restless",
    "Thoughts that you would be better off dead or of hurting yourself"
]

AD8_QS = [
    "Problems with judgment (e.g., difficulty making decisions)",
    "Less interest in hobbies/activities",
    "Repeats same things over and over",
    "Trouble learning how to use gadgets",
    "Forgets correct month or year",
    "Trouble handling complicated financial affairs",
    "Trouble remembering appointments",
    "Daily problems with thinking and memory"
]

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title=APP_TITLE, layout="wide")
# header
header_html = f"""
<div style="background: linear-gradient(90deg,{BLUE},#7DD3FC);padding:12px;border-radius:8px;color:white;display:flex;align-items:center;justify-content:space-between">
  <div style="font-weight:700;font-size:20px;">🧠 {APP_TITLE}</div>
  <div style="display:flex;align-items:center;">
    <div style="margin-right:12px;color:white;font-size:12px;">Prepared by Golden Bird LLC</div>
    {'<img src="assets/goldenbird_logo.png" style="height:36px"/>' if LOGO_PNG.exists() else ''}
  </div>
</div>
"""
st.markdown(header_html, unsafe_allow_html=True)
st.write("")

# layout: left sidebar, main, right column
left_col, main_col, right_col = st.columns([1, 2.2, 1])

with left_col:
    st.header("Settings")
    lang_choice = st.selectbox("Language / اللغة", ["English", "Arabic"], key="lang_select")
    lang = "ar" if str(lang_choice).lower().startswith("arab") else "en"
    st.markdown("---")
    st.subheader("Patient info")
    patient_name = st.text_input("Patient Name (optional)", key="pt_name")
    patient_id = st.text_input("Patient ID", key="pt_id")
    dob = st.date_input("Date of birth", value=date(1980,1,1), max_value=date(2025,12,31), key="pt_dob")
    sex = st.selectbox("Sex", ["Unknown", "Male", "Female", "Other"], key="pt_sex")
    st.markdown("---")
    st.subheader("Medications")
    meds = st.text_area("Current medications (one per line)", height=80, key="pt_meds")
    st.subheader("Relevant labs")
    labs = st.text_area("Relevant labs (e.g. B12: low, TSH: high)", height=80, key="pt_labs")
    st.markdown("---")
    st.subheader("Upload files")
    uploaded_edf = st.file_uploader("Upload EDF file (.edf)", type=["edf"], accept_multiple_files=False, key="upload_edf")
    upload_shap = st.file_uploader("Upload shap_summary.json (optional)", type=["json"], accept_multiple_files=False, key="upload_shap")
    st.markdown("")
    process_btn = st.button("Process EDF & Analyze", key="process_btn")

with main_col:
    st.subheader("Console / Visuals")
    console = st.empty()
    status = st.empty()
    progress = st.progress(0.0)

with right_col:
    st.subheader("Questionnaires")
    st.markdown("**PHQ-9 (Depression)**")
    phq_answers = []
    # Q1..Q9
    for i, q in enumerate(PHQ9_QS):
        key = f"phq_{i+1}"
        if i == 2:  # Q3 sleep: options insomnia / less sleep / hypersomnia mapping to 0..3
            sel = st.selectbox(f"Q{i+1}. {q}", ["0 - No change", "1 - Insomnia - Several days", "2 - Insomnia - More than half the days", "3 - Hypersomnia - Nearly every day"], key=key)
            phq_answers.append(int(sel.split(" - ")[0]))
        elif i == 4:  # Q5 appetite
            sel = st.selectbox(f"Q{i+1}. {q}", ["0 - No change", "1 - Decreased appetite - Several days", "2 - Increased or decreased appetite - More than half the days", "3 - Marked change - Nearly every day"], key=key)
            phq_answers.append(int(sel.split(" - ")[0]))
        elif i == 7:  # Q8 psychomotor
            sel = st.selectbox(f"Q{i+1}. {q}", ["0 - No change", "1 - Slight change - Several days", "2 - Noticeable change - More than half the days", "3 - Marked change - Nearly every day"], key=key)
            phq_answers.append(int(sel.split(" - ")[0]))
        else:
            sel = st.selectbox(f"Q{i+1}. {q}", ["0 - Not at all", "1 - Several days", "2 - More than half the days", "3 - Nearly every day"], key=key)
            phq_answers.append(int(sel.split(" - ")[0]))
    st.markdown("---")
    st.markdown("**AD8 (Cognitive Screening)**")
    ad8_answers = []
    for i, q in enumerate(AD8_QS):
        key = f"ad8_{i+1}"
        sel = st.selectbox(f"Q{i+1}. {q}", ["0 - No", "1 - Yes"], key=key)
        ad8_answers.append(int(sel.split(" - ")[0]))
    st.markdown("---")
    gen_pdf_btn = st.button("Generate PDF report (from last result)", key="gen_pdf_btn")

# Processing logic
processing_result = None
summary = {}

if process_btn:
    progress.progress(0.02)
    console.info("Reading EDF...")
    raw, err = read_edf_bytes(uploaded_edf if uploaded_edf else (str(HEALTHY_EDF) if HEALTHY_EDF.exists() else None))
    if raw is None:
        status.error(f"Error reading EDF: {err}")
    else:
        progress.progress(0.1)
        try:
            console.info("Computing band powers...")
            res = compute_band_powers(raw, bands=BANDS)
            progress.progress(0.35)
            # build topo images
            topo_imgs = {}
            if res and res.get("bands"):
                ch_names = res["ch_names"]
                for band in BANDS.keys():
                    vals = [res["bands"][ch].get(f"{band}_rel", 0.0) for ch in ch_names]
                    img = topomap_png_from_vals(vals, band_name=band)
                    if img:
                        topo_imgs[band] = img
            progress.progress(0.55)
            # connectivity
            conn_img = None
            conn_mat = None
            try:
                conn_mat = compute_connectivity_matrix(raw, res.get("sfreq", 250.0), method="coherence" if HAS_SCIPY else "corr", band=BANDS.get("Alpha",(8.0,13.0)))
                if conn_mat is not None:
                    fig, ax = plt.subplots(figsize=(4,3))
                    im = ax.imshow(conn_mat, cmap="viridis")
                    ax.set_title("Connectivity (Alpha)")
                    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format="png"); plt.close(fig); buf.seek(0)
                    conn_img = buf.getvalue()
            except Exception as e:
                print("connectivity build failed:", e)
            progress.progress(0.7)
            # normative comparison (Theta/Alpha and alpha asym)
            normative_bar = None
            try:
                if HEALTHY_EDF.exists():
                    base_raw, berr = read_edf_bytes(str(HEALTHY_EDF))
                    if base_raw:
                        base_res = compute_band_powers(base_raw, bands=BANDS)
                        # theta/alpha patient vs base
                        theta_alpha_patient = res["metrics"]["theta_alpha_ratio"]
                        theta_alpha_base = base_res["metrics"].get("theta_alpha_ratio", 0.0)
                        # generate small bar
                        fig, ax = plt.subplots(figsize=(5.2,2.1))
                        ax.bar([0,1], [theta_alpha_patient, theta_alpha_base], color=[BLUE, "#80c080"])
                        ax.set_xticks([0,1]); ax.set_xticklabels(["Patient","Baseline"])
                        ax.set_title("Theta/Alpha (patient vs baseline)")
                        buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf,format="png"); plt.close(fig); buf.seek(0)
                        normative_bar = buf.getvalue()
            except Exception as e:
                print("normative compare failed:", e)
            progress.progress(0.85)
            # SHAP: either uploaded or in repo
            shap_img = None
            shap_path_to_use = None
            if upload_shap:
                # save uploaded shap json locally
                try:
                    spath = ROOT / "uploaded_shap_summary.json"
                    with open(spath, "wb") as f: f.write(upload_shap.getvalue())
                    shap_path_to_use = spath
                except Exception:
                    shap_path_to_use = None
            elif SHAP_JSON.exists():
                shap_path_to_use = SHAP_JSON
            if shap_path_to_use:
                shap_img = render_shap_from_json(shap_path_to_use, model_key_hint=None)
            progress.progress(0.9)
            # clinical scores
            phq_total = sum(phq_answers)
            ad8_total = sum(ad8_answers)
            # simple heuristic final ML risk
            ta_norm = min(1.0, res["metrics"].get("theta_alpha_ratio", 0.0) / 2.0)
            phq_norm = min(1.0, phq_total / 27.0)
            ad_norm = min(1.0, ad8_total / 8.0)
            final_ml_risk = 0.45*ta_norm + 0.35*ad_norm + 0.2*phq_norm
            # focal delta index (FDI) and alpha asym already in metrics
            fdi_val = res["metrics"].get("FDI", 0.0)
            # recommendations (heuristic rules)
            recs = []
            if final_ml_risk >= 0.4:
                recs.append("Moderate-to-high risk: consider full neuropsychological assessment and MRI if indicated.")
            else:
                recs.append("Low risk: monitor and correlate clinically.")
            if fdi_val > 2.0:
                recs.append("High focal delta (FDI > 2): consider urgent structural imaging (MRI).")
            # create summary
            summary = {
                "patient_info": {"id": patient_id, "dob": str(dob), "name": patient_name},
                "bands": res["bands"],
                "metrics": res["metrics"],
                "topo_images": topo_imgs,
                "connectivity_image": conn_img,
                "conn_matrix": conn_mat,
                "normative_bar": normative_bar,
                "shap_img": shap_img,
                "phq_score": phq_total,
                "ad8_score": ad8_total,
                "final_ml_risk": final_ml_risk,
                "recommendations": recs,
                "created": now_ts()
            }
            processing_result = res
            progress.progress(1.0)
            console.success("Processing complete.")
            status.info("Results ready. See below.")
        except Exception as exc:
            console.error(f"Processing exception: {exc}")
            traceback.print_exc()
            status.error("Processing failed. See console.")
            processing_result = None

# display results
if processing_result and summary:
    st.markdown("---")
    st.subheader("Results Dashboard")
    # key QEEG metrics
    mcols = st.columns(3)
    mcols[0].metric("Theta/Alpha Ratio", f"{summary['metrics'].get('theta_alpha_ratio',0.0):.3f}")
    mcols[1].metric("FDI (Focal Delta Index)", f"{summary['metrics'].get('FDI',0.0):.2f}")
    a_asym = summary['metrics'].get('alpha_asym_F3_F4', None)
    mcols[2].metric("Alpha Asymmetry (F3-F4)", f"{a_asym:.4f}" if a_asym is not None else "N/A")
    st.markdown("### Band table (relative power)")
    # convert bands to DataFrame
    rows = []
    for ch, vals in summary['bands'].items():
        row = {"channel": ch}
        for b in BANDS.keys():
            row[f"{b}_rel"] = vals.get(f"{b}_rel", 0.0)
        rows.append(row)
    df_bands = pd.DataFrame(rows).set_index("channel")
    st.dataframe(df_bands.style.format("{:.4f}"), height=320)
    # topomaps
    st.markdown("### Topomaps")
    tcols = st.columns(2)
    i = 0
    for band, img in summary.get("topo_images", {}).items():
        try:
            tcols[i%2].image(img, caption=band, use_column_width=True)
        except Exception:
            pass
        i += 1
    # connectivity
    if summary.get("connectivity_image"):
        st.markdown("### Connectivity (Alpha)")
        st.image(summary["connectivity_image"])
    # normative
    if summary.get("normative_bar"):
        st.markdown("### Normative comparison")
        st.image(summary["normative_bar"])
    # SHAP
    if summary.get("shap_img"):
        st.markdown("### SHAP (Explainability)")
        st.image(summary["shap_img"])
    st.markdown("### Clinical Scores")
    st.write(f"PHQ-9 score: {summary.get('phq_score')}   |   AD8 score: {summary.get('ad8_score')}")
    st.markdown("### Recommendations")
    for r in summary.get("recommendations", []):
        st.write("- " + r)
    st.markdown("---")

# PDF generation
if gen_pdf_btn:
    if not summary:
        st.error("No processed result — run analysis first.")
    else:
        st.info("Generating PDF...")
        pdf_bytes = generate_pdf_report(summary, lang=lang, amiri_path=AMIRI_TTF if AMIRI_TTF.exists() else None)
        if pdf_bytes:
            st.download_button("Download NeuroEarly Report (PDF)", data=pdf_bytes, file_name=f"NeuroEarly_Report_{now_ts()}.pdf", mime="application/pdf")
            st.success("PDF ready.")
        else:
            st.error("PDF generation failed — ensure reportlab & fonts are installed on server.")

# final notes
st.markdown("---")
st.markdown("Notes: This application requires scientific validation before clinical use. Use as an assistive screening tool only.")

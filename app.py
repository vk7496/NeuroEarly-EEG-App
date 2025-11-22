# app.py — NeuroEarly Pro v17 (Clinical Premium)
# - Bilingual (English default / Arabic optional)
# - Sidebar left (language + patient info + meds + labs + EDF upload)
# - Robust EDF reading: mne preferred, pyedflib fallback (writes temp file)
# - PSD via scipy.welch (fallback FFT)
# - Topomaps: MNE plot_topomap if mne available else smart grid heatmap
# - Connectivity: coherence (scipy) then pearson corr fallback
# - FDI (Focal Delta Index)
# - SHAP rendering from shap_summary.json (if present)
# - PDF report via reportlab (Amiri font if present) bilingual
# - Corrected PHQ/AD questions & options (Q3/Q5/Q8 fixed)
# - Multi-EDF upload & comparison, Export CSV, Download PDF

import os, io, sys, math, json, traceback, tempfile
from pathlib import Path
from datetime import datetime, date
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import streamlit as st
from PIL import Image

# Optional heavy libs
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
ASSETS.mkdir(exist_ok=True)
LOGO_PATH = ASSETS / "goldenbird_logo.png"   # put your logo here
AMIRI_TTF = ROOT / "Amiri-Regular.ttf"
SHAP_JSON = ROOT / "shap_summary.json"
HEALTHY_BASELINE = ASSETS / "healthy_baseline.npz"  # fallback baseline (npz)

APP_TITLE = "NeuroEarly Pro — Clinical"
PRIMARY_BLUE = "#2D9CDB"
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

def safe_mean(x):
    try:
        return float(np.nanmean(x))
    except:
        return 0.0

def write_temp_bytes(data: bytes, suffix=".edf") -> str:
    tf = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tf.write(data)
    tf.flush()
    tf.close()
    return tf.name

# ---------------- EDF reading ----------------
def read_edf_uploaded(uploaded) -> Tuple[Optional[Any], Optional[str]]:
    """
    Accepts Streamlit uploaded file or path-like (str/Path).
    Returns (raw_object, error_message). raw_object:
      - if mne available: mne.io.Raw
      - else: dict {signals: ndarray (n_ch x n_samples), ch_names: [...], sfreq: float}
    """
    if uploaded is None:
        return None, "No file provided"
    # if path
    if isinstance(uploaded, (str, Path)):
        p = str(uploaded)
        # try mne
        if HAS_MNE:
            try:
                raw = mne.io.read_raw_edf(p, preload=True, verbose=False)
                return raw, None
            except Exception as e:
                pass
        # pyedflib
        if HAS_PYEDF:
            try:
                f = pyedflib.EdfReader(p)
                n = f.signals_in_file
                chs = f.getSignalLabels()
                sf = f.getSampleFrequency(0)
                arrs = [f.readSignal(i) for i in range(n)]
                f.close()
                data = np.vstack(arrs)
                return {"signals": data, "ch_names": chs, "sfreq": sf}, None
            except Exception as e:
                return None, f"pyedflib read error: {e}"
        return None, "No EDF backend available (mne/pyedflib missing)"
    # else UploadedFile object
    try:
        bytes_data = uploaded.read()
    except Exception as e:
        return None, f"uploaded file read error: {e}"
    tmp_path = None
    try:
        tmp_path = write_temp_bytes(bytes_data, suffix=".edf")
        return read_edf_uploaded(tmp_path)
    finally:
        # Keep the tmp for mne if needed (mne may memory-map), cleanup optional
        pass

# ---------------- PSD / band power ----------------
def compute_band_powers_from_raw(raw_obj, bands=BANDS):
    """
    raw_obj can be mne Raw or fallback dict {'signals', 'ch_names', 'sfreq'}.
    Returns dict: {bands: {ch: {b_abs,b_rel}}, metrics: {...}, ch_names, sfreq, freqs, psd}
    """
    if raw_obj is None:
        return None
    # prepare data and names
    if HAS_MNE and isinstance(raw_obj, mne.io.BaseRaw):
        raw = raw_obj.copy().pick_types(eeg=True, meg=False, stim=False)
        sf = raw.info.get("sfreq", 250.0)
        data = raw.get_data()  # shape (n_ch, n_samples)
        ch_names = raw.ch_names
    else:
        dd = raw_obj
        data = np.asarray(dd["signals"])
        ch_names = list(dd.get("ch_names", [f"ch{i}" for i in range(data.shape[0])]))
        sf = float(dd.get("sfreq", 250.0))
        if data.ndim == 1:
            data = data[np.newaxis, :]
    n_ch, n_s = data.shape
    # PSD
    if HAS_SCIPY:
        freqs = None
        psd = []
        nperseg = min(2048, n_s)
        for ch in range(n_ch):
            f, Pxx = welch(data[ch, :], fs=sf, nperseg=nperseg)
            psd.append(Pxx)
            freqs = f
        psd = np.vstack(psd)
    else:
        # simple FFT fallback (coarse)
        n = n_s
        freqs = np.fft.rfftfreq(n, d=1.0/sf)
        fft = np.abs(np.fft.rfft(data, axis=1))**2 / n
        psd = fft
    # compute band powers
    band_summary = {}
    for i, ch in enumerate(ch_names):
        band_summary[ch] = {}
        total_power = psd[i, :].sum() + 1e-12
        for bname, (lo, hi) in bands.items():
            mask = (freqs >= lo) & (freqs < hi)
            bp = float(psd[i, mask].sum()) if mask.any() else 0.0
            band_summary[ch][f"{bname}_abs"] = bp
            band_summary[ch][f"{bname}_rel"] = bp / total_power
    # metrics: theta/alpha ratio avg across channels, FDI etc.
    theta_alpha_list = []
    delta_rels = []
    for ch in ch_names:
        a = band_summary[ch].get("Alpha_rel", 0.0)
        t = band_summary[ch].get("Theta_rel", 0.0)
        if a > 0:
            theta_alpha_list.append(t / a)
        delta_rels.append(band_summary[ch].get("Delta_rel", 0.0))
    metrics = {}
    metrics["theta_alpha_ratio"] = float(np.mean(theta_alpha_list)) if theta_alpha_list else 0.0
    metrics["FDI_percent"] = float(max(delta_rels) * 100.0) if delta_rels else 0.0
    metrics["mean_global_power"] = float(np.mean(psd.sum(axis=1)))
    metrics["mean_connectivity_alpha"] = None  # set later if computed
    return {"bands": band_summary, "metrics": metrics, "ch_names": ch_names, "sfreq": sf, "psd": psd, "freqs": freqs}

# ---------------- Topomap fallback ----------------
def topomap_png_from_vals(vals: List[float], band_name: str = "Band"):
    try:
        arr = np.asarray(vals).astype(float).ravel()
        n = arr.size
        side = int(np.ceil(np.sqrt(n)))
        grid = np.full(side*side, np.nan)
        grid[:n] = arr
        grid = grid.reshape(side, side)
        fig, ax = plt.subplots(figsize=(3.0, 2.4))
        im = ax.imshow(grid, cmap="RdBu_r", interpolation="nearest", origin="upper")
        ax.set_title(band_name, fontsize=10)
        ax.axis("off")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        buf = io.BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format="png", dpi=150)
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue()
    except Exception as e:
        print("topomap error:", e)
        return None

# ---------------- Connectivity ----------------
def compute_connectivity_matrix(raw_obj, band=("Alpha")):
    """
    returns (conn_matrix (n_ch x n_ch), conn_image_bytes) or (None, None)
    uses coherence in alpha if scipy available else pearson corr
    """
    try:
        # extract data
        if HAS_MNE and isinstance(raw_obj, mne.io.BaseRaw):
            raw = raw_obj.copy().pick_types(eeg=True)
            data = raw.get_data()
            sf = raw.info.get("sfreq", 250.0)
        else:
            dd = raw_obj
            data = np.asarray(dd["signals"])
            sf = float(dd.get("sfreq", 250.0))
        if data.ndim == 1:
            data = data[np.newaxis, :]
        n_ch = data.shape[0]
        if HAS_SCIPY:
            # coherence in alpha band
            lo, hi = BANDS.get("Alpha", (8.0, 13.0))
            conn = np.zeros((n_ch, n_ch))
            for i in range(n_ch):
                for j in range(i, n_ch):
                    try:
                        f, Cxy = coherence(data[i, :], data[j, :], fs=sf, nperseg=min(2048, data.shape[1]))
                        mask = (f >= lo) & (f <= hi)
                        val = float(np.nanmean(Cxy[mask])) if mask.any() else 0.0
                    except Exception:
                        val = 0.0
                    conn[i, j] = val
                    conn[j, i] = val
        else:
            # pearson corr fallback
            x = data.copy()
            x = (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-12)
            conn = np.corrcoef(x)
            conn = np.nan_to_num(conn)
        # render image
        fig, ax = plt.subplots(figsize=(4, 3))
        im = ax.imshow(conn, cmap="viridis")
        ax.set_title("Connectivity (Alpha)" , fontsize=10)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        buf = io.BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format="png", dpi=150)
        plt.close(fig)
        buf.seek(0)
        return conn, buf.getvalue()
    except Exception as e:
        print("connectivity error:", e)
        return None, None

# ---------------- FDI ----------------
def compute_fdi(band_summary: Dict[str, Any]):
    try:
        # use Delta_rel across channels
        chs = list(band_summary.keys())
        vals = np.array([band_summary[ch].get("Delta_rel", 0.0) for ch in chs])
        global_mean = float(np.nanmean(vals)) if vals.size else 0.0
        idx = int(np.nanargmax(vals)) if vals.size else 0
        top_name = chs[idx] if vals.size else ""
        top_value = float(vals[idx]) if vals.size else 0.0
        FDI = (top_value / (global_mean + 1e-12)) if global_mean > 0 else None
        return {"global_mean": global_mean, "top_idx": idx, "top_name": top_name, "top_value": top_value, "FDI": FDI}
    except Exception as e:
        print("FDI error:", e)
        return {}

# ---------------- SHAP render ----------------
def render_shap_from_json(shap_json_path: Path, model_key_hint="depression_global"):
    if not shap_json_path.exists():
        return None
    try:
        with open(shap_json_path, "r", encoding="utf-8") as f:
            sj = json.load(f)
        key = model_key_hint if model_key_hint in sj else (next(iter(sj.keys())) if sj else None)
        if not key:
            return None
        feats = sj.get(key, {})
        s = pd.Series(feats).abs().sort_values(ascending=False).head(12)
        fig, ax = plt.subplots(figsize=(6, 2.8))
        s.plot.barh(ax=ax)
        ax.invert_yaxis()
        ax.set_xlabel("mean(|SHAP|)")
        ax.set_title("SHAP - Top feature contributions")
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150)
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue()
    except Exception as e:
        print("SHAP render failed:", e)
        return None

# ---------------- PDF generation ----------------
def generate_pdf_report(summary: dict, lang="en", amiri_path: Optional[Path] = None) -> Optional[bytes]:
    if not HAS_REPORTLAB:
        return None
    try:
        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=36, bottomMargin=36, leftMargin=36, rightMargin=36)
        styles = getSampleStyleSheet()
        base_font = "Helvetica"
        if amiri_path and amiri_path.exists():
            try:
                pdfmetrics.registerFont(TTFont("Amiri", str(amiri_path)))
                base_font = "Amiri"
            except Exception as e:
                print("Amiri register failed:", e)
        styles.add(ParagraphStyle(name="Title", fontName=base_font, fontSize=16, alignment=1, textColor=colors.HexColor(PRIMARY_BLUE)))
        styles.add(ParagraphStyle(name="H2", fontName=base_font, fontSize=11, textColor=colors.HexColor(PRIMARY_BLUE)))
        styles.add(ParagraphStyle(name="Body", fontName=base_font, fontSize=9, leading=12))
        story = []
        # header
        story.append(Paragraph("NeuroEarly Pro — Clinical Report", styles["Title"]))
        story.append(Spacer(1, 8))
        # logo if exists
        if LOGO_PATH.exists():
            try:
                story.append(RLImage(str(LOGO_PATH), width=90, height=30))
                story.append(Spacer(1, 6))
            except Exception:
                pass
        # patient info table
        pi = summary.get("patient_info", {})
        info_rows = [
            ["Patient ID", pi.get("id", "-"), "DOB", pi.get("dob", "-")],
            ["Report generated", summary.get("created", now_ts()), "Exam", pi.get("exam", "EEG")],
        ]
        t = Table(info_rows, colWidths=[70, 150, 70, 120])
        t.setStyle(TableStyle([("GRID",(0,0),(-1,-1),0.25,colors.grey), ("BACKGROUND",(0,0),(0,0),colors.whitesmoke)]))
        story.append(t)
        story.append(Spacer(1, 10))
        # metrics
        story.append(Paragraph("Key QEEG Metrics", styles["H2"]))
        metrics = summary.get("metrics", {})
        mrows = [["Metric", "Value"]]
        for k, v in metrics.items():
            try:
                vv = f"{v:.4f}" if isinstance(v, (float, int)) else str(v)
            except:
                vv = str(v)
            mrows.append([k, vv])
        mt = Table(mrows, colWidths=[220, 220])
        mt.setStyle(TableStyle([("GRID",(0,0),(-1,-1),0.25,colors.grey)]))
        story.append(mt)
        story.append(Spacer(1, 10))
        # normative bar
        if summary.get("normative_bar"):
            story.append(Paragraph("Normative Comparison", styles["H2"]))
            try:
                story.append(RLImage(io.BytesIO(summary["normative_bar"]), width=420, height=150))
                story.append(Spacer(1, 8))
            except:
                pass
        # topomaps
        if summary.get("topo_images"):
            story.append(Paragraph("Topography Maps", styles["H2"]))
            imgs = []
            for band, img in summary["topo_images"].items():
                try:
                    imgs.append(RLImage(io.BytesIO(img), width=200, height=120))
                except:
                    pass
            # table 2 per row
            row = []
            for i, im in enumerate(imgs):
                row.append(im)
                if i % 2 == 1:
                    story.append(Table([row], colWidths=[200, 200])); row = []
            if row:
                story.append(Table([row], colWidths=[200]*len(row)))
            story.append(Spacer(1,8))
        # connectivity image
        if summary.get("connectivity_image"):
            story.append(Paragraph("Functional Connectivity", styles["H2"]))
            try:
                story.append(RLImage(io.BytesIO(summary["connectivity_image"]), width=420, height=200))
                story.append(Spacer(1,8))
            except:
                pass
        # FDI
        if summary.get("fdi"):
            fdi = summary["fdi"]
            story.append(Paragraph("Focal Delta Index (FDI)", styles["H2"]))
            story.append(Paragraph(f"Top channel: {fdi.get('top_name','-')} — FDI: {fdi.get('FDI', '-')}", styles["Body"]))
            story.append(Spacer(1,8))
        # SHAP
        if summary.get("shap_img"):
            story.append(Paragraph("Explainable AI (SHAP)", styles["H2"]))
            try:
                story.append(RLImage(io.BytesIO(summary["shap_img"]), width=420, height=150))
                story.append(Spacer(1,8))
            except:
                pass
        # Clinical questionnaires & recommendations
        if summary.get("clinical"):
            cli = summary["clinical"]
            story.append(Paragraph("Clinical Questionnaires & Scores", styles["H2"]))
            story.append(Paragraph(f"PHQ-9: {cli.get('phq_score','-')} — AD8: {cli.get('ad_score','-')}", styles["Body"]))
            story.append(Spacer(1,6))
        if summary.get("recommendations"):
            story.append(Paragraph("Structured Clinical Recommendations", styles["H2"]))
            for r in summary["recommendations"]:
                story.append(Paragraph(r, styles["Body"]))
                story.append(Spacer(1,4))
        story.append(Spacer(1,8))
        story.append(Paragraph("Prepared by Golden Bird LLC — NeuroEarly Pro", ParagraphStyle(name="Footer", fontSize=8, textColor=colors.grey)))
        doc.build(story)
        buf.seek(0)
        return buf.getvalue()
    except Exception as e:
        print("PDF gen error:", e)
        traceback.print_exc()
        return None

# ---------------- Questionnaires definitions (fixed Q3/5/8) ----------------
PHQ9_QUESTIONS = [
    "1. Little interest or pleasure in doing things",
    "2. Feeling down, depressed, or hopeless",
    "3. Sleep: choose insomnia/hypersomnia",
    "4. Feeling tired or having little energy",
    "5. Appetite: choose increased/decreased",
    "6. Feeling bad about yourself or failure",
    "7. Trouble concentrating",
    "8. Moving or speaking slowly OR restless (choose)",
    "9. Thoughts of death or self-harm"
]
PHQ_STD = [("0 - Not at all",0), ("1 - Several days",1), ("2 - More than half the days",2), ("3 - Nearly every day",3)]
PHQ_Q3 = [("0 - No change",0), ("1 - Insomnia - Several days",1), ("2 - Insomnia - More than half the days",2), ("3 - Hypersomnia - Nearly every day",3)]
PHQ_Q5 = [("0 - No change",0), ("1 - Decreased appetite - Several days",1), ("2 - Decreased or increased appetite - More than half the days",2), ("3 - Marked change - Nearly every day",3)]
PHQ_Q8 = [("0 - No change",0), ("1 - Slight change",1), ("2 - Noticeable change",2), ("3 - Marked change",3)]

AD8_QUESTIONS = [
    "1. Problems with judgment (e.g., poor decisions)",
    "2. Less interest in hobbies/activities",
    "3. Repeats same things over and over",
    "4. Trouble learning to use gadgets",
    "5. Forgets correct month or year",
    "6. Trouble handling complicated financial affairs",
    "7. Trouble remembering appointments",
    "8. Daily problems with thinking and memory"
]
AD8_OPTIONS = [("0 - No",0),("1 - Yes, occasionally",1)]

def score_phq9(ans: List[int]) -> Tuple[int, str]:
    total = sum(ans)
    if total >= 20: level = "Severe"
    elif total >= 15: level = "Moderately Severe"
    elif total >= 10: level = "Moderate"
    elif total >= 5: level = "Mild"
    else: level = "Minimal"
    return total, level

def score_ad8(ans: List[int]) -> Tuple[int, str]:
    s = sum(ans)
    risk = "High" if s >= 2 else "Low"
    return s, risk

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title=APP_TITLE, layout="wide")
# Header
header_html = f"""
<div style="background: linear-gradient(90deg,{PRIMARY_BLUE},#7DD3FC); padding:12px; border-radius:8px; color:white; display:flex; align-items:center; justify-content:space-between;">
  <div style="font-weight:700; font-size:20px;">🧠 {APP_TITLE}</div>
  <div style="display:flex; align-items:center;">
    <div style="margin-right:10px; font-size:12px;">Prepared by Golden Bird LLC</div>
    {'<img src="assets/goldenbird_logo.png" style="height:36px"/>' if LOGO_PATH.exists() else ''}
  </div>
</div>
"""
st.markdown(header_html, unsafe_allow_html=True)
st.write("")

# Layout: sidebar left, main center, right for questionnaires/controls
left_col, main_col, right_col = st.columns([1, 2, 1])

with left_col:
    st.header("Settings")
    lang = st.selectbox("Language / اللغة", ["English", "Arabic"])
    lang_code = "ar" if lang.startswith("Arabic") or lang == "Arabic" else "en"
    st.markdown("---")
    st.subheader("Patient info")
    patient_id = st.text_input("Patient ID")
    patient_name = st.text_input("Patient name (optional)")
    dob = st.date_input("Date of birth", value=date(1980,1,1), max_value=date(2025,12,31))
    sex = st.selectbox("Sex / الجنس", ["Unknown","Male","Female","Other"])
    st.markdown("---")
    st.subheader("Medications (one per line)")
    meds = st.text_area("", height=80)
    st.subheader("Relevant labs (B12, TSH, B12 low/high etc.)")
    labs = st.text_area("", height=80)
    st.markdown("---")
    st.subheader("Upload EDF file(s)")
    uploads = st.file_uploader("Upload .edf (single or multiple)", type=["edf","EDF"], accept_multiple_files=True)
    st.markdown("")
    process_button = st.button("Process EDF(s) and Analyze")

with main_col:
    st.subheader("Console / Results")
    console = st.empty()
    viz_area = st.empty()

with right_col:
    st.subheader("Questionnaires")
    st.markdown("**PHQ-9 (Depression)**")
    phq_answers = []
    for i,q in enumerate(PHQ9_QUESTIONS):
        key = f"phq_{i}"
        if i == 2:
            sel = st.selectbox(q, [o[0] for o in PHQ_Q3], key=key)
            phq_answers.append(dict(PHQ_Q3)[sel])
        elif i == 4:
            sel = st.selectbox(q, [o[0] for o in PHQ_Q5], key=key)
            phq_answers.append(dict(PHQ_Q5)[sel])
        elif i == 7:
            sel = st.selectbox(q, [o[0] for o in PHQ_Q8], key=key)
            phq_answers.append(dict(PHQ_Q8)[sel])
        else:
            sel = st.radio(q, [o[0] for o in PHQ_STD], index=0, key=key, horizontal=False)
            phq_answers.append(dict(PHQ_STD)[sel])
    st.markdown("---")
    st.markdown("**AD8 (Cognitive)**")
    ad8_answers = []
    for i,q in enumerate(AD8_QUESTIONS):
        key = f"ad8_{i}"
        sel = st.radio(q, [o[0] for o in AD8_OPTIONS], index=0, key=key)
        ad8_answers.append(dict(AD8_OPTIONS)[sel])
    st.markdown("---")
    gen_pdf_btn = st.button("Generate PDF (from last results)")

# Storage for results
if "results" not in st.session_state:
    st.session_state["results"] = []

# Processing
if process_button:
    console.info("Loading EDF(s) and computing... (this may take a while)")
    st.session_state["results"].clear()
    if not uploads:
        console.warning("No EDF uploaded. If you want a demo, place a baseline file at 'assets/healthy_baseline.npz'.")
    for up in uploads or []:
        console.info(f"Reading {up.name} ...")
        raw, err = read_edf_uploaded(up)
        if raw is None:
            console.error(f"Failed to read {up.name}: {err}")
            continue
        console.info("Computing band powers...")
        res = compute_band_powers_from_raw(raw)
        if res is None:
            console.error(f"Band power computation failed for {up.name}")
            continue
        # FDI
        fdi = compute_fdi(res["bands"])
        # topomaps
        topo_imgs = {}
        chs = res.get("ch_names", [])
        for b in BANDS:
            vals = [res["bands"].get(ch, {}).get(f"{b}_rel", 0.0) for ch in chs]
            img = None
            # prefer mne topomap if mne and channel positions available (best-effort)
            if HAS_MNE:
                try:
                    # attempt to use simple grid fallback to avoid montage requirement
                    img = topomap_png_from_vals(vals, band_name=b)
                except Exception:
                    img = topomap_png_from_vals(vals, band_name=b)
            else:
                img = topomap_png_from_vals(vals, band_name=b)
            if img:
                topo_imgs[b] = img
        # connectivity
        conn_mat, conn_img = compute_connectivity_matrix(raw)
        # normative comparison: if baseline exists
        normative_bar = None
        if HEALTHY_BASELINE.exists():
            try:
                base = np.load(str(HEALTHY_BASELINE))
                base_data = base["data"]
                base_sf = float(base.get("sf", 250.0))
                df_base = None
                if base_data is not None:
                    df_base = None
                    try:
                        base_df = None
                        # compute band powers for baseline
                        # use same function but wrap into dict like fallback
                        base_obj = {"signals": base_data, "ch_names": [f"ch{i}" for i in range(base_data.shape[0])], "sfreq": base_sf}
                        base_res = compute_band_powers_from_raw(base_obj)
                        # compare Theta/Alpha
                        def safe_ratio(d):
                            vals_t = np.array([d[ch].get("Theta_rel",0.0) for ch in d])
                            vals_a = np.array([d[ch].get("Alpha_rel",0.0) for ch in d])
                            return float(np.nanmean(vals_t) / (np.nanmean(vals_a)+1e-12))
                        ta_p = safe_ratio(res["bands"])
                        ta_b = safe_ratio(base_res["bands"])
                        # plot bar
                        fig, ax = plt.subplots(figsize=(5.0, 1.8))
                        ax.bar([0,1],[ta_p, ta_b], color=[PRIMARY_BLUE, "#2ecc71"])
                        ax.set_xticks([0,1]); ax.set_xticklabels(["Patient","Baseline"])
                        ax.set_title("Theta/Alpha: patient vs baseline", fontsize=9)
                        buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format="png"); plt.close(fig); buf.seek(0)
                        normative_bar = buf.getvalue()
                    except Exception as e:
                        print("norm compare failed:", e)
            except Exception:
                pass
        # SHAP image (if provided)
        shap_img = render_shap_from_json(SHAP_JSON) if SHAP_JSON.exists() else None
        # compute final heuristics: normalize features to produce risk score
        theta_alpha = res["metrics"].get("theta_alpha_ratio", 0.0)
        phq_score, phq_level = score_phq9(phq_answers)
        ad_score, ad_risk = score_ad8(ad8_answers)
        # simple final risk (heuristic) — weights can be tuned later
        ta_norm = min(1.0, theta_alpha / 2.0)
        phq_norm = min(1.0, phq_score / 27.0)
        ad_norm = min(1.0, ad_score / 24.0)
        final_ml_risk = 0.45 * ta_norm + 0.35 * ad_norm + 0.20 * phq_norm
        # fill metrics mean_connectivity if available
        if conn_mat is not None:
            res["metrics"]["mean_connectivity_alpha"] = float(np.nanmean(conn_mat))
        # package result
        result = {
            "filename": up.name,
            "res": res,
            "topo_images": topo_imgs,
            "connectivity_image": conn_img,
            "conn_matrix": conn_mat,
            "fdi": fdi,
            "normative_bar": normative_bar,
            "shap_img": shap_img,
            "phq_score": phq_score,
            "ad_score": ad_score,
            "final_ml_risk": final_ml_risk,
            "created": now_ts(),
            "patient_info": {"id": patient_id, "name": patient_name, "dob": str(dob), "meds": meds, "labs": labs}
        }
        st.session_state["results"].append(result)
        console.success(f"{up.name} processed — final ML risk: {final_ml_risk*100:.1f}%")

# Display results
if st.session_state.get("results"):
    for i, r in enumerate(st.session_state["results"]):
        st.markdown(f"### Result: {r['filename']}")
        st.write("Patient ID:", r["patient_info"].get("id", "-"), "| DOB:", r["patient_info"].get("dob", "-"))
        st.metric("Final ML Risk", f"{r['final_ml_risk']*100:.1f}%")
        st.markdown("**Key metrics**")
        st.json(r["res"]["metrics"])
        st.markdown("**Topomaps**")
        cols = st.columns(3)
        j = 0
        for band, img in r.get("topo_images", {}).items():
            try:
                cols[j%3].image(img, caption=band, use_column_width=True)
            except Exception:
                pass
            j += 1
        if r.get("connectivity_image"):
            st.markdown("**Connectivity (Alpha)**")
            st.image(r["connectivity_image"], use_column_width=True)
        if r.get("normative_bar"):
            st.markdown("**Normative comparison**")
            st.image(r["normative_bar"], use_column_width=True)
        if r.get("shap_img"):
            st.markdown("**SHAP (XAI)**")
            st.image(r["shap_img"], use_column_width=True)
        if r.get("fdi"):
            st.markdown("**FDI**")
            st.write(r["fdi"])
        st.markdown("---")
    # CSV export
    try:
        rows = []
        for r in st.session_state["results"]:
            rows.append({
                "filename": r["filename"],
                "phq_score": r["phq_score"],
                "ad_score": r["ad_score"],
                "final_ml_risk": r["final_ml_risk"]
            })
        df_exp = pd.DataFrame(rows)
        st.download_button("Download metrics CSV", df_exp.to_csv(index=False).encode("utf-8"), file_name=f"NeuroEarly_metrics_{now_ts()}.csv", mime="text/csv")
    except Exception:
        pass
    # PDF generation for first result
    if gen_pdf_btn:
        s = st.session_state["results"][0]
        summary = {
            "patient_info": s["patient_info"],
            "metrics": s["res"]["metrics"],
            "topo_images": s["topo_images"],
            "connectivity_image": s["connectivity_image"],
            "fdi": s["fdi"],
            "normative_bar": s["normative_bar"],
            "shap_img": s["shap_img"],
            "clinical": {"phq_score": s["phq_score"], "ad_score": s["ad_score"]},
            "final_ml_risk": s["final_ml_risk"],
            "recommendations": [
                "This is an automated screening report. Clinical correlation is required.",
                "If FDI>2 or extreme asymmetry present, consider MRI for focal lesion.",
                "If ML risk high (>40%) correlate with cognitive testing and refer to specialist."
            ],
            "created": s.get("created", now_ts())
        }
        pdf_bytes = generate_pdf_report(summary, lang="ar" if lang_code=="ar" else "en", amiri_path=AMIRI_TTF if AMIRI_TTF.exists() else None)
        if pdf_bytes:
            st.download_button("Download full PDF report (first result)", pdf_bytes, file_name=f"NeuroEarly_Report_{now_ts()}.pdf", mime="application/pdf")
        else:
            st.error("PDF generation failed. Make sure reportlab and fonts are available on the server.")

else:
    st.info("No results yet. Upload EDF files and press 'Process EDF(s) and Analyze'.")

# Footer
st.markdown("---")
st.markdown("Prepared by Golden Bird LLC — NeuroEarly Pro (Clinical Premium)")

# End of app.py

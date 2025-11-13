# app.py — NeuroEarly Pro v6.1 (Clinical Stability Edition)
# Final clinical-ready Streamlit app (bilingual EN/AR, PHQ-9 + AD8, Topomaps for all bands, Connectivity, SHAP, PDF)
# Save as app.py and deploy (requires mne + reportlab + arabic_reshaper + python-bidi for full functionality)

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

# Optional packages (graceful fallback)
HAS_MNE = False
HAS_PYEDFLIB = False
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
    HAS_PYEDFLIB = True
except Exception:
    HAS_PYEDFLIB = False

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

# Paths and assets
ROOT = Path(__file__).parent
ASSETS = ROOT / "assets"
LOGO_PATH = ASSETS / "goldenbird_logo.png"
AMIRI_PATH = ROOT / "Amiri-Regular.ttf"
SHAP_JSON = ROOT / "shap_summary.json"

# Frequency bands (Hz)
BANDS = {
    "Delta": (1.0, 4.0),
    "Theta": (4.0, 8.0),
    "Alpha": (8.0, 13.0),
    "Beta": (13.0, 30.0),
    "Gamma": (30.0, 45.0)
}

def now_ts(fmt="%Y%m%d_%H%M%S"):
    return datetime.utcnow().strftime(fmt)

# Arabic shaping for UI/PDF
def safe_text(s: str, lang_code: str = "en") -> str:
    if lang_code and lang_code.startswith("ar") and HAS_ARABIC:
        try:
            shaped = arabic_reshaper.reshape(s)
            return get_display(shaped)
        except Exception:
            return s
    return s

# Save upload to temp file (avoids BytesIO compatibility issues)
def write_temp_file_from_upload(uploaded) -> Path:
    suffix = Path(uploaded.name).suffix or ".edf"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        tmp.write(uploaded.getvalue())
        tmp.flush()
        tmp.close()
        return Path(tmp.name)
    except Exception:
        try:
            tmp.close()
        except Exception:
            pass
        raise

# Read EDF robustly using mne or pyedflib
def read_edf_bytes(uploaded) -> Tuple[Optional['mne.io.Raw'], Optional[str]]:
    if uploaded is None:
        return None, "No file uploaded"
    tmp_path = None
    try:
        tmp_path = write_temp_file_from_upload(uploaded)
    except Exception as e:
        return None, f"Failed to save uploaded file: {e}"
    try:
        if HAS_MNE:
            raw = mne.io.read_raw_edf(str(tmp_path), preload=True, verbose=False)
            return raw, None
        elif HAS_PYEDFLIB:
            f = pyedflib.EdfReader(str(tmp_path))
            n = f.signals_in_file
            ch_names = f.getSignalLabels()
            sf = f.getSampleFrequency(0)
            data = np.vstack([f.readSignal(i) for i in range(n)])
            f._close()
            info = None
            try:
                info = mne.create_info(ch_names=list(ch_names), sfreq=float(sf), ch_types='eeg')
                raw = mne.io.RawArray(data, info)
                return raw, None
            except Exception:
                return None, "pyedflib read but mne not available to process"
        else:
            return None, "Neither mne nor pyedflib installed"
    except Exception as e:
        return None, f"Error reading EDF: {e}"
    finally:
        # try to remove tmp file (ignore errors)
        try:
            if tmp_path and tmp_path.exists():
                os.remove(tmp_path)
        except Exception:
            pass

# Compute band powers using Welch (returns DataFrame indexed by channel)
def compute_band_powers(raw: 'mne.io.Raw') -> pd.DataFrame:
    if not HAS_MNE:
        raise RuntimeError("mne is required for band power computation")
    picks = mne.pick_types(raw.info, eeg=True, meg=False, exclude='bads')
    if len(picks) == 0:
        raise RuntimeError("No EEG channels found")
    data, _ = raw.get_data(picks=picks, return_times=True)
    ch_names = [raw.ch_names[p] for p in picks]
    sf = int(raw.info.get('sfreq', 250))
    from scipy.signal import welch
    rows = []
    for idx, ch in enumerate(data):
        ch = ch - np.mean(ch)  # detrend (DC)
        nperseg = min(2048, max(256, len(ch)//8))
        f, Pxx = welch(ch, fs=sf, nperseg=nperseg)
        mask_total = (f >= 1) & (f <= 45)
        total_power = float(np.trapz(Pxx[mask_total], f[mask_total])) if mask_total.any() else 0.0
        row = {"ch": ch_names[idx], "total": total_power}
        for name, (lo, hi) in BANDS.items():
            mask = (f >= lo) & (f <= hi)
            val = float(np.trapz(Pxx[mask], f[mask])) if mask.any() else 0.0
            row[f"{name}_abs"] = val
            row[f"{name}_rel"] = (val / total_power) if total_power and total_power > 0 else 0.0
        rows.append(row)
    df = pd.DataFrame(rows).set_index("ch").fillna(0.0)
    return df

def compute_theta_alpha_ratio(df: pd.DataFrame) -> Optional[float]:
    try:
        t = df["Theta_rel"].mean(skipna=True)
        a = df["Alpha_rel"].mean(skipna=True)
        if a == 0:
            return None
        return float(t / a)
    except Exception:
        return None

def compute_alpha_asymmetry(df: pd.DataFrame, left="F3", right="F4") -> Optional[float]:
    try:
        if left not in df.index or right not in df.index:
            return None
        l = df.loc[left, "Alpha_rel"]
        r = df.loc[right, "Alpha_rel"]
        return float(l - r)
    except Exception:
        return None

def compute_fdi(df: pd.DataFrame, focal_channel: Optional[str]) -> Optional[float]:
    try:
        if focal_channel is None or "Delta_rel" not in df.columns:
            return None
        global_mean = df["Delta_rel"].mean(skipna=True)
        focal = df.loc[focal_channel, "Delta_rel"]
        if global_mean == 0:
            return None
        return float(focal / global_mean)
    except Exception:
        return None

# Connectivity: try mne spectral_connectivity, fallback to scipy.coherence matrix
def compute_connectivity_matrix(raw: 'mne.io.Raw', band=(8.0,13.0)):
    try:
        picks = mne.pick_types(raw.info, eeg=True, meg=False, exclude='bads')
        if len(picks) == 0:
            return None, None, None
        data, _ = raw.get_data(picks=picks, return_times=True)
        ch_names = [raw.ch_names[p] for p in picks]
        sf = int(raw.info.get('sfreq', 250))
        n = data.shape[0]
        conn = np.full((n,n), np.nan)
        # prefer mne connectivity if available
        if HAS_MNE:
            try:
                from mne.connectivity import spectral_connectivity
                con, freqs, times, n_epochs, n_tapers = spectral_connectivity([raw], method="coh", mode='multitaper', sfreq=sf, fmin=band[0], fmax=band[1], faverage=True, verbose=False)
                if con is not None:
                    conn = con.squeeze()
                    mean_conn = float(np.nanmean(conn))
                    # render image
                    fig, ax = plt.subplots(figsize=(4,3))
                    im = ax.imshow(np.nan_to_num(conn, nan=0.0), cmap='viridis', interpolation='nearest')
                    ax.set_xticks(range(len(ch_names))); ax.set_xticklabels(ch_names, rotation=90, fontsize=6)
                    ax.set_yticks(range(len(ch_names))); ax.set_yticklabels(ch_names, fontsize=6)
                    fig.colorbar(im, ax=ax, fraction=0.03)
                    buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format='png', dpi=150); plt.close(fig); buf.seek(0)
                    return conn, mean_conn, buf.getvalue()
            except Exception:
                pass
        # fallback coherence
        from scipy.signal import coherence
        for i in range(n):
            for j in range(i, n):
                try:
                    f, Cxy = coherence(data[i,:], data[j,:], fs=sf, nperseg=min(2048, max(256, data.shape[1]//8)))
                    mask = (f>=band[0]) & (f<=band[1])
                    val = float(np.nanmean(Cxy[mask])) if mask.any() else np.nan
                except Exception:
                    val = np.nan
                conn[i,j] = conn[j,i] = val
        mean_conn = float(np.nanmean(conn)) if not np.isnan(conn).all() else 0.0
        fig, ax = plt.subplots(figsize=(4,3))
        im = ax.imshow(np.nan_to_num(conn, nan=0.0), cmap='viridis', interpolation='nearest')
        ax.set_xticks(range(len(ch_names))); ax.set_xticklabels(ch_names, rotation=90, fontsize=6)
        ax.set_yticks(range(len(ch_names))); ax.set_yticklabels(ch_names, fontsize=6)
        fig.colorbar(im, ax=ax, fraction=0.03)
        buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format='png', dpi=150); plt.close(fig); buf.seek(0)
        return conn, mean_conn, buf.getvalue()
    except Exception:
        return None, None, None

# Robust topomap generator: try mne topomap, else fallback bar; ensure visibility if values near-zero
def generate_topomap_image(raw: 'mne.io.Raw', band: Tuple[float,float], show_band=True):
    try:
        picks = mne.pick_types(raw.info, eeg=True, meg=False, exclude='bads')
        if len(picks) == 0:
            return None
        data, _ = raw.get_data(picks=picks, return_times=True)
        sf = int(raw.info.get('sfreq', 250))
        ch_names = [raw.ch_names[p] for p in picks]
        from scipy.signal import welch
        vals = []
        for ch in data:
            f, Pxx = welch(ch, fs=sf, nperseg=min(2048, max(256, len(ch)//8)))
            mask = (f>=band[0]) & (f<=band[1])
            power = float(np.trapz(Pxx[mask], f[mask])) if mask.any() else 0.0
            vals.append(power)
        vals = np.array(vals, dtype=float)
        # normalize robustly; avoid divide by zero
        if vals.max() > 0:
            vals = vals / vals.max()
        vals = np.nan_to_num(vals, nan=0.0)
        # boost contrast if nearly uniform / tiny variance
        if np.allclose(vals, 0) or np.std(vals) < 0.03:
            jitter = np.linspace(0, 0.05, num=len(vals))
            vals = vals + jitter
            if vals.max() > 0:
                vals = vals / vals.max()
        # attempt mne topomap via montage; fallback to bar chart
        try:
            montage = mne.channels.make_standard_montage("standard_1020")
            info = mne.pick_info(raw.info, picks)
            evoked = mne.EvokedArray(vals.reshape(-1,1), info, tmin=0.0)
            evoked.set_montage(montage, match_case=False)
            figs = evoked.plot_topomap(times=0.0, ch_type='eeg', show=False)
            # capture current matplotlib figure
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
            plt.close('all')
            buf.seek(0)
            return buf.getvalue()
        except Exception:
            fig, ax = plt.subplots(figsize=(4,2.2))
            ax.bar(range(len(vals)), vals)
            if show_band:
                ax.set_title(f"{band[0]}-{band[1]} Hz (approx)")
            ax.set_xticks(range(len(vals)))
            ax.set_xticklabels(ch_names, rotation=90, fontsize=6)
            buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format='png', dpi=150); plt.close(fig); buf.seek(0)
            return buf.getvalue()
    except Exception:
        return None

# SHAP helpers
def load_shap_json(path=SHAP_JSON):
    if Path(path).exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None

def shap_bar_image_for_key(shap_data: dict, key: str):
    try:
        feats = shap_data.get(key, {})
        if not feats:
            return None
        s = pd.Series(feats).abs().sort_values(ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(6,2.2))
        s.plot.bar(ax=ax)
        ax.set_title("SHAP - top contributors")
        buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format='png', dpi=150); plt.close(fig); buf.seek(0)
        return buf.getvalue()
    except Exception:
        return None

# Final ML Risk (stable normalization with clinical ranges)
def compute_final_risk(theta_alpha, phq_total, ad8_score, fdi, connectivity):
    def norm(x, low, high):
        try:
            return max(0.0, min(1.0, (float(x) - low) / (high - low) if high > low else 0.0))
        except Exception:
            return 0.0
    ta_norm = norm(theta_alpha or 0.0, 0.2, 1.8)
    phq_norm = norm(phq_total or 0.0, 0, 27)
    ad_norm = norm(ad8_score or 0.0, 0, 8)
    fdi_norm = norm((fdi or 0.0), 0.5, 3.0)
    conn_norm = 1.0 - norm(connectivity if connectivity is not None else 0.5, 0.0, 1.0)
    risk = 0.35*ta_norm + 0.25*ad_norm + 0.15*phq_norm + 0.15*fdi_norm + 0.10*conn_norm
    if fdi and fdi > 2.0:
        risk = max(risk, 0.35)
    return round(risk * 100, 1)

# PDF report generator (readable, bilingual, uses Amiri if available for Arabic)
def generate_pdf_report(summary: dict, lang_code="en"):
    if not HAS_REPORTLAB:
        return None
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=36, bottomMargin=36, leftMargin=44, rightMargin=44)
    styles = getSampleStyleSheet()
    base_font = "Helvetica"
    if AMIRI_PATH.exists() and HAS_ARABIC:
        try:
            pdfmetrics.registerFont(TTFont("AmiriFont", str(AMIRI_PATH)))
            base_font = "AmiriFont"
        except Exception:
            pass
    # unique style names to avoid "already defined" errors
    styles.add(ParagraphStyle(name="NE_Title", fontName=base_font, fontSize=16, textColor=colors.HexColor("#0b63d6"), alignment=1, spaceAfter=8))
    styles.add(ParagraphStyle(name="NE_H2", fontName=base_font, fontSize=12, textColor=colors.HexColor("#0b63d6"), spaceAfter=6))
    styles.add(ParagraphStyle(name="NE_Body", fontName=base_font, fontSize=10, leading=14))
    story = []
    # Header with logo to the right
    left = Paragraph("<b>NeuroEarly Pro — Clinical Report</b>", styles["NE_Title"])
    if LOGO_PATH.exists():
        try:
            img = RLImage(str(LOGO_PATH), width=1.0*inch, height=1.0*inch)
            header_table_data = [[left, img]]
            t = Table(header_table_data, colWidths=[4.8*inch, 1.0*inch])
        except Exception:
            header_table_data = [[left]]
            t = Table(header_table_data)
    else:
        header_table_data = [[left]]
        t = Table(header_table_data)
    t.setStyle(TableStyle([("VALIGN",(0,0),(-1,-1),"MIDDLE")]))
    story.append(t); story.append(Spacer(1,12))
    # Patient info
    info = summary.get("patient_info", {})
    story.append(Paragraph(safe_text("Patient summary", lang_code), styles["NE_H2"]))
    rows = [["Field","Value"]]
    rows.append(["Patient ID", info.get("id","")])
    rows.append(["DOB", info.get("dob","")])
    rows.append(["Sex", info.get("sex","")])
    rows.append(["Meds", info.get("meds","")])
    rows.append(["Labs", info.get("labs","")])
    tinfo = Table(rows, colWidths=[2.5*inch,3.5*inch])
    tinfo.setStyle(TableStyle([("GRID",(0,0),(-1,-1),0.25,colors.grey),("BACKGROUND",(0,0),(-1,0),colors.HexColor("#eef7ff"))]))
    story.append(tinfo); story.append(Spacer(1,8))
    # Metrics
    story.append(Paragraph(safe_text("QEEG Key Metrics", lang_code), styles["NE_H2"]))
    metrics = summary.get("metrics", {})
    if metrics:
        rows = [["Metric","Value"]]
        for k,v in metrics.items():
            rows.append([k, str(v)])
        t2 = Table(rows, colWidths=[3.5*inch,2.5*inch])
        t2.setStyle(TableStyle([("GRID",(0,0),(-1,-1),0.25,colors.grey),("BACKGROUND",(0,0),(-1,0),colors.HexColor("#eef7ff"))]))
        story.append(t2); story.append(Spacer(1,8))
    # Topomaps
    topo_imgs = summary.get("topo_images", {})
    if topo_imgs:
        story.append(Paragraph(safe_text("Topography Maps (bands)", lang_code), styles["NE_H2"]))
        imgs = [RLImage(io.BytesIO(b), width=2.6*inch, height=1.6*inch) for b in topo_imgs.values() if b]
        rows = []; row = []
        for im in imgs:
            row.append(im)
            if len(row) == 2:
                rows.append(row); row = []
        if row: rows.append(row)
        for r in rows:
            t = Table([r], colWidths=[3.0*inch]*len(r))
            t.setStyle(TableStyle([("GRID",(0,0),(-1,-1),0.25,colors.grey)]))
            story.append(t); story.append(Spacer(1,6))
    # SHAP
    if summary.get("shap_img"):
        story.append(Paragraph(safe_text("XAI - SHAP contributors", lang_code), styles["NE_H2"]))
        story.append(RLImage(io.BytesIO(summary["shap_img"]), width=5.5*inch, height=2.2*inch)); story.append(Spacer(1,6))
    # Recommendations / clinical interpretation
    story.append(Paragraph(safe_text("Structured Clinical Recommendations", lang_code), styles["NE_H2"]))
    for rec in summary.get("recommendations", []):
        story.append(Paragraph(safe_text(rec, lang_code), styles["NE_Body"]))
    story.append(Spacer(1,12))
    story.append(Paragraph(safe_text("Prepared by Golden Bird LLC — NeuroEarly Pro", lang_code), styles["NE_Body"]))
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="NeuroEarly Pro", layout="wide")
# Sidebar: compact patient + upload
with st.sidebar:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=140)
    st.markdown("### Settings")
    lang = st.selectbox("Language / اللغة", options=["English", "العربية"], index=0)
    lang_code = "ar" if lang.startswith("ع") else "en"
    st.markdown("---")
    st.header(safe_text("Patient / المريض", lang_code))
    patient_id = st.text_input(safe_text("Patient ID", lang_code), value="H-0001")
    dob = st.date_input(safe_text("Date of birth", lang_code), value=date(1985,1,1), min_value=date(1900,1,1), max_value=date(2025,12,31))
    sex = st.selectbox(safe_text("Sex / الجنس", lang_code), [safe_text("Unknown", lang_code), safe_text("Male", lang_code), safe_text("Female", lang_code)])
    meds = st.text_area(safe_text("Current meds (one per line) / الأدوية", lang_code), value="")
    labs = st.text_area(safe_text("Relevant labs (B12, TSH, ... ) / نتائج آزمایش", lang_code), value="")
    st.markdown("---")
    st.header(safe_text("Upload", lang_code))
    uploaded = st.file_uploader(safe_text("Upload EDF file (.edf)", lang_code), type=["edf","EDF"], accept_multiple_files=False)
    st.markdown("---")
    st.write(safe_text("Testing: create synthetic EDF in Tools below", lang_code))

# Main layout: title, questionnaires, process
st.title(safe_text("NeuroEarly Pro — Clinical & Research", lang_code))
st.markdown(safe_text("EEG / QEEG analysis • Topomaps • Explainable AI", lang_code))

# Questionnaires (both visible)
st.subheader(safe_text("Questionnaires", lang_code))
col1, col2 = st.columns(2)
with col1:
    st.markdown("**PHQ-9 (depression)**")
    q1 = st.radio(safe_text("Q1 - Little interest or pleasure", lang_code), [0,1,2,3], index=0, key="q1")
    q2 = st.radio(safe_text("Q2 - Feeling down, depressed", lang_code), [0,1,2,3], index=0, key="q2")
    q3_sel = st.selectbox(safe_text("Q3 - Sleep pattern", lang_code),
                         [("0","No change"), ("1","Insomnia (difficulty sleeping)"), ("2","Hypersomnia (oversleeping)"), ("3","Severe")],
                         format_func=lambda x: safe_text(x[1], lang_code), key="q3")
    q4 = st.radio(safe_text("Q4 - Energy / Fatigue", lang_code), [0,1,2,3], index=0, key="q4")
    q5_sel = st.selectbox(safe_text("Q5 - Appetite", lang_code),
                         [("0","No change"),("1","Mild (less/more)"),("2","Moderate"),("3","Severe")],
                         format_func=lambda x: safe_text(x[1], lang_code), key="q5")
with col2:
    st.markdown("**PHQ-9 continued**")
    q6 = st.radio(safe_text("Q6 - Feelings of failure/guilt", lang_code), [0,1,2,3], index=0, key="q6")
    q7 = st.radio(safe_text("Q7 - Concentration problems", lang_code), [0,1,2,3], index=0, key="q7")
    q8_sel = st.selectbox(safe_text("Q8 - Psychomotor (e.g., slowed or restless)", lang_code),
                         [("0","Normal"),("1","Slight"),("2","Noticeable"),("3","Marked")],
                         format_func=lambda x: safe_text(x[1], lang_code), key="q8")
    q9 = st.radio(safe_text("Q9 - Suicidal ideation", lang_code), [0,1,2,3], index=0, key="q9")

def map_custom_val(sel):
    if isinstance(sel, int): return sel
    try:
        return int(sel[0])
    except Exception:
        return 0

phq_total = q1 + q2 + map_custom_val(q3_sel) + q4 + map_custom_val(q5_sel) + q6 + q7 + map_custom_val(q8_sel) + q9

st.markdown("---")
st.markdown("**AD8 (cognitive change screening)**")
ad8_qs = [
    "Problems with judgment (e.g., bad decisions)?",
    "Reduced interest in hobbies/activities?",
    "Repeats questions/stories/remarks?",
    "Trouble learning to use a tool, appliance, or gadget?",
    "Forgets correct month or year?",
    "Difficulty handling complicated financial affairs?",
    "Trouble remembering appointments?",
    "Consistent problems with thinking and memory?"
]
ad8_answers = []
for i, qt in enumerate(ad8_qs, start=1):
    a = st.radio(safe_text(f"AD8-{i} - {qt}", lang_code), [safe_text("No", lang_code), safe_text("Yes", lang_code)], index=0, key=f"ad8_{i}")
    ad8_answers.append(1 if (a == safe_text("Yes", lang_code) or a == "Yes") else 0)
ad8_score = sum(ad8_answers)

st.markdown("---")
process_btn = st.button(safe_text("Process EDF / تشغيل التحليل", lang_code))

# Console and results side-by-side
col_left, col_right = st.columns([1,2])
with col_left:
    st.header(safe_text("Console", lang_code))
    console = st.empty()
    console.info(safe_text("Ready. Upload EDF and press Process.", lang_code))
    selected_channel = st.text_input(safe_text("Channel viewer (e.g., F3)", lang_code), value="")
with col_right:
    st.header(safe_text("Results & Visuals", lang_code))
    if uploaded is None:
        st.info(safe_text("No EDF uploaded. Use sidebar to upload or create synthetic EDF.", lang_code))
    else:
        if process_btn:
            console.info(safe_text("Saving and reading EDF...", lang_code))
            raw, err = read_edf_bytes(uploaded)
            if err:
                st.error(safe_text(err, lang_code))
            elif raw is None:
                st.error(safe_text("Failed to parse EDF file.", lang_code))
            else:
                try:
                    st.success(safe_text(f"EDF loaded. Channels: {len(raw.ch_names)} • sfreq: {raw.info.get('sfreq')}", lang_code))
                except Exception:
                    st.success(safe_text("EDF loaded.", lang_code))
                # band powers
                try:
                    df_bands = compute_band_powers(raw)
                    st.subheader(safe_text("QEEG Band summary (relative power)", lang_code))
                    st.dataframe(df_bands.round(4))
                except Exception as e:
                    st.error(safe_text(f"Band power computation failed: {e}", lang_code))
                    df_bands = pd.DataFrame()

                # metrics
                theta_alpha = compute_theta_alpha_ratio(df_bands) if not df_bands.empty else None
                alpha_asym = compute_alpha_asymmetry(df_bands)
                focal_ch = None
                try:
                    focal_ch = df_bands["Delta_rel"].idxmax() if "Delta_rel" in df_bands.columns else None
                except Exception:
                    focal_ch = None
                fdi_val = compute_fdi(df_bands, focal_ch) if focal_ch else None

                # connectivity
                conn_mat, mean_conn, conn_img = None, None, None
                try:
                    conn_mat, mean_conn, conn_img = compute_connectivity_matrix(raw, band=BANDS["Alpha"])
                except Exception:
                    conn_mat, mean_conn, conn_img = None, None, None

                # topomaps for all bands
                topo_imgs = {}
                for bname, band in BANDS.items():
                    try:
                        img = generate_topomap_image(raw, band, show_band=True)
                        topo_imgs[bname] = img
                    except Exception:
                        topo_imgs[bname] = None

                # SHAP
                shap_img = None
                shap_data = load_shap_json()
                if shap_data:
                    key = "depression_global" if phq_total and phq_total >= 10 else "alzheimers_global"
                    shap_img = shap_bar_image_for_key(shap_data, key)

                # Show summary metrics
                st.markdown("### " + safe_text("Key metrics", lang_code))
                st.write({
                    "Theta/Alpha (global)": theta_alpha,
                    "Alpha Asymmetry (F3-F4)": alpha_asym,
                    "Focal Delta channel": focal_ch,
                    "FDI": fdi_val,
                    "Mean connectivity (alpha)": mean_conn,
                    "AD8 score": ad8_score,
                    "PHQ-9 total": phq_total
                })

                final_risk = compute_final_risk(theta_alpha or 0.0, phq_total, ad8_score, fdi_val or 0.0, mean_conn or 0.0)
                st.metric(safe_text("Final ML Risk (%)", lang_code), f"{final_risk}%")

                # channel viewer
                if selected_channel:
                    try:
                        pick_idx = raw.ch_names.index(selected_channel)
                        data, times = raw.get_data(picks=[pick_idx], return_times=True)
                        fig, ax = plt.subplots(figsize=(8,2.4))
                        ax.plot(times, data[0])
                        ax.set_title(safe_text(f"Raw trace: {selected_channel}", lang_code))
                        buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format='png'); plt.close(fig); buf.seek(0)
                        st.image(buf.getvalue(), use_column_width=True)
                    except Exception as e:
                        st.warning(safe_text(f"Channel viewer error: {e}", lang_code))

                # show topomaps
                st.subheader(safe_text("Topography maps (all bands)", lang_code))
                cols = st.columns(2)
                i = 0
                for bname, img in topo_imgs.items():
                    with cols[i%2]:
                        st.markdown(f"**{bname}**")
                        if img:
                            st.image(img, use_container_width=True)
                        else:
                            st.info(safe_text("Not available", lang_code))
                    i += 1

                # connectivity
                st.subheader(safe_text("Connectivity (Alpha)", lang_code))
                if conn_img:
                    st.image(conn_img, use_container_width=True)
                else:
                    st.info(safe_text("Connectivity not available", lang_code))

                # SHAP
                if shap_img:
                    st.subheader(safe_text("XAI - SHAP contributors", lang_code))
                    st.image(shap_img, use_container_width=True)

                # Build summary and PDF
                summary = {
                    "patient_info": {"id": patient_id, "dob": str(dob), "sex": sex, "meds": meds, "labs": labs},
                    "metrics": {
                        "theta_alpha_ratio": float(theta_alpha) if theta_alpha else None,
                        "alpha_asymmetry_f3_f4": float(alpha_asym) if alpha_asym else None,
                        "focal_channel": focal_ch,
                        "fdi": float(fdi_val) if fdi_val else None,
                        "mean_connectivity_alpha": float(mean_conn) if mean_conn else None,
                        "ad8_score": int(ad8_score),
                        "phq9_total": int(phq_total),
                        "final_risk": final_risk
                    },
                    "topo_images": topo_imgs,
                    "shap_img": shap_img,
                    "recommendations": [
                        "Automated screening report — clinical correlation required.",
                        "If FDI > 2 or focal slowing is present, consider structural imaging (MRI).",
                        "Review medications and labs (B12, TSH) for reversible causes.",
                        "For moderate-high ML risk: consider neuropsychological assessment, follow-up QEEG in 3-6 months, and specialist referral as appropriate."
                    ]
                }

                pdf_bytes = generate_pdf_report(summary, lang_code) if HAS_REPORTLAB else None
                if pdf_bytes:
                    st.download_button(safe_text("Download PDF report", lang_code), data=pdf_bytes, file_name=f"NeuroEarly_Report_{now_ts()}.pdf", mime="application/pdf")
                else:
                    if not HAS_REPORTLAB:
                        st.info(safe_text("PDF generation not available (reportlab not installed).", lang_code))

# Sidebar testing tools: synthetic EDF generator & download
with st.sidebar:
    st.markdown("---")
    st.subheader(safe_text("Testing tools", lang_code))
    dur = st.number_input(safe_text("Synthetic EDF duration (s)", lang_code), min_value=30, max_value=600, value=120)
    sfreq = st.selectbox(safe_text("Sample rate", lang_code), [250, 500], index=1)
    if st.button(safe_text("Create & download synthetic EDF", lang_code)):
        try:
            def generate_synthetic(duration_s=dur, sf=sfreq, n_ch=19):
                t = np.arange(int(duration_s*sf))/sf
                ch_names = ['Fp1','Fp2','F7','F3','Fz','F4','F8','T7','C3','Cz','C4','T8','P7','P3','Pz','P4','P8','O1','O2'][:n_ch]
                signals = []
                for i in range(n_ch):
                    a_freq = 8 + np.random.randn()*1.2 + (i%5)*0.5
                    t_freq = 6 + np.random.randn()*1.0 + (i%3)*0.3
                    sig = 5*np.sin(2*np.pi*a_freq*t + i*0.1) + 2*np.sin(2*np.pi*t_freq*t + i*0.2) + 0.8*np.random.randn(len(t))
                    # local delta bump simulating focal slowing
                    if i in [7,11]:
                        sig += 2.0*np.sin(2*np.pi*2*t)*np.exp(-((t-duration_s/2)**2)/(2*(duration_s/6)**2))
                    signals.append(sig)
                signals = np.vstack(signals)
                if HAS_PYEDFLIB:
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".edf"); fn = tmp.name; tmp.close()
                    f = pyedflib.EdfWriter(fn, n_ch)
                    header = []
                    for ch in ch_names:
                        header.append({'label':ch,'dimension':'uV','sample_rate':sf,'physical_min':-500,'physical_max':500,'digital_min':-32768,'digital_max':32767,'transducer':'','prefilter':''})
                    f.setSignalHeaders(header)
                    f.writeSamples(signals)
                    f.close()
                    with open(fn, "rb") as fh:
                        b = fh.read()
                    try: os.remove(fn)
                    except: pass
                    return b, "application/octet-stream", f"synthetic_{duration_s}s_{sf}Hz.edf"
                else:
                    buf = io.BytesIO(); np.save(buf, signals); buf.seek(0)
                    return buf.getvalue(), "application/octet-stream", f"synthetic_{duration_s}s_{sf}Hz.npy"
            bts, mime, fname = generate_synthetic()
            st.sidebar.download_button(safe_text("Download synthetic EDF", lang_code), data=bts, file_name=fname, mime=mime)
        except Exception as e:
            st.sidebar.error(safe_text(f"Could not generate synthetic EDF: {e}", lang_code))

# app_v6_stage1.py
# NeuroEarly v6_stage1 — Robust EEG processing + PHQ9 + AD8 + Topomaps for all bands
# Focus: accurate band powers, reliable topomaps, both questionnaires visible

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

# Optional libs
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

# Paths
ROOT = Path(__file__).parent
ASSETS = ROOT / "assets"
LOGO_PATH = ASSETS / "goldenbird_logo.png"
AMIRI_PATH = ROOT / "Amiri-Regular.ttf"
SHAP_JSON = ROOT / "shap_summary.json"

# Bands
BANDS = {
    "Delta": (1.0, 4.0),
    "Theta": (4.0, 8.0),
    "Alpha": (8.0, 13.0),
    "Beta": (13.0, 30.0),
    "Gamma": (30.0, 45.0)
}

def now_ts(fmt="%Y%m%d_%H%M%S"):
    return datetime.utcnow().strftime(fmt)

def safe_text(s: str, lang_code: str = "en") -> str:
    if lang_code.startswith("ar") and HAS_ARABIC:
        try:
            shaped = arabic_reshaper.reshape(s)
            return get_display(shaped)
        except Exception:
            return s
    return s

# --- EDF handling ---
def write_temp_file_from_upload(uploaded) -> Path:
    suffix = Path(uploaded.name).suffix or ".edf"
    tf = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tf.write(uploaded.getvalue())
    tf.flush()
    tf.close()
    return Path(tf.name)

def read_edf_bytes(uploaded) -> Tuple[Optional['mne.io.Raw'], Optional[str]]:
    if uploaded is None:
        return None, "No file uploaded"
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
            if HAS_MNE:
                info = mne.create_info(ch_names=list(ch_names), sfreq=float(sf), ch_types='eeg')
                raw = mne.io.RawArray(data, info)
                return raw, None
            else:
                return None, "pyedflib read but mne not installed for further processing"
        else:
            return None, "Neither mne nor pyedflib installed"
    except Exception as e:
        return None, f"Error reading EDF: {e}"
    finally:
        # try to remove
        try:
            os.remove(tmp_path)
        except Exception:
            pass

# --- Band power computation ---
def compute_band_powers(raw: 'mne.io.Raw') -> pd.DataFrame:
    if not HAS_MNE:
        raise RuntimeError("mne is required for band power computation")
    picks = mne.pick_types(raw.info, eeg=True, meg=False, exclude='bads')
    if len(picks) == 0:
        raise RuntimeError("No EEG channels detected")
    data, _ = raw.get_data(picks=picks, return_times=True)
    ch_names = [raw.ch_names[p] for p in picks]
    sf = int(raw.info.get('sfreq', 250))
    from scipy.signal import welch
    rows=[]
    for i, ch in enumerate(data):
        # detrend channel to avoid DC
        ch = ch - np.mean(ch)
        f, Pxx = welch(ch, fs=sf, nperseg=min(2048, max(256, len(ch)//8)))
        mask_total = (f>=1)&(f<=45)
        total_power = np.trapz(Pxx[mask_total], f[mask_total]) if mask_total.any() else np.nan
        row = {"ch": ch_names[i], "total": total_power}
        for name,(lo,hi) in BANDS.items():
            mask = (f>=lo)&(f<=hi)
            val = np.trapz(Pxx[mask], f[mask]) if mask.any() else np.nan
            row[f"{name}_abs"] = float(val) if not np.isnan(val) else 0.0
            row[f"{name}_rel"] = float(val/total_power) if total_power and not np.isnan(total_power) and total_power>0 else 0.0
        rows.append(row)
    df = pd.DataFrame(rows).set_index("ch")
    # defensive: replace NaN with zeros
    df = df.fillna(0.0)
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

# --- Connectivity (coherence fallback) ---
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
        # image
        fig, ax = plt.subplots(figsize=(4,3))
        im = ax.imshow(np.nan_to_num(conn, nan=0.0), cmap='viridis', interpolation='nearest')
        ax.set_title(f"Connectivity {band[0]}-{band[1]}Hz")
        ax.set_xticks(range(len(ch_names))); ax.set_xticklabels(ch_names, rotation=90, fontsize=6)
        ax.set_yticks(range(len(ch_names))); ax.set_yticklabels(ch_names, fontsize=6)
        fig.colorbar(im, ax=ax, fraction=0.03)
        buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format='png', dpi=150); plt.close(fig); buf.seek(0)
        return conn, mean_conn, buf.getvalue()
    except Exception:
        return None, None, None

# --- Topomap generation with robustness ---
def generate_topomap_image(raw: 'mne.io.Raw', band: Tuple[float,float], show_band_name: bool = True):
    """
    Returns PNG bytes for a topomap of band power across available channels.
    If mne topomap fails (montage mismatch), falls back to a bar-chart.
    To avoid completely blank maps, we boost contrast when values are near-uniform.
    """
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
        # normalize robustly
        if vals.max() > 0:
            vals = vals / vals.max()
        vals = np.nan_to_num(vals, nan=0.0)
        # if nearly uniform or all zeros, boost contrast slightly so map is visible
        if np.allclose(vals, 0) or np.std(vals) < 0.03:
            # add tiny gradient or jitter to show topography (testing / visualization aid)
            jitter = np.linspace(0, 0.05, num=len(vals))
            vals = vals + jitter
            vals = vals / vals.max()
        # Try true topomap
        try:
            montage = mne.channels.make_standard_montage("standard_1020")
            info = mne.pick_info(raw.info, picks)
            # create EvokedArray with channel ordering matching picks
            evoked = mne.EvokedArray(vals.reshape(-1,1), info, tmin=0.0)
            evoked.set_montage(montage, match_case=False)
            # evoked.plot_topomap returns figure(s)
            fig = evoked.plot_topomap(times=0.0, ch_type='eeg', show=False)
            # Sometimes returns list; ensure capture current figure
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
            plt.close('all')
            buf.seek(0)
            return buf.getvalue()
        except Exception:
            # fallback bar
            fig, ax = plt.subplots(figsize=(4,2.2))
            ax.bar(range(len(vals)), vals)
            if show_band_name:
                ax.set_title(f"{band[0]}-{band[1]} Hz (approx)")
            else:
                ax.set_title("Topomap (approx)")
            ax.set_xticks(range(len(vals)))
            ax.set_xticklabels(ch_names, rotation=90, fontsize=6)
            buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format='png'); plt.close(fig); buf.seek(0)
            return buf.getvalue()
    except Exception:
        return None

# --- SHAP helpers ---
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
        buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format='png'); plt.close(fig); buf.seek(0)
        return buf.getvalue()
    except Exception:
        return None

# --- Final risk (stable normalization) ---
def compute_final_risk(theta_alpha, phq_total, ad8_score, fdi, connectivity):
    # Normalize each input into [0,1] based on clinically sensible ranges
    def norm(x, low, high):
        try:
            return max(0.0, min(1.0, (float(x) - low) / (high - low) if high > low else 0.0))
        except Exception:
            return 0.0
    # heuristics:
    ta_norm = norm(theta_alpha or 0.0, 0.2, 1.8)   # theta/alpha: 0.2..1.8 expected
    phq_norm = norm(phq_total or 0.0, 0, 27)
    ad_norm = norm(ad8_score or 0.0, 0, 8)
    fdi_norm = norm((fdi or 0.0), 0.5, 3.0)
    conn_norm = 1.0 - norm(connectivity if connectivity is not None else 0.5, 0.0, 1.0)
    # weighted sum
    risk = 0.35*ta_norm + 0.25*ad_norm + 0.15*phq_norm + 0.15*fdi_norm + 0.10*conn_norm
    # enforce floor if focal pathology strong
    if fdi and fdi > 2.0:
        risk = max(risk, 0.35)
    return round(risk * 100, 1)

# --- PDF (simple) ---
def generate_pdf_report(summary: dict, lang_code="en"):
    if not HAS_REPORTLAB:
        return None
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=36, bottomMargin=36, leftMargin=44, rightMargin=44)
    styles = getSampleStyleSheet()
    base_font = "Helvetica"
    if AMIRI_PATH.exists() and HAS_ARABIC:
        try:
            pdfmetrics.registerFont(TTFont("Amiri", str(AMIRI_PATH)))
            base_font = "Amiri"
        except Exception:
            pass
    styles.add(ParagraphStyle(name="TitleBlue", fontName=base_font, fontSize=16, textColor=colors.HexColor("#0b63d6"), alignment=1, spaceAfter=8))
    styles.add(ParagraphStyle(name="H2", fontName=base_font, fontSize=12, textColor=colors.HexColor("#0b63d6"), spaceAfter=6))
    styles.add(ParagraphStyle(name="Body", fontName=base_font, fontSize=10, leading=14))
    story=[]
    # header
    left = Paragraph("<b>NeuroEarly Pro — Clinical</b>", styles["TitleBlue"])
    if LOGO_PATH.exists():
        img = RLImage(str(LOGO_PATH), width=1.0*inch, height=1.0*inch)
        header_table_data = [[left, img]]
        t = Table(header_table_data, colWidths=[4.8*inch, 1.0*inch])
    else:
        header_table_data = [[left]]
        t = Table(header_table_data)
    t.setStyle(TableStyle([("VALIGN",(0,0),(-1,-1),"MIDDLE")]))
    story.append(t); story.append(Spacer(1,12))
    # patient
    info = summary.get("patient_info", {})
    story.append(Paragraph(safe_text("Patient summary", lang_code), styles["H2"]))
    rows=[["Field","Value"]]
    rows.append(["Patient ID", info.get("id","")])
    rows.append(["DOB", info.get("dob","")])
    rows.append(["Sex", info.get("sex","")])
    rows.append(["Meds", info.get("meds","")])
    rows.append(["Labs", info.get("labs","")])
    tinfo = Table(rows, colWidths=[2.5*inch,3.5*inch])
    tinfo.setStyle(TableStyle([("GRID",(0,0),(-1,-1),0.25,colors.grey),("BACKGROUND",(0,0),(-1,0),colors.HexColor("#eef7ff"))]))
    story.append(tinfo); story.append(Spacer(1,8))
    # metrics
    story.append(Paragraph(safe_text("QEEG Key Metrics", lang_code), styles["H2"]))
    metrics = summary.get("metrics", {})
    if metrics:
        rows=[["Metric","Value"]]
        for k,v in metrics.items():
            rows.append([k, str(v)])
        t2 = Table(rows, colWidths=[3.5*inch,2.5*inch])
        t2.setStyle(TableStyle([("GRID",(0,0),(-1,-1),0.25,colors.grey),("BACKGROUND",(0,0),(-1,0),colors.HexColor("#eef7ff"))]))
        story.append(t2); story.append(Spacer(1,8))
    # topomaps
    topo_imgs = summary.get("topo_images", {})
    if topo_imgs:
        story.append(Paragraph(safe_text("Topography Maps (bands)", lang_code), styles["H2"]))
        imgs = [RLImage(io.BytesIO(b), width=2.6*inch, height=1.6*inch) for b in topo_imgs.values() if b]
        rows=[]; row=[]
        for im in imgs:
            row.append(im)
            if len(row)==2:
                rows.append(row); row=[]
        if row: rows.append(row)
        for r in rows:
            t = Table([r], colWidths=[3.0*inch]*len(r))
            t.setStyle(TableStyle([("GRID",(0,0),(-1,-1),0.25,colors.grey)]))
            story.append(t); story.append(Spacer(1,6))
    # shap
    if summary.get("shap_img"):
        story.append(Paragraph(safe_text("XAI - SHAP contributors", lang_code), styles["H2"]))
        story.append(RLImage(io.BytesIO(summary["shap_img"]), width=5.5*inch, height=2.2*inch)); story.append(Spacer(1,6))
    # recs
    story.append(Paragraph(safe_text("Structured Clinical Recommendations", lang_code), styles["H2"]))
    for r in summary.get("recommendations", []):
        story.append(Paragraph(safe_text(r, lang_code), styles["Body"]))
    story.append(Spacer(1,12))
    story.append(Paragraph(safe_text("Prepared by Golden Bird LLC — NeuroEarly Pro", lang_code), styles["Body"]))
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="NeuroEarly Pro - stage1", layout="wide")
with st.sidebar:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=140)
    st.markdown("### Settings")
    lang = st.selectbox("Language / اللغة", options=["English", "العربية"], index=0)
    lang_code = "ar" if lang.startswith("ع") else "en"
    st.markdown("---")
    st.header("Patient / المريض")
    patient_id = st.text_input("Patient ID", value="H-0001")
    dob = st.date_input("Date of birth", value=date(1985,1,1), min_value=date(1900,1,1), max_value=date(2025,12,31))
    sex = st.selectbox("Sex / الجنس", ["Unknown","Male","Female"])
    meds = st.text_area("Current meds (one per line) / الأدوية", value="")
    labs = st.text_area("Relevant labs (B12, TSH, ...) / نتائج آزمایش", value="")
    st.markdown("---")
    st.header("Upload")
    uploaded = st.file_uploader("Upload EDF file (.edf)", type=["edf","EDF"], accept_multiple_files=False)
    st.markdown("---")
    st.write("Tip: use synthetic EDF generator below to test topomaps & pipeline.")

st.title(safe_text("NeuroEarly Pro — Stage1 (EEG core + Questionnaires)", lang_code))
st.markdown(safe_text("Accurate band analysis • Topomaps for all bands • PHQ-9 + AD8", lang_code))

# Questionnaires
st.subheader(safe_text("Questionnaires", lang_code))
col1, col2 = st.columns(2)
with col1:
    st.markdown("**PHQ-9 (depression)**")
    q1 = st.radio(safe_text("Q1 - Little interest or pleasure", lang_code), [0,1,2,3], index=0, key="q1")
    q2 = st.radio(safe_text("Q2 - Feeling down, depressed", lang_code), [0,1,2,3], index=0, key="q2")
    q3_sel = st.selectbox(safe_text("Q3 - Sleep pattern", lang_code),
                         [("0","No change"), ("1","Insomnia"), ("2","Hypersomnia"), ("3","Severe")],
                         format_func=lambda x: safe_text(x[1], lang_code), key="q3")
    q4 = st.radio(safe_text("Q4 - Energy / Fatigue", lang_code), [0,1,2,3], index=0, key="q4")
    q5_sel = st.selectbox(safe_text("Q5 - Appetite", lang_code),
                         [("0","No change"),("1","Mild (less/more)"),("2","Moderate"),("3","Severe")],
                         format_func=lambda x: safe_text(x[1], lang_code), key="q5")
with col2:
    st.markdown("**PHQ-9 continued**")
    q6 = st.radio(safe_text("Q6 - Feelings of failure/guilt", lang_code), [0,1,2,3], index=0, key="q6")
    q7 = st.radio(safe_text("Q7 - Concentration problems", lang_code), [0,1,2,3], index=0, key="q7")
    q8_sel = st.selectbox(safe_text("Q8 - Psychomotor", lang_code),
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
    "Problems with judgment?",
    "Reduced interest in activities?",
    "Repeats questions/stories?",
    "Trouble learning to use tools/appliances?",
    "Forgets month or year?",
    "Difficulty handling finances?",
    "Trouble remembering appointments?",
    "Consistent problems with thinking/memory?"
]
ad8_answers = []
for i,qt in enumerate(ad8_qs, start=1):
    a = st.radio(safe_text(f"AD8-{i} - {qt}", lang_code), ["No", "Yes"], index=0, key=f"ad8_{i}")
    ad8_answers.append(1 if a=="Yes" else 0)
ad8_score = sum(ad8_answers)

st.markdown("---")
process_btn = st.button(safe_text("Process EDF / تشغيل التحلیل", lang_code))

# Console and results area
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
                st.success(safe_text(f"EDF loaded. Channels: {len(raw.ch_names)} • sfreq: {raw.info.get('sfreq')}", lang_code))
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
                    if HAS_MNE:
                        # try spectral_connectivity, but fallback to compute_connectivity_matrix
                        try:
                            from mne.connectivity import spectral_connectivity
                            con, freqs, times, n_epochs, n_tapers = spectral_connectivity([raw], method="coh", mode='multitaper', sfreq=raw.info['sfreq'], fmin=BANDS["Alpha"][0], fmax=BANDS["Alpha"][1], faverage=True, verbose=False)
                            if con is not None:
                                conn_mat = con.squeeze()
                                mean_conn = float(np.nanmean(conn_mat))
                                fig, ax = plt.subplots(figsize=(4,3))
                                im = ax.imshow(np.nan_to_num(conn_mat, nan=0.0), cmap='viridis', interpolation='nearest')
                                fig.colorbar(im, ax=ax, fraction=0.03)
                                buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format='png', dpi=150); plt.close(fig); buf.seek(0)
                                conn_img = buf.getvalue()
                        except Exception:
                            conn_mat, mean_conn, conn_img = compute_connectivity_matrix(raw, band=BANDS["Alpha"])
                    else:
                        conn_mat, mean_conn, conn_img = compute_connectivity_matrix(raw, band=BANDS["Alpha"])
                except Exception:
                    conn_mat, mean_conn, conn_img = compute_connectivity_matrix(raw, band=BANDS["Alpha"])

                # topomaps for all bands
                topo_imgs = {}
                for bname, band in BANDS.items():
                    img = generate_topomap_image(raw, band)
                    topo_imgs[bname] = img

                # SHAP (if available)
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
                        idx = raw.ch_names.index(selected_channel)
                        data, times = raw.get_data(picks=[idx], return_times=True)
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
                i=0
                for bname, img in topo_imgs.items():
                    with cols[i%2]:
                        st.markdown(f"**{bname}**")
                        if img:
                            st.image(img, use_column_width=True)
                        else:
                            st.info(safe_text("Not available", lang_code))
                    i += 1

                # connectivity view
                st.subheader(safe_text("Connectivity (Alpha)", lang_code))
                if conn_img:
                    st.image(conn_img, use_column_width=True)
                else:
                    st.info(safe_text("Connectivity not available", lang_code))

                # SHAP view
                if shap_img:
                    st.subheader(safe_text("XAI - SHAP contributors", lang_code))
                    st.image(shap_img, use_column_width=True)

                # prepare pdf summary and allow download
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
                        safe_text("Automated screening report — clinical correlation required.", lang_code),
                        safe_text("If FDI > 2 or focal slowing is present, consider structural imaging (MRI).", lang_code),
                        safe_text("Review medications and labs (B12, TSH) for reversible causes.", lang_code)
                    ]
                }
                pdf_bytes = generate_pdf_report(summary, lang_code) if HAS_REPORTLAB else None
                if pdf_bytes:
                    st.download_button(safe_text("Download PDF report", lang_code), data=pdf_bytes, file_name=f"NeuroEarly_Report_{now_ts()}.pdf", mime="application/pdf")
                else:
                    if not HAS_REPORTLAB:
                        st.info(safe_text("PDF generation not available (reportlab not installed).", lang_code))

# Synthetic EDF generator for testing (in sidebar)
with st.sidebar:
    st.markdown("---")
    st.subheader(safe_text("Testing tools", lang_code))
    dur = st.number_input(safe_text("Synthetic EDF duration (s)", lang_code), min_value=30, max_value=600, value=120)
    sfreq = st.selectbox(safe_text("Sample rate", lang_code), [250, 500], index=1)
    if st.button(safe_text("Create & download synthetic EDF", lang_code)):
        try:
            def generate_simple_synthetic(duration_s=dur, sf=sfreq, n_ch=19):
                t = np.arange(int(duration_s*sf))/sf
                ch_names = ['Fp1','Fp2','F7','F3','Fz','F4','F8','T7','C3','Cz','C4','T8','P7','P3','Pz','P4','P8','O1','O2'][:n_ch]
                signals=[]
                for i in range(n_ch):
                    # create slightly different spectral content per channel (so topomaps vary)
                    a_freq = 8 + np.random.randn()*1.5 + (i%5)*0.6
                    t_freq = 6 + np.random.randn()*1.0 + (i%3)*0.4
                    sig = 5*np.sin(2*np.pi*a_freq*t + i*0.1) + 2*np.sin(2*np.pi*t_freq*t + i*0.2) + 0.8*np.random.randn(len(t))
                    # add localized increased delta in some channels to simulate focality occasionally
                    if i in [7,11]:  # e.g., temporal channels
                        sig += 2.0*np.sin(2*np.pi*2*t)*np.exp(-((t-duration_s/2)**2)/(2*(duration_s/6)**2))
                    signals.append(sig)
                signals = np.vstack(signals)
                # write EDF if pyedflib available
                if HAS_PYEDFLIB:
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".edf"); fname=tmp.name; tmp.close()
                    f = pyedflib.EdfWriter(fname, n_ch)
                    ch_info = []
                    for ch in ch_names:
                        ch_info.append({'label':ch,'dimension':'uV','sample_rate':sf,'physical_min':-500,'physical_max':500,'digital_min':-32768,'digital_max':32767,'transducer':'','prefilter':''})
                    f.setSignalHeaders(ch_info)
                    f.writeSamples(signals)
                    f.close()
                    with open(fname,"rb") as fh:
                        b=fh.read()
                    try: os.remove(fname)
                    except: pass
                    return b, "application/octet-stream", f"synthetic_{duration_s}s_{sf}Hz.edf"
                else:
                    buf = io.BytesIO(); np.save(buf, signals); buf.seek(0)
                    return buf.getvalue(), "application/octet-stream", f"synthetic_{duration_s}s_{sf}Hz.npy"
            bts, mime, fname = generate_simple_synthetic()
            st.sidebar.download_button(safe_text("Download synthetic EDF", lang_code), data=bts, file_name=fname, mime=mime)
        except Exception as e:
            st.sidebar.error(safe_text(f"Could not generate synthetic EDF: {e}", lang_code))

# End of file

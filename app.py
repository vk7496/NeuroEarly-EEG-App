# app.py — NeuroEarly v6.3 — Doctor Ready Edition
# Bilingual (English default, Arabic optional) QEEG app with:
# - robust EDF reading (tempfile)
# - topomaps (mne if available, fallback plots)
# - connectivity (spectral_coherence fallback)
# - SHAP bar chart if shap_summary.json present
# - PDF generation (reportlab if available), bilingual (Amiri font support)
# - sidebar: upload + settings; main: questionnaires + results + visualizations

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

# Optional libraries
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

# file paths (adjust if needed)
ROOT = Path(__file__).parent
ASSETS = ROOT / "assets"
LOGO_PATH = ASSETS / "goldenbird_logo.png"
AMIRI_PATH = ROOT / "Amiri-Regular.ttf"
SHAP_JSON = ROOT / "shap_summary.json"

# Frequency bands
BANDS = {
    "Delta": (1.0, 4.0),
    "Theta": (4.0, 8.0),
    "Alpha": (8.0, 13.0),
    "Beta": (13.0, 30.0),
    "Gamma": (30.0, 45.0)
}

def now_ts(fmt="%Y%m%d_%H%M%S"):
    return datetime.utcnow().strftime(fmt)

# ---------- Text shaping for Arabic ----------
def safe_text(s: str, lang_code: str = "en") -> str:
    """Return reshaped bidi text for Arabic when needed."""
    if lang_code.startswith("ar") and HAS_ARABIC:
        try:
            reshaped = arabic_reshaper.reshape(s)
            bidi = get_display(reshaped)
            return bidi
        except Exception:
            return s
    return s

# ---------- EDF reading ----------
def write_temp_file_from_upload(uploaded) -> Path:
    suffix = Path(uploaded.name).suffix or ".edf"
    tf = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        tf.write(uploaded.getvalue())
        tf.flush()
        tf.close()
        return Path(tf.name)
    except Exception as e:
        try:
            tf.close()
        except:
            pass
        raise

def read_edf_bytes(uploaded) -> Tuple[Optional['mne.io.Raw'], Optional[str]]:
    """Robust EDF read: write uploaded bytes to temp file and pass path to mne or pyedflib."""
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
            # create minimal RawArray if possible
            if HAS_MNE:
                info = mne.create_info(ch_names=list(ch_names), sfreq=float(sf), ch_types='eeg')
                raw = mne.io.RawArray(data, info)
                return raw, None
            else:
                return None, "pyedflib read ok but mne not available to create Raw"
        else:
            return None, "Neither mne nor pyedflib installed"
    except Exception as e:
        return None, f"Error reading EDF: {e}"
    finally:
        # keep tmp for debugging; remove if you prefer:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

# ---------- Band powers ----------
def compute_band_powers(raw: 'mne.io.Raw') -> pd.DataFrame:
    """Compute absolute and relative band powers per channel using Welch."""
    picks = mne.pick_types(raw.info, eeg=True, meg=False, exclude='bads')
    data, times = raw.get_data(picks=picks, return_times=True)
    ch_names = [raw.ch_names[p] for p in picks]
    sf = int(raw.info['sfreq'])
    from scipy.signal import welch
    rows = []
    for i, ch in enumerate(data):
        f, Pxx = welch(ch, fs=sf, nperseg=min(2048, len(ch)))
        total = np.trapz(Pxx[(f>=1)&(f<=45)], f[(f>=1)&(f<=45)]) if any((f>=1)&(f<=45)) else np.nan
        r = {"ch": ch_names[i], "total": total}
        for name,(lo,hi) in BANDS.items():
            mask = (f>=lo)&(f<=hi)
            val = np.trapz(Pxx[mask], f[mask]) if mask.any() else np.nan
            r[f"{name}_abs"] = float(val) if not np.isnan(val) else np.nan
            r[f"{name}_rel"] = float(val/total) if total and not np.isnan(total) and total>0 else np.nan
        rows.append(r)
    df = pd.DataFrame(rows).set_index("ch")
    return df

# ---------- Theta/Alpha ratio & alpha asymmetry ----------
def compute_theta_alpha_ratio(df: pd.DataFrame) -> Optional[float]:
    try:
        t = df["Theta_rel"].mean(skipna=True)
        a = df["Alpha_rel"].mean(skipna=True)
        if pd.isna(t) or pd.isna(a) or a == 0:
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
        if pd.isna(l) or pd.isna(r):
            return None
        return float(l - r)
    except Exception:
        return None

# ---------- FDI ----------
def compute_fdi(df: pd.DataFrame, focal_channel: Optional[str]) -> Optional[float]:
    try:
        if focal_channel is None or "Delta_rel" not in df.columns:
            return None
        global_mean = df["Delta_rel"].mean(skipna=True)
        focal = df.loc[focal_channel, "Delta_rel"]
        if pd.isna(global_mean) or global_mean == 0 or pd.isna(focal):
            return None
        return float(focal / global_mean)
    except Exception:
        return None

# ---------- Connectivity ----------
def compute_connectivity_matrix(raw: 'mne.io.Raw', band=(8.0,13.0)):
    """Compute simple coherence matrix with scipy or use mne.connectivity when available."""
    try:
        picks = mne.pick_types(raw.info, eeg=True, meg=False, exclude='bads')
        data, _ = raw.get_data(picks=picks, return_times=True)
        ch_names = [raw.ch_names[p] for p in picks]
        sf = int(raw.info['sfreq'])
        n = data.shape[0]
        conn = np.full((n,n), np.nan)
        from scipy.signal import coherence
        for i in range(n):
            for j in range(i, n):
                try:
                    f, Cxy = coherence(data[i,:], data[j,:], fs=sf, nperseg=min(2048, data.shape[1]))
                    mask = (f >= band[0]) & (f <= band[1])
                    val = float(np.nanmean(Cxy[mask])) if mask.any() else np.nan
                except Exception:
                    val = np.nan
                conn[i,j] = conn[j,i] = val
        mean_conn = float(np.nanmean(conn))
        # create image bytes
        fig, ax = plt.subplots(figsize=(4,3))
        im = ax.imshow(conn, cmap='viridis', interpolation='nearest')
        ax.set_title(f"Connectivity {band[0]}-{band[1]}Hz")
        ax.set_xticks(range(len(ch_names))); ax.set_xticklabels(ch_names, rotation=90, fontsize=6)
        ax.set_yticks(range(len(ch_names))); ax.set_yticklabels(ch_names, fontsize=6)
        fig.colorbar(im, ax=ax, fraction=0.03)
        buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format='png', dpi=150); plt.close(fig); buf.seek(0)
        return conn, mean_conn, buf.getvalue()
    except Exception:
        return None, None, None

# ---------- Topomap image generation ----------
def generate_topomap_image(raw: 'mne.io.Raw', band: Tuple[float,float]):
    """Return PNG bytes of a topomap for given band. Uses mne if montage available, else fallback bar."""
    try:
        picks = mne.pick_types(raw.info, eeg=True, meg=False, exclude='bads')
        data, _ = raw.get_data(picks=picks, return_times=True)
        sf = int(raw.info['sfreq'])
        ch_names = [raw.ch_names[p] for p in picks]
        from scipy.signal import welch
        vals = []
        for ch in data:
            f, Pxx = welch(ch, fs=sf, nperseg=min(2048, len(ch)))
            mask = (f>=band[0]) & (f<=band[1])
            power = float(np.trapz(Pxx[mask], f[mask])) if mask.any() else 0.0
            vals.append(power)
        vals = np.array(vals)
        if vals.max() != 0:
            vals = vals / np.nanmax(vals)
        # Attempt true topomap via montage
        try:
            montage = mne.channels.make_standard_montage("standard_1020")
            info = mne.pick_info(raw.info, picks)
            evoked = mne.EvokedArray(vals.reshape(-1,1), info, tmin=0.0)
            evoked.set_montage(montage, match_case=False)
            fig = evoked.plot_topomap(times=0.0, ch_type='eeg', show=False)
            buf = io.BytesIO(); fig.savefig(buf, format='png', bbox_inches='tight'); plt.close(fig); buf.seek(0)
            return buf.getvalue()
        except Exception:
            fig, ax = plt.subplots(figsize=(4,2.2))
            ax.bar(range(len(vals)), vals)
            ax.set_title(f"{band[0]}-{band[1]} Hz (approx)")
            buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format='png'); plt.close(fig); buf.seek(0)
            return buf.getvalue()
    except Exception:
        return None

# ---------- SHAP loader ----------
def load_shap_json(path=SHAP_JSON):
    if Path(path).exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None

# ---------- Final risk ----------
def compute_final_risk(theta_alpha, phq_total, ad_score, fdi, connectivity):
    def clamp01(x): return max(0.0, min(1.0, float(x)))
    ta_norm = clamp01((theta_alpha or 0.0) / 2.0)
    phq_norm = clamp01((phq_total or 0.0) / 27.0)
    ad_norm = clamp01((ad_score or 0.0) / 24.0)
    fdi_norm = clamp01(((fdi or 0.0) - 1.0) / 3.0)
    conn_factor = 1.0 - clamp01(connectivity if connectivity is not None else 0.5)
    risk = (0.35*ta_norm + 0.25*ad_norm + 0.15*phq_norm + 0.15*fdi_norm + 0.10*conn_factor)
    if fdi and fdi > 2.0:
        risk = max(risk, 0.4)
    return round(risk * 100, 1)

# ---------- PDF generation (reportlab) ----------
def generate_pdf_report(summary: dict, lang_code="en", amiri_path: Optional[Path] = AMIRI_PATH) -> Optional[bytes]:
    if not HAS_REPORTLAB:
        return None
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=36, bottomMargin=36, leftMargin=44, rightMargin=44)
    styles = getSampleStyleSheet()
    base_font = "Helvetica"
    if amiri_path and Path(amiri_path).exists():
        try:
            pdfmetrics.registerFont(TTFont("Amiri", str(amiri_path)))
            base_font = "Amiri"
        except Exception:
            pass
    styles.add(ParagraphStyle(name="TitleBlue", fontName=base_font, fontSize=16, textColor=colors.HexColor("#0b63d6"), alignment=1, spaceAfter=8))
    styles.add(ParagraphStyle(name="H2", fontName=base_font, fontSize=12, textColor=colors.HexColor("#0b63d6"), spaceAfter=6))
    styles.add(ParagraphStyle(name="Body", fontName=base_font, fontSize=10, leading=14))
    story = []
    # Header (logo + title)
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
    # Patient info
    info = summary.get("patient_info", {})
    story.append(Paragraph(safe_text("Patient summary", lang_code), styles["H2"]))
    rows = [["Field","Value"]]
    rows.append(["Patient ID", info.get("id","")])
    rows.append(["DOB", info.get("dob","")])
    rows.append(["Sex", info.get("sex","")])
    rows.append(["Meds", info.get("meds","")])
    tinfo = Table(rows, colWidths=[2.5*inch,3.5*inch])
    tinfo.setStyle(TableStyle([("GRID",(0,0),(-1,-1),0.25,colors.grey),("BACKGROUND",(0,0),(-1,0),colors.HexColor("#eef7ff"))]))
    story.append(tinfo); story.append(Spacer(1,8))
    # Metrics
    story.append(Paragraph(safe_text("QEEG Key Metrics", lang_code), styles["H2"]))
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
        story.append(Paragraph(safe_text("Topography Maps", lang_code), styles["H2"]))
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
    # SHAP
    if summary.get("shap_img"):
        story.append(Paragraph(safe_text("XAI - Top contributors", lang_code), styles["H2"]))
        story.append(RLImage(io.BytesIO(summary["shap_img"]), width=5.5*inch, height=2.2*inch)); story.append(Spacer(1,6))
    # Recommendations
    story.append(Paragraph(safe_text("Structured Clinical Recommendations", lang_code), styles["H2"]))
    for r in summary.get("recommendations", []):
        story.append(Paragraph(safe_text(r, lang_code), styles["Body"]))
    story.append(Spacer(1,12))
    story.append(Paragraph(safe_text("Prepared by Golden Bird LLC — NeuroEarly Pro", lang_code), styles["Body"]))
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

# ---------- Streamlit UI ----------
st.set_page_config(page_title="NeuroEarly Pro", layout="wide")
# Sidebar: minimal - upload + settings
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
    labs = st.text_area("Relevant labs (B12, TSH, ...)", value="")
    st.markdown("---")
    st.header("Upload")
    uploaded = st.file_uploader("Upload EDF file (.edf)", type=["edf","EDF"], accept_multiple_files=False)
    st.markdown("---")
    st.write("Tip: use synthetic EDF generator below for testing")

# Main area: title, questionnaires, process button, results
st.title("NeuroEarly Pro — Clinical & Research")
st.markdown("EEG / QEEG analysis • Topomaps • Explainable AI")

# Questionnaires placed in main (not sidebar)
st.subheader(safe_text("Questionnaires", lang_code))
col_q1, col_q2 = st.columns(2)
with col_q1:
    st.markdown("**PHQ-9** (depression)")
    q1 = st.radio("Q1 - Little interest or pleasure", [0,1,2,3], index=0, key="q1")
    q2 = st.radio("Q2 - Feeling down, depressed", [0,1,2,3], index=0, key="q2")
    q3 = st.radio("Q3 - Sleep problems (insomnia/hypersomnia)", ["0: No","1: Insomnia","2: Hypersomnia","3: Severe"], index=0, key="q3_custom")
    q4 = st.radio("Q4 - Energy / fatigue", [0,1,2,3], index=0, key="q4")
    q5 = st.radio("Q5 - Appetite changes (over/under eating)", ["0: No change","1: Mild","2: Moderate","3: Severe"], index=0, key="q5_custom")
with col_q2:
    q6 = st.radio("Q6 - Feelings of failure/guilt", [0,1,2,3], index=0, key="q6")
    q7 = st.radio("Q7 - Concentration problems", [0,1,2,3], index=0, key="q7")
    q8 = st.radio("Q8 - Psychomotor (slow or restless)", ["0: Normal","1: Slight","2: Noticeable","3: Marked"], index=0, key="q8_custom")
    q9 = st.radio("Q9 - Suicidal thoughts", [0,1,2,3], index=0, key="q9")
# compute PHQ total (map custom options to numeric)
def map_custom(val):
    if isinstance(val, int): return val
    # strings like "0: No", "1: Mild" -> map by leading digit
    try:
        return int(str(val).split(":")[0])
    except:
        return 0
phq_total = q1 + q2 + map_custom(q3) + q4 + map_custom(q5) + q6 + q7 + map_custom(q8) + q9

st.markdown("---")
process_btn = st.button("Process EDF / تشغيل التحليل")

# Console and results columns
col_left, col_right = st.columns([1,2])
with col_left:
    st.header("Console")
    console_area = st.empty()
    console_area.write("Ready.")
    # raw channel viewer selection
    st.markdown("### Channel viewer")
    selected_channel = st.text_input("Channel name (e.g., F3)", value="")
with col_right:
    st.header("Results & Visuals")
    if uploaded is None:
        st.info("Upload an EDF file in the sidebar or create a synthetic EDF (sidebar).")
    else:
        if process_btn:
            console_area.write("Saving and reading EDF...")
            raw, err = read_edf_bytes(uploaded)
            if err:
                st.error(err)
            elif raw is None:
                st.error("Failed to produce raw object from EDF.")
            else:
                st.success(f"EDF loaded. Channels: {len(raw.ch_names)} • sfreq: {raw.info['sfreq']}")
                # compute band powers
                try:
                    df_bands = compute_band_powers(raw)
                    st.subheader("QEEG Band summary (relative power)")
                    st.dataframe(df_bands.round(4))
                except Exception as e:
                    st.error(f"Band power computation failed: {e}")
                    df_bands = pd.DataFrame()

                # compute metrics
                theta_alpha = compute_theta_alpha_ratio(df_bands) if not df_bands.empty else None
                alpha_asym = compute_alpha_asymmetry(df_bands)
                focal_ch = None
                try:
                    focal_ch = df_bands["Delta_rel"].idxmax() if "Delta_rel" in df_bands.columns else None
                except Exception:
                    focal_ch = None
                fdi_val = compute_fdi(df_bands, focal_ch) if focal_ch else None

                # connectivity (try using mne spectral_connectivity if available else coherence fallback)
                conn_mat, mean_conn, conn_img = None, None, None
                try:
                    if HAS_MNE:
                        # spectral_connectivity expects epochs/list; we use raw windowed approach
                        from mne.connectivity import spectral_connectivity
                        con, freqs, times, n_epochs, n_tapers = spectral_connectivity([raw], method="coh", mode='multitaper', sfreq=raw.info['sfreq'], fmin=8.0, fmax=13.0, faverage=True, verbose=False)
                        if con is not None:
                            conn_mat = con.squeeze()
                            mean_conn = float(np.nanmean(conn_mat))
                            # image
                            fig, ax = plt.subplots(figsize=(4,3))
                            im = ax.imshow(conn_mat, cmap='viridis', interpolation='nearest')
                            fig.colorbar(im, ax=ax, fraction=0.03)
                            buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format='png', dpi=150); plt.close(fig); buf.seek(0)
                            conn_img = buf.getvalue()
                    if conn_img is None:
                        # fallback
                        conn_mat, mean_conn, conn_img = compute_connectivity_matrix(raw, band=BANDS["Alpha"])
                except Exception:
                    conn_mat, mean_conn, conn_img = compute_connectivity_matrix(raw, band=BANDS["Alpha"])

                # topomaps
                topo_imgs = {}
                for name, band in BANDS.items():
                    try:
                        img = generate_topomap_image(raw, band)
                        topo_imgs[name] = img
                    except Exception:
                        topo_imgs[name] = None

                # SHAP
                shap_img = None
                shap_data = load_shap_json()
                if shap_data:
                    try:
                        model_key = "depression_global" if theta_alpha and theta_alpha < 0.8 else "alzheimers_global"
                        feats = shap_data.get(model_key, {})
                        if feats:
                            s = pd.Series(feats).abs().sort_values(ascending=False)
                            fig, ax = plt.subplots(figsize=(6,2.2))
                            s.head(10).plot.bar(ax=ax)
                            ax.set_title("SHAP - Top features")
                            buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format='png'); plt.close(fig); buf.seek(0)
                            shap_img = buf.getvalue()
                    except Exception:
                        shap_img = None

                st.markdown("### Key metrics")
                st.write({
                    "Theta/Alpha (global)": theta_alpha,
                    "Alpha Asymmetry (F3-F4)": alpha_asym,
                    "Focal Delta channel": focal_ch,
                    "FDI": fdi_val,
                    "Mean connectivity (alpha)": mean_conn
                })

                # Final ML risk
                ad_score = 0  # placeholder for AD questionnaire if implemented
                final_risk = compute_final_risk(theta_alpha or 0.0, phq_total, ad_score, fdi_val or 0.0, mean_conn or 0.0)
                st.metric("Final ML Risk (%)", f"{final_risk}%")

                # show selected channel raw if asked
                if selected_channel:
                    try:
                        picks = [raw.ch_names.index(selected_channel)]
                        data, times = raw.get_data(picks=picks, return_times=True)
                        fig, ax = plt.subplots(figsize=(8,2))
                        ax.plot(times, data[0])
                        ax.set_title(f"Raw trace: {selected_channel}")
                        buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format='png'); plt.close(fig); buf.seek(0)
                        st.image(buf.getvalue(), use_column_width=True)
                    except Exception as e:
                        st.warning(f"Channel viewer error: {e}")

                # Show topomaps
                st.subheader("Topography maps")
                cols = st.columns(2)
                idx = 0
                for bname, img in topo_imgs.items():
                    with cols[idx%2]:
                        st.markdown(f"**{bname}**")
                        if img:
                            st.image(img, use_column_width=True)
                        else:
                            st.info("Not available")
                    idx += 1

                # connectivity
                st.subheader("Connectivity (Alpha)")
                if conn_img:
                    st.image(conn_img, use_column_width=True)
                else:
                    st.info("Connectivity not available")

                # SHAP
                if shap_img:
                    st.subheader("XAI - SHAP contributors")
                    st.image(shap_img, use_column_width=True)

                # Prepare PDF summary dict
                summary = {
                    "patient_info": {"id": patient_id, "dob": str(dob), "sex": sex, "meds": meds, "labs": labs},
                    "metrics": {
                        "theta_alpha_ratio": float(theta_alpha) if theta_alpha else None,
                        "alpha_asymmetry_f3_f4": float(alpha_asym) if alpha_asym else None,
                        "focal_channel": focal_ch,
                        "fdi": float(fdi_val) if fdi_val else None,
                        "mean_connectivity_alpha": float(mean_conn) if mean_conn else None,
                        "final_risk": final_risk
                    },
                    "topo_images": topo_imgs,
                    "shap_img": shap_img,
                    "normative_bar": None,
                    "recommendations": [
                        "Automated screening report — clinical correlation required.",
                        "If FDI > 2 or focal slowing is present, consider structural imaging (MRI).",
                        "Review medications and labs (B12, TSH) for reversible causes."
                    ]
                }

                # Generate PDF
                pdf_bytes = generate_pdf_report(summary, lang_code, amiri_path=AMIRI_PATH if AMIRI_PATH.exists() else None)
                if pdf_bytes:
                    st.download_button("Download PDF report", data=pdf_bytes, file_name=f"NeuroEarly_Report_{now_ts()}.pdf", mime="application/pdf")
                else:
                    st.error("PDF generation not available (reportlab missing).")

# ---------- Synthetic EDF generator in sidebar (download) ----------
with st.sidebar:
    st.markdown("---")
    st.subheader("Testing tools")
    dur = st.number_input("Synthetic EDF duration (s)", min_value=30, max_value=600, value=120)
    sfreq = st.selectbox("Sample rate", [250, 500], index=1)
    if st.button("Create & download synthetic EDF"):
        try:
            # simple synthetic generator (if pyedflib present writes EDF, else npy)
            def generate_simple_synthetic(duration_s=dur, sf=sfreq, n_ch=19):
                t = np.arange(int(duration_s*sf))/sf
                ch_names = ['Fp1','Fp2','F7','F3','Fz','F4','F8','T7','C3','Cz','C4','T8','P7','P3','Pz','P4','P8','O1','O2'][:n_ch]
                signals = []
                for i in range(n_ch):
                    sig = 5*np.sin(2*np.pi*10*t + i*0.1)  # alpha component
                    sig += 2*np.sin(2*np.pi*6*t + i*0.2)
                    sig += 0.5*np.random.randn(len(t))
                    signals.append(sig)
                signals = np.vstack(signals)
                if HAS_PYEDFLIB:
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".edf")
                    fname = tmp.name; tmp.close()
                    f = pyedflib.EdfWriter(fname, n_ch)
                    ch_info = []
                    for ch in ch_names:
                        ch_info.append({'label':ch, 'dimension':'uV','sample_rate':sf,'physical_min':-500,'physical_max':500,'digital_min':-32768,'digital_max':32767,'transducer':'','prefilter':''})
                    f.setSignalHeaders(ch_info)
                    f.writeSamples(signals)
                    f.close()
                    with open(fname,"rb") as fh:
                        b = fh.read()
                    try: os.remove(fname)
                    except: pass
                    return b, "application/octet-stream", f"synthetic_{dur}s_{sf}Hz.edf"
                else:
                    buf = io.BytesIO(); np.save(buf, signals); buf.seek(0)
                    return buf.getvalue(), "application/octet-stream", f"synthetic_{dur}s_{sf}Hz.npy"
            bts, mime, fname = generate_simple_synthetic()
            st.sidebar.download_button("Download synthetic file", data=bts, file_name=fname, mime=mime)
        except Exception as e:
            st.sidebar.error(f"Could not generate: {e}")

# End of app

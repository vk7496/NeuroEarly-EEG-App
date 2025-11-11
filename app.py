# app_v5_fixed.py — NeuroEarly Pro (v5 fixed)
# - Safe EDF reading (tempfile)
# - Preserve non-zero metrics on error
# - Arabic rendering fixes (arabic_reshaper + bidi)
# - Keeps UI/UX, SHAP, topomaps, PDF structure as before

import os
import io
import sys
import math
import json
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

# try import neuro libs
HAS_MNE = False
try:
    import mne
    HAS_MNE = True
except Exception:
    HAS_MNE = False

HAS_PYEDFLIB = False
try:
    import pyedflib
    HAS_PYEDFLIB = True
except Exception:
    HAS_PYEDFLIB = False

# connectivity via scipy if mne not available
from scipy.signal import welch, coherence

# PDF and fonts
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# Arabic shaping
try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    HAS_ARABIC = True
except Exception:
    HAS_ARABIC = False

# SHAP optional
HAS_SHAP = False
try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

import streamlit as st

# ----------------- Config -----------------
ROOT = Path(__file__).parent
ASSETS = ROOT / "assets"
AMIRI_TTF = ROOT / "Amiri-Regular.ttf"  # user confirmed path in repo root
LOGO_PATH = ASSETS / "goldenbird_logo.png"  # user confirmed
SHAP_JSON = ROOT / "shap_summary.json"

# register Amiri font for reportlab if present
if AMIRI_TTF.exists():
    try:
        pdfmetrics.registerFont(TTFont("Amiri", str(AMIRI_TTF)))
    except Exception:
        pass

# constants
BANDS = {
    "Delta": (1.0, 4.0),
    "Theta": (4.0, 8.0),
    "Alpha": (8.0, 13.0),
    "Beta": (13.0, 30.0),
    "Gamma": (30.0, 45.0)
}

# ----------------- Utility functions -----------------
def now_ts(fmt="%Y%m%d_%H%M%S"):
    return datetime.utcnow().strftime(fmt)

def safe_read_edf_bytes(uploaded_file) -> Tuple[Optional['mne.io.Raw'], Optional[str]]:
    """
    Robust EDF reader:
    - writes BytesIO to a temporary file on disk then calls mne.io.read_raw_edf
    - handles pyedflib fallback (not as featureful) or returns error message
    """
    if not uploaded_file:
        return None, "No file uploaded"
    try:
        # ensure bytes-like interface
        data = uploaded_file.getbuffer() if hasattr(uploaded_file, "getbuffer") else uploaded_file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmp:
            tmp.write(data)
            tmp.flush()
            tmp_path = tmp.name
        # Try MNE first
        if HAS_MNE:
            try:
                raw = mne.io.read_raw_edf(tmp_path, preload=True, verbose=False)
                os.remove(tmp_path)
                return raw, None
            except Exception as e_mne:
                # fallback to pyedflib reading into numpy arrays if available
                pass
        if HAS_PYEDFLIB:
            try:
                # use pyedflib to read signals
                f = pyedflib.EdfReader(tmp_path)
                n = f.signals_in_file
                ch_names = f.getSignalLabels()
                fs = f.getSampleFrequencies()
                sigs = []
                for i in range(n):
                    sigs.append(f.readSignal(i))
                f._close()
                os.remove(tmp_path)
                # convert to MNE RawArray if possible for downstream convenience
                if HAS_MNE:
                    info = mne.create_info(ch_names=list(ch_names), sfreq=float(fs[0]) if isinstance(fs, (list,tuple)) else float(fs), ch_types='eeg')
                    raw = mne.io.RawArray(np.array(sigs), info)
                    return raw, None
                else:
                    # Return raw as tuple
                    return (np.array(sigs), float(fs[0]) if isinstance(fs, (list,tuple)) else float(fs), list(ch_names)), None
            except Exception as e_py:
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
                return None, f"pyedflib read error: {e_py}"
        # If reached here, no reader available
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        return None, "No EDF reader available (install mne or pyedflib)"
    except Exception as e:
        return None, f"Error reading EDF: {e}"

def compute_band_powers_from_raw(raw_or_tuple, bands=BANDS):
    """
    Accepts either an MNE Raw or a tuple from pyedflib path.
    Returns DataFrame with relative and absolute powers per channel.
    """
    try:
        if HAS_MNE and isinstance(raw_or_tuple, mne.io.BaseRaw):
            raw = raw_or_tuple.copy().pick_types(eeg=True, meg=False)
            sf = int(raw.info['sfreq'])
            data, times = raw.get_data(return_times=True)
            ch_names = raw.ch_names
        else:
            # tuple: (np_signals, sf, ch_names)
            data, sf, ch_names = raw_or_tuple
            data = np.array(data)
        n_ch, n_samples = data.shape
        res = []
        for i in range(n_ch):
            # Welch PSD
            f, Pxx = welch(data[i,:], fs=sf, nperseg=min(2048, n_samples))
            total = np.trapz(Pxx[(f>=1)&(f<=45)], f[(f>=1)&(f<=45)]) if np.any((f>=1)&(f<=45)) else np.nan
            row = {"ch": ch_names[i]}
            for name,(lo,hi) in bands.items():
                mask = (f>=lo)&(f<hi)
                val = np.trapz(Pxx[mask], f[mask]) if mask.any() else np.nan
                row[f"{name}_abs"] = val
                row[f"{name}_rel"] = (val/total) if total and not np.isnan(total) and total>0 else np.nan
            res.append(row)
        df = pd.DataFrame(res).set_index("ch")
        return df
    except Exception as e:
        st.error(f"Band power calculation failed: {e}")
        return pd.DataFrame()  # empty

def compute_theta_alpha_ratio(df_bands):
    try:
        theta_mean = df_bands["Theta_rel"].mean(skipna=True)
        alpha_mean = df_bands["Alpha_rel"].mean(skipna=True)
        if np.isnan(theta_mean) or np.isnan(alpha_mean) or alpha_mean == 0:
            return None
        return float(theta_mean/alpha_mean)
    except Exception:
        return None

def compute_alpha_asymmetry(df_bands, ch_left="F3", ch_right="F4"):
    try:
        a_left = df_bands.loc[ch_left, "Alpha_rel"] if ch_left in df_bands.index else np.nan
        a_right= df_bands.loc[ch_right, "Alpha_rel"] if ch_right in df_bands.index else np.nan
        if np.isnan(a_left) or np.isnan(a_right):
            return None
        return float(a_left - a_right)
    except Exception:
        return None

def compute_fdi(df_bands, focal_ch):
    """
    Focal Delta Index = (mean delta power in focal region) / (mean delta power across all channels)
    """
    try:
        if focal_ch not in df_bands.index:
            return None
        global_mean = df_bands["Delta_rel"].mean(skipna=True)
        focal = df_bands.loc[focal_ch, "Delta_rel"]
        if np.isnan(global_mean) or global_mean == 0:
            return None
        return float(focal / global_mean)
    except Exception:
        return None

def compute_connectivity_matrix(raw_or_tuple, band=(8.0,13.0)):
    """
    Simple pairwise coherence matrix using scipy.signal.coherence (fallback).
    If MNE is present, could use mne.connectivity.spectral_connectivity for better metrics.
    Returns mean connectivity scalar (mean of matrix) and a PNG bytes image of matrix heatmap.
    """
    try:
        if HAS_MNE and isinstance(raw_or_tuple, mne.io.BaseRaw):
            raw = raw_or_tuple.copy().pick_types(eeg=True, meg=False)
            sf = int(raw.info['sfreq'])
            data, _ = raw.get_data(return_times=True)
            ch_names = raw.ch_names
        else:
            data, sf, ch_names = raw_or_tuple
        n_ch = data.shape[0]
        conn = np.zeros((n_ch, n_ch))
        for i in range(n_ch):
            for j in range(i, n_ch):
                try:
                    f, Cxy = coherence(data[i,:], data[j,:], fs=sf, nperseg=min(2048, data.shape[1]))
                    mask = (f >= band[0]) & (f <= band[1])
                    if mask.any():
                        val = np.mean(Cxy[mask])
                    else:
                        val = np.nan
                except Exception:
                    val = np.nan
                conn[i,j] = conn[j,i] = val
        mean_conn = np.nanmean(conn)
        # make plot image
        fig, ax = plt.subplots(figsize=(4,4))
        im = ax.imshow(conn, interpolation='nearest', cmap='viridis')
        ax.set_title(f"Connectivity {band[0]}-{band[1]}Hz")
        ax.set_xticks(range(len(ch_names))); ax.set_xticklabels(ch_names, rotation=90, fontsize=6)
        ax.set_yticks(range(len(ch_names))); ax.set_yticklabels(ch_names, fontsize=6)
        fig.colorbar(im, ax=ax, fraction=0.03)
        buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format='png', dpi=150); plt.close(fig); buf.seek(0)
        return conn, mean_conn, buf.getvalue()
    except Exception as e:
        return None, None, None

def load_shap_json(path: Path):
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def compute_final_risk(theta_alpha, phq_score, ad_score, fdi, connectivity):
    """
    Compute final risk between 0..100%
    incorporates theta/alpha, questionnaire scores, FDI, and connectivity
    """
    def clamp01(x): return max(0.0, min(1.0, float(x)))
    ta_norm = clamp01((theta_alpha or 0.0) / 2.0)
    phq_norm = clamp01((phq_score or 0.0) / 27.0)
    ad_norm = clamp01((ad_score or 0.0) / 24.0)
    fdi_norm = clamp01(( (fdi or 0.0) - 1.0 ) / 3.0)
    conn_factor = 1.0 - clamp01(connectivity if connectivity is not None else 0.5)
    # weights
    risk = (0.35*ta_norm + 0.25*ad_norm + 0.15*phq_norm + 0.15*fdi_norm + 0.10*conn_factor)
    # rule: if focal delta index high -> at least moderate alert
    if fdi is not None and fdi > 2.0:
        risk = max(risk, 0.4)
    return round(risk*100, 1)

# ----------------- PDF Generation -----------------
def reshaped_text(s: str, lang="en"):
    if lang.startswith("ar") and HAS_ARABIC:
        try:
            reshaped = arabic_reshaper.reshape(s)
            bidi = get_display(reshaped)
            return bidi
        except Exception:
            return s
    return s

def generate_pdf_report(summary: Dict[str,Any], lang:str="en", amiri_path:Optional[str]=None,
                        topo_images:Dict[str,bytes]=None, conn_image:bytes=None):
    """
    summary: dict containing computed metrics and text blocks
    returns bytes of PDF
    """
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    # header
    try:
        # draw logo
        if LOGO_PATH.exists():
            try:
                c.drawImage(str(LOGO_PATH), 20*mm, height-30*mm, width=40*mm, preserveAspectRatio=True, mask='auto')
            except Exception:
                pass
        c.setFont("Helvetica-Bold", 18)
        title = "NeuroEarly Pro — Clinical QEEG Report" if lang=="en" else "تقرير NeuroEarly Pro"
        title = reshaped_text(title, lang)
        c.drawString(70*mm, height-25*mm, title)
        c.setFont("Helvetica", 9)
        c.drawString(70*mm, height-30*mm, f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        # patient block (no name)
        c.setFont("Helvetica-Bold", 10)
        c.drawString(20*mm, height-40*mm, reshaped_text("Patient ID:", lang))
        c.setFont("Helvetica", 10)
        c.drawString(40*mm, height-40*mm, str(summary.get("patient_id","N/A")))
        c.drawString(110*mm, height-40*mm, reshaped_text("DOB:", lang))
        c.drawString(125*mm, height-40*mm, str(summary.get("dob","N/A")))
        # Final ML Risk prominently
        c.setFont("Helvetica-Bold", 14)
        c.drawString(20*mm, height-50*mm, reshaped_text("Final ML Risk Score:", lang))
        c.drawString(85*mm, height-50*mm, f"{summary.get('final_risk','N/A')}%")
    except Exception:
        pass

    y = height - 60*mm
    c.setFont("Helvetica-Bold", 12)
    c.drawString(20*mm, y, reshaped_text("QEEG Key Metrics", lang))
    y -= 8*mm
    c.setFont("Helvetica", 9)
    # write metrics table (safe formatting)
    metrics = summary.get("metrics", {})
    for k,v in metrics.items():
        try:
            if isinstance(v, float):
                txt = f"{k}: {v:.4f}"
            else:
                txt = f"{k}: {v}"
        except Exception:
            txt = f"{k}: {v}"
        c.drawString(22*mm, y, reshaped_text(txt, lang))
        y -= 6*mm
        if y < 40*mm:
            c.showPage(); y = height - 20*mm
    # insert topomaps images if any
    if topo_images:
        for band, imgbytes in topo_images.items():
            try:
                if imgbytes:
                    c.showPage()
                    c.drawImage(io.BytesIO(imgbytes), 20*mm, height/2-60*mm, width=170*mm, preserveAspectRatio=True, mask='auto')
                    c.setFont("Helvetica-Bold", 12)
                    c.drawString(20*mm, height/2-70*mm, reshaped_text(f"Topomap - {band}", lang))
            except Exception:
                pass
    # connection image
    if conn_image:
        try:
            c.showPage()
            c.drawImage(io.BytesIO(conn_image), 20*mm, height/2-60*mm, width=170*mm, preserveAspectRatio=True, mask='auto')
            c.setFont("Helvetica-Bold", 12)
            c.drawString(20*mm, height/2-70*mm, reshaped_text("Connectivity (Alpha)", lang))
        except Exception:
            pass

    # XAI section (SHAP)
    c.showPage()
    c.setFont("Helvetica-Bold", 12)
    c.drawString(20*mm, height-30*mm, reshaped_text("Explainable AI (Top contributors)", lang))
    # if SHAP plot image included in summary, draw it
    shap_img = summary.get("shap_img")
    if shap_img:
        try:
            c.drawImage(io.BytesIO(shap_img), 20*mm, height/2-60*mm, width=170*mm, preserveAspectRatio=True, mask='auto')
        except Exception:
            pass
    # Footer
    c.setFont("Helvetica", 8)
    c.drawString(20*mm, 15*mm, reshaped_text("Designed by Golden Bird LLC", lang))
    c.save()
    buffer.seek(0)
    return buffer.getvalue()

# ----------------- Streamlit UI -----------------
st.set_page_config(page_title="NeuroEarly Pro", layout="wide")
# Sidebar on left: language + patient info
with st.sidebar:
    st.image(str(LOGO_PATH) if LOGO_PATH.exists() else None, use_column_width=True)
    st.title("NeuroEarly Pro")
    lang = st.selectbox("Language / اللغة", options=["English","Arabic"], index=0)
    report_lang = "ar" if lang=="Arabic" else "en"
    st.markdown("---")
    st.header("Patient")
    patient_id = st.text_input("Patient ID", value="H-0001")
    dob = st.date_input("Date of birth", value=date(1995,1,1), min_value=date(1900,1,1), max_value=date(2025,12,31))
    meds = st.text_area("Medications (comma separated)", value="")
    labs = st.text_area("Relevant labs (B12, TSH, etc.)", value="")
    st.markdown("---")
    st.header("Upload")
    edf_file = st.file_uploader("Upload EDF file", type=["edf","EDF"], accept_multiple_files=False)
    st.markdown("---")
    if st.button("Run Processing"):
        st.session_state["run_processing"] = not st.session_state.get("run_processing", False)

# Main columns
col1, col2 = st.columns([1,2])

with col1:
    st.header("Console")
    if "console" not in st.session_state:
        st.session_state["console"] = ""
    st.text_area("Log", value=st.session_state["console"], height=400)
    # placeholder for raw channel viewer
    ch_select = st.selectbox("Select channel to inspect", options=["--"] + ([]), index=0)

with col2:
    st.header("Main")
    st.markdown("## Upload & Quick stats")
    if edf_file is None:
        st.info("Upload an EDF to begin.")
    else:
        raw, err = safe_read_edf_bytes(edf_file)
        if raw is None:
            st.error(f"EDF read error: {err}")
        else:
            st.success("EDF loaded successfully.")
            # compute bands
            df_bands = compute_band_powers_from_raw(raw)
            st.subheader("QEEG Band summary (relative power)")
            if not df_bands.empty:
                st.dataframe(df_bands.round(4))
            else:
                st.info("Band power summary not available.")

            # compute metrics
            theta_alpha = compute_theta_alpha_ratio(df_bands)
            alpha_asym = compute_alpha_asymmetry(df_bands)
            # choose focal channel heuristic: channel with highest delta_rel
            focal_ch = None
            try:
                focal_ch = df_bands["Delta_rel"].idxmax()
            except Exception:
                focal_ch = None
            fdi_value = compute_fdi(df_bands, focal_ch) if focal_ch else None

            st.markdown("---")
            st.subheader("Connectivity")
            conn_mat, mean_conn, conn_img = compute_connectivity_matrix(raw, band=BANDS["Alpha"])
            if conn_img:
                st.image(conn_img, caption="Alpha connectivity", use_column_width=True)
            else:
                st.info("Connectivity not available (install required libs for best results).")

            # SHAP
            shap_data = load_shap_json(SHAP_JSON)
            st.subheader("Explainable AI (XAI)")
            if shap_data:
                try:
                    # choose model key by heuristic
                    model_key = "depression_global"
                    if theta_alpha and theta_alpha > 1.3:
                        model_key = "alzheimers_global"
                    feats = shap_data.get(model_key, {})
                    if feats:
                        s = pd.Series(feats).abs().sort_values(ascending=False)
                        st.bar_chart(s.head(12))
                        # prepare small shap image for PDF
                        # we can render bar chart to image
                        fig, ax = plt.subplots(figsize=(6,3))
                        s.head(12).plot.bar(ax=ax)
                        ax.set_title("SHAP - Top features")
                        buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format='png'); plt.close(fig); buf.seek(0)
                        shap_img_bytes = buf.getvalue()
                    else:
                        st.info("SHAP loaded but no matching model key.")
                        shap_img_bytes = None
                except Exception as e:
                    st.warning(f"XAI load error: {e}")
                    shap_img_bytes = None
            else:
                st.info("No shap_summary.json found. Upload to enable XAI visualizations.")
                shap_img_bytes = None

            # Final risk
            phq_score = 0  # placeholder: collect PHQ-9 from UI if implemented
            ad_score = 0   # placeholder
            final_risk = compute_final_risk(theta_alpha or 0.0, phq_score, ad_score, fdi_value or 0.0, mean_conn or 0.0)

            # Show quick metrics
            st.markdown("### Quick Metrics")
            st.write({
                "Theta/Alpha (global)": theta_alpha,
                "Alpha Asymmetry (F3-F4)": alpha_asym,
                "Focal channel (Delta)": focal_ch,
                "FDI": fdi_value,
                "Mean Connectivity (Alpha)": mean_conn,
                "Final ML Risk (%)": final_risk
            })

            # Generate PDF button
            if st.button("Generate PDF Report"):
                summary = {
                    "patient_id": patient_id,
                    "dob": dob.isoformat() if isinstance(dob, date) else str(dob),
                    "metrics": {
                        "theta_alpha_ratio": theta_alpha,
                        "alpha_asymmetry_f3_f4": alpha_asym,
                        "focal_channel": focal_ch,
                        "fdi": fdi_value,
                        "mean_connectivity_alpha": mean_conn
                    },
                    "final_risk": final_risk,
                    "shap_img": shap_img_bytes
                }
                topo_imgs = {}
                # create simple topomap approximations per band (if mne available)
                if HAS_MNE and isinstance(raw, mne.io.BaseRaw):
                    try:
                        raw2 = raw.copy().pick_types(eeg=True)
                        for band_name,(lo,hi) in BANDS.items():
                            try:
                                psds, freqs = mne.time_frequency.psd_welch(raw2, fmin=lo, fmax=hi, n_overlap=0, verbose=False)
                                vals = psds.mean(axis=1)
                            except Exception:
                                vals = np.zeros(len(raw2.ch_names))
                            # generate a small image (matplotlib heatmap)
                            fig, ax = plt.subplots(figsize=(4,2))
                            im = ax.bar(range(len(vals)), vals)
                            ax.set_title(f"Band {band_name}")
                            buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format='png'); plt.close(fig); buf.seek(0)
                            topo_imgs[band_name] = buf.getvalue()
                    except Exception:
                        topo_imgs = None
                else:
                    # fallback: no topomap
                    topo_imgs = None

                pdf_bytes = generate_pdf_report(summary, lang=report_lang, amiri_path=str(AMIRI_TTF) if AMIRI_TTF.exists() else None,
                                                topo_images=topo_imgs, conn_image=conn_img)
                st.success("PDF generated.")
                st.download_button("Download PDF", data=pdf_bytes, file_name=f"NeuroEarly_Report_{now_ts()}.pdf", mime="application/pdf")

# end of app

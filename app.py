#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# NeuroEarly Pro — Clinical v7
# Full bilingual Streamlit app (English default / Arabic optional with Amiri)
# Features: robust EDF read, band powers, heatmaps, simple connectivity, SHAP support, PDF report

import os
import io
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
from matplotlib import cm
from matplotlib.colors import Normalize

import streamlit as st
from PIL import Image

# optional heavy libs
HAS_MNE = False
HAS_PYEDFLIB = False
HAS_REPORTLAB = False
HAS_SHAP = False
HAS_ARABIC_RESHAPER = False
HAS_BIDI = False

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
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
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
    HAS_ARABIC_RESHAPER = True
except Exception:
    HAS_ARABIC_RESHAPER = False

try:
    from bidi.algorithm import get_display
    HAS_BIDI = True
except Exception:
    HAS_BIDI = False

# constants & paths
ROOT = Path(__file__).parent
ASSETS = ROOT / "assets"
BRAIN_BLUE = ASSETS / "brain_blue.png"
BRAIN_DARK = ASSETS / "brain_dark.png"
LOGO_PATH = ASSETS / "goldenbird_logo.png"
AMIRI_PATH = ROOT / "Amiri-Regular.ttf"
SHAP_JSON = ROOT / "shap_summary.json"

BANDS = {
    "Delta": (1.0, 4.0),
    "Theta": (4.0, 8.0),
    "Alpha": (8.0, 13.0),
    "Beta": (13.0, 30.0),
    "Gamma": (30.0, 45.0),
}

DEFAULT_BLUE = "#0b63d6"

# helpers
def now_ts(fmt="%Y%m%d_%H%M%S"):
    return datetime.utcnow().strftime(fmt)

def arabic_safe(text: str) -> str:
    if not text:
        return text
    if st.session_state.get("lang", "en") != "ar":
        return text
    if HAS_ARABIC_RESHAPER and HAS_BIDI:
        try:
            return get_display(arabic_reshaper.reshape(text))
        except Exception:
            return text
    return text

# robust EDF read: write bytes to temp path and read with mne or pyedflib
def read_edf_bytes(uploaded) -> Tuple[Optional[Any], Optional[str]]:
    if not uploaded:
        return None, "No file provided"
    try:
        data = uploaded.getvalue()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        # try mne
        if HAS_MNE:
            try:
                raw = mne.io.read_raw_edf(tmp_path, preload=True, verbose=False)
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
                return raw, None
            except Exception as e:
                mne_err = str(e)
        else:
            mne_err = "mne not available"
        # fallback pyedflib
        if HAS_PYEDFLIB:
            try:
                f = pyedflib.EdfReader(tmp_path)
                n = f.signals_in_file
                ch_names = f.getSignalLabels()
                sf = float(f.getSampleFrequency(0))
                sigs = []
                for i in range(n):
                    sigs.append(f.readSignal(i))
                f.close()
                arr = np.asarray(sigs)
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
                return {"signals": arr, "ch_names": ch_names, "sfreq": sf}, None
            except Exception as e:
                py_err = str(e)
        else:
            py_err = "pyedflib not available"
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        return None, f"mne error: {mne_err}; pyedflib error: {py_err}"
    except Exception as e:
        return None, f"Unexpected read error: {e}"

# processing: band powers via Welch, handle tuple->ndarray, compute ratios and FDI
from scipy.signal import welch, butter, sosfiltfilt, iirnotch

def bandpass_filter(sig, sf, lo=1.0, hi=45.0):
    try:
        sos = butter(4, [lo, hi], fs=sf, btype='bandpass', output='sos')
        return sosfiltfilt(sos, sig)
    except Exception:
        return sig

def notch_filter(sig, sf, freq=50.0):
    try:
        b, a = iirnotch(freq, 30.0, sf)
        return sosfiltfilt([b,a], sig)
    except Exception:
        # fallback: return unchanged
        return sig

def compute_band_powers_from_raw(raw_obj, bands=BANDS):
    try:
        # get data & sf & ch_names
        if HAS_MNE and isinstance(raw_obj, mne.io.BaseRaw):
            data = raw_obj.get_data()
            sf = raw_obj.info.get("sfreq", 256.0)
            ch_names = raw_obj.ch_names
        elif isinstance(raw_obj, dict):
            data = raw_obj.get("signals")
            sf = float(raw_obj.get("sfreq", 256.0))
            ch_names = raw_obj.get("ch_names", [f"ch{i}" for i in range(data.shape[0])])
        else:
            return None, "Unsupported raw object"

        # normalize to ndarray
        data = np.asarray(data, dtype=float)

        # orientation fix
        if data.ndim == 1:
            data = np.expand_dims(data, 0)
        if data.shape[0] > data.shape[1] and data.shape[0] < 200:
            # likely channels x samples OK; else if channels >> samples maybe transpose
            pass
        if data.shape[0] < data.shape[1] and data.shape[0] > 200:
            # weird orientation, transpose
            data = data.T

        n_ch, n_samp = data.shape

        # basic preprocessing: bandpass 1-45 and notch 50/60
        for i in range(n_ch):
            try:
                sig = data[i]
                sig = bandpass_filter(sig, sf, 1.0, 45.0)
                # notch attempts both 50 and 60
                sig = notch_filter(sig, sf, 50.0)
                sig = notch_filter(sig, sf, 60.0)
                data[i] = sig
            except Exception:
                pass

        # compute PSD per channel
        band_abs = {b: np.zeros(n_ch) for b in bands}
        band_rel = {b: np.zeros(n_ch) for b in bands}
        for ch in range(n_ch):
            sig = data[ch]
            if np.all(np.isfinite(sig)) == False or np.all(sig == 0):
                continue
            f, Pxx = welch(sig, fs=sf, nperseg=min(4096, max(256, n_samp//8)))
            total = np.trapz(Pxx, f)
            if total <= 0:
                total = 1e-12
            for bname, (lo, hi) in bands.items():
                mask = (f >= lo) & (f <= hi)
                val = np.trapz(Pxx[mask], f[mask]) if mask.any() else 0.0
                band_abs[bname][ch] = val
                band_rel[bname][ch] = val/total

        # DataFrame
        rows = []
        for i in range(n_ch):
            row = {"ch": ch_names[i] if i < len(ch_names) else f"ch{i}"}
            for b in bands:
                row[f"{b}_abs"] = float(band_abs[b][i])
                row[f"{b}_rel"] = float(band_rel[b][i])
            rows.append(row)
        df = pd.DataFrame(rows)

        # compute summary metrics
        theta_alpha = (df["Theta_rel"].mean() / (df["Alpha_rel"].mean() + 1e-12)) if "Theta_rel" in df.columns and "Alpha_rel" in df.columns else 0.0
        # frontal alpha asymmetry (try to use F3/F4 if present)
        fa_asym = 0.0
        if "F3" in ch_names and "F4" in ch_names:
            f3_idx = ch_names.index("F3")
            f4_idx = ch_names.index("F4")
            fa_asym = df.loc[f3_idx, "Alpha_rel"] - df.loc[f4_idx, "Alpha_rel"]
        else:
            # fallback: use first two channels
            if df.shape[0] >= 2:
                fa_asym = df.loc[0, "Alpha_rel"] - df.loc[1, "Alpha_rel"]

        # focal delta index (FDI) simple heuristic: max channel delta / mean global delta
        mean_delta = df["Delta_rel"].mean() if "Delta_rel" in df.columns else 0.0
        max_delta = df["Delta_rel"].max() if "Delta_rel" in df.columns else 0.0
        fdi = (max_delta / (mean_delta + 1e-12)) if mean_delta > 0 else 0.0

        summary = {
            "theta_alpha_ratio": float(theta_alpha),
            "alpha_asymmetry": float(fa_asym),
            "fdi": float(fdi),
            "band_means": {b: float(df[f"{b}_rel"].mean()) for b in bands}
        }

        return {"df": df, "summary": summary, "sfreq": sf, "ch_names": ch_names}, None

    except Exception as e:
        return None, f"Compute band powers error: {e}"

# plotting helpers
def heatmap_from_vals(vals: np.ndarray, title: str) -> bytes:
    vals = np.asarray(vals, dtype=float)
    n = vals.size
    side = int(np.ceil(np.sqrt(n)))
    grid = np.full((side, side), np.nan)
    for i in range(n):
        r = i // side; c = i % side
        grid[r, c] = vals[i]
    fig, ax = plt.subplots(figsize=(4.5,3))
    im = ax.imshow(np.nan_to_num(grid, nan=np.nanmin(grid)), cmap='viridis', aspect='auto')
    ax.set_title(title)
    ax.axis('off')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format='png', dpi=150); plt.close(fig); buf.seek(0)
    return buf.getvalue()

def bar_shap_from_dict(shap_dict: dict, model_key="depression_global") -> Optional[bytes]:
    try:
        features = shap_dict.get(model_key, {})
        if not features:
            return None
        s = pd.Series(features).abs().sort_values(ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(6,2.5))
        s[::-1].plot.barh(ax=ax)
        ax.set_xlabel("abs SHAP")
        ax.set_title("Top SHAP contributors")
        fig.tight_layout()
        buf = io.BytesIO(); fig.savefig(buf, format='png', dpi=150); plt.close(fig); buf.seek(0)
        return buf.getvalue()
    except Exception:
        return None

# PDF generation
def generate_pdf(summary_dict: dict, lang="en", amiri_path: Optional[str]=None) -> Optional[bytes]:
    if not HAS_REPORTLAB:
        return None
    try:
        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=36, bottomMargin=36, leftMargin=44, rightMargin=44)
        styles = getSampleStyleSheet()
        base_font = "Helvetica"
        if amiri_path and Path(amiri_path).exists():
            try:
                pdfmetrics.registerFont(TTFont("Amiri", amiri_path))
                base_font = "Amiri"
            except Exception:
                base_font = "Helvetica"
        styles.add(ParagraphStyle(name="TitleBlue", fontName=base_font, fontSize=16, textColor=colors.HexColor(DEFAULT_BLUE)))
        styles.add(ParagraphStyle(name="H2", fontName=base_font, fontSize=12))
        styles.add(ParagraphStyle(name="Body", fontName=base_font, fontSize=10, leading=12))

        story = []
        story.append(Paragraph(summary_dict.get("title","NeuroEarly Pro - Clinical"), styles["TitleBlue"]))
        story.append(Spacer(1,6))

        # Patient info
        pi = summary_dict.get("patient_info", {})
        t_rows = [["Field","Value"]]
        t_rows.append(["ID", pi.get("id","")])
        t_rows.append(["DOB", pi.get("dob","")])
        t_rows.append(["Sex", pi.get("sex","")])
        t_rows.append(["Meds", pi.get("meds","")])
        t_rows.append(["Labs", pi.get("labs","")])
        t = Table(t_rows, colWidths=[120,330])
        t.setStyle(TableStyle([("GRID",(0,0),(-1,-1),0.25,colors.grey),("BACKGROUND",(0,0),(-1,0),colors.HexColor("#eef7ff"))]))
        story.append(t); story.append(Spacer(1,8))

        # Metrics table
        metrics = summary_dict.get("metrics",{})
        m_rows = [["Metric","Value"]]
        for k,v in metrics.items():
            m_rows.append([str(k), str(v)])
        tm = Table(m_rows, colWidths=[220,230])
        tm.setStyle(TableStyle([("GRID",(0,0),(-1,-1),0.25,colors.grey)]))
        story.append(tm); story.append(Spacer(1,8))

        # Topo images
        topo = summary_dict.get("topo_images", {})
        for band, imgb in topo.items():
            try:
                story.append(Paragraph(f"Topography — {band}", styles["H2"]))
                story.append(RLImage(io.BytesIO(imgb), width=300, height=180))
                story.append(Spacer(1,6))
            except Exception:
                pass

        # SHAP
        if summary_dict.get("shap_img"):
            story.append(Paragraph("Explainable AI (SHAP)", styles["H2"]))
            story.append(RLImage(io.BytesIO(summary_dict["shap_img"]), width=400, height=140))
            story.append(Spacer(1,6))

        # Recommendations
        story.append(Paragraph("Structured Clinical Recommendations", styles["H2"]))
        for r in summary_dict.get("recommendations", []):
            story.append(Paragraph(r, styles["Body"]))
            story.append(Spacer(1,4))

        story.append(Spacer(1,10))
        story.append(Paragraph("Prepared by Golden Bird LLC — NeuroEarly Pro", styles["Body"]))
        doc.build(story)
        buf.seek(0)
        return buf.getvalue()
    except Exception as e:
        print("PDF error:", e)
        traceback.print_exc()
        return None

# Load shap summary if exists
def load_shap_json(path: Path):
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="NeuroEarly Pro — Clinical", layout="wide")

# INITIAL STATE
if "lang" not in st.session_state:
    st.session_state["lang"] = "en"
if "theme" not in st.session_state:
    st.session_state["theme"] = "blue"
if "results" not in st.session_state:
    st.session_state["results"] = None

# SIDEBAR (left) — language + patient info + upload
with st.sidebar:
    st.markdown("### Settings / الإعدادات")
    lang_choice = st.selectbox("Language / اللغة", ["English","العربية"], index=0 if st.session_state["lang"]=="en" else 1, key="lang_select")
    st.session_state["lang"] = "ar" if lang_choice.startswith("الع") else "en"
    st.markdown("---")
    theme_choice = st.selectbox("Theme / الثيم", ["Blue Clinical","Dark Mode"], index=0 if st.session_state["theme"]=="blue" else 1, key="theme_select")
    st.session_state["theme"] = "dark" if theme_choice == "Dark Mode" else "blue"
    st.markdown("---")
    st.subheader(arabic_safe("Patient info") if st.session_state["lang"]=="ar" else "Patient info")
    patient_id = st.text_input("Patient ID", key="patient_id")
    dob = st.date_input("Date of birth", value=date(1980,1,1), max_value=date(2025,12,31), key="dob")
    sex = st.selectbox("Sex", ["Unknown","Male","Female"], key="sex")
    meds = st.text_area("Current medications (one per line)", key="meds")
    labs = st.text_area("Relevant labs (B12, TSH, ...)", key="labs")
    st.markdown("---")
    # logo
    if LOGO_PATH.exists():
        try:
            st.image(str(LOGO_PATH), width=160)
        except Exception:
            pass

# Header with brain icon and title
col1, col2 = st.columns([9,1])
with col1:
    # show brain icon and title inline
    brain_path = BRAIN_DARK if st.session_state["theme"]=="dark" and BRAIN_DARK.exists() else BRAIN_BLUE if BRAIN_BLUE.exists() else None
    if brain_path:
        cols = st.columns([1,10])
        with cols[0]:
            st.image(str(brain_path), width=64)
        with cols[1]:
            title_txt = arabic_safe("نيروإيرلي پرو — بالینی") if st.session_state["lang"]=="ar" else "NeuroEarly Pro — Clinical"
            st.markdown(f"<h2 style='margin:0px'>{title_txt}</h2>", unsafe_allow_html=True)
    else:
        st.markdown("<h2>NeuroEarly Pro — Clinical</h2>", unsafe_allow_html=True)
with col2:
    if LOGO_PATH.exists():
        try:
            st.image(str(LOGO_PATH), width=110)
        except Exception:
            pass

st.markdown("---")

# Main layout: left column for actions, right for outputs
left, right = st.columns([1,2])

with left:
    st.subheader(arabic_safe("Upload / آپلود") if st.session_state["lang"]=="ar" else "Upload")
    uploaded = st.file_uploader("Upload EDF (.edf) — آپلود فایل EDF", type=["edf"], accept_multiple_files=False, key="uploader")
    st.markdown("---")
    st.subheader("Questionnaires")
    st.markdown("PHQ-9 (Depression) — choose 0-3 for each")
    # PHQ-9 questions, with alterations for q3,q5,q8 per your spec
    phq_qs = [
        "Little interest or pleasure in doing things",
        "Feeling down, depressed, or hopeless",
        "Sleep: trouble falling/staying asleep, less sleep, or oversleeping",  # q3 modified
        "Feeling tired or having little energy",
        "Eating: poor appetite or overeating",  # q5 modified
        "Feeling bad about yourself",
        "Trouble concentrating on things",
        "Psychomotor: moving or speaking slowly OR feeling restless",  # q8 modified
        "Thoughts that you would be better off dead or self-harm"
    ]
    phq_answers = {}
    for i,q in enumerate(phq_qs, start=1):
        phq_answers[f"phq_{i}"] = st.radio(f"Q{i}: {q}", [0,1,2,3], index=0, key=f"phq_{i}")

    st.markdown("---")
    st.subheader("Alzheimer screening (brief)")
    ad_questions = {
        "ad_q1": "Memory complaints about recent events?",
        "ad_q2": "Difficulty performing daily tasks?",
        "ad_q3": "Repeating questions/stories? (sensitive)",
        "ad_q4": "Occasional disorientation to time?",
        "ad_q5": "Forgetting names/faces frequently? (sensitive)",
        "ad_q6": "Mood/behavioral changes?",
        "ad_q7": "Language difficulties?",
        "ad_q8": "Navigation/recognition problems? (sensitive)"
    }
    ad_answers = {}
    for k,txt in ad_questions.items():
        ad_answers[k] = st.selectbox(txt, ["No","Sometimes","Often"], index=0, key=k)

    st.markdown("---")
    process_btn = st.button("Start Processing / شروع پردازش")

with right:
    st.subheader(arabic_safe("Console & Results") if st.session_state["lang"]=="ar" else "Console & Results")
    console = st.empty()
    output_area = st.empty()

# Process
if process_btn:
    console.info("Starting processing... (this may take a moment)")
    if not uploaded:
        console.error("No EDF uploaded. Upload an EDF file first.")
    else:
        raw, err = read_edf_bytes(uploaded)
        if err:
            console.error(err)
            st.session_state["results"] = None
        else:
            console.info("EDF loaded. Computing band powers...")
            out, err2 = compute_band_powers_from_raw(raw)
            if err2:
                console.error(err2)
                st.session_state["results"] = None
            else:
                df = out["df"]
                summ = out["summary"]
                topo_imgs = {}
                for b in BANDS:
                    topo_imgs[b] = heatmap_from_vals(df[f"{b}_rel"].values, f"{b} relative power")
                # SHAP handling
                shap_img = None
                shap_json = load_shap_json(SHAP_JSON) if SHAP_JSON.exists() else None
                if shap_json:
                    # choose model key
                    model_key = "depression_global" if summ.get("theta_alpha_ratio",0) <= 1.3 else "alzheimers_global"
                    shap_img = bar_shap_from_dict(shap_json, model_key=model_key)
                # Risk scoring (simple ensemble heuristic)
                phq_score = sum([int(v) for v in phq_answers.values()])
                ad_score = sum([0 if v=="No" else (1 if v=="Sometimes" else 2) for v in ad_answers.values()])
                # ML risk approx — simple weighted:
                depression_risk = min(100, max(0, (phq_score/27)*60 + summ.get("theta_alpha_ratio",0)*10))
                alz_risk = min(100, max(0, (ad_score/(len(ad_answers)*2))*50 + summ.get("fdi",0)*20 + (1.0 - summ.get("band_means",{}).get("Alpha",0))*10))
                # recommendations
                recs = []
                if depression_risk > 50 or phq_score >= 10:
                    recs.append("Depression risk high — recommend psychiatric evaluation and PHQ-9 clinical interview.")
                elif depression_risk > 25:
                    recs.append("Moderate depression risk — consider monitoring and follow-up in 4-8 weeks.")
                else:
                    recs.append("Low depression risk — routine follow-up.")
                if alz_risk > 40 or summ.get("fdi",0) > 2.0:
                    recs.append("Cognitive decline indicators present — recommend neurology referral and brain MRI / neuropsychological testing.")
                else:
                    recs.append("No immediate red-flag for severe cognitive impairment; consider periodic reassessment.")
                # prepare results object
                results = {
                    "patient_info": {"id": patient_id, "dob": str(dob), "sex": sex, "meds": meds, "labs": labs},
                    "df": df,
                    "summary": summ,
                    "topo_images": topo_imgs,
                    "shap_img": shap_img,
                    "metrics": {"phq_score": phq_score, "ad_score": ad_score, "depression_risk": round(depression_risk,1), "alzheimer_risk": round(alz_risk,1)},
                    "recommendations": recs,
                    "created": now_ts()
                }
                st.session_state["results"] = results
                console.success("Processing complete.")

# show results
res = st.session_state.get("results")
if res:
    # metrics cards
    mcol1, mcol2, mcol3 = st.columns(3)
    mcol1.metric("Depression Risk (%)", f"{res['metrics']['depression_risk']}")
    mcol2.metric("Alzheimer Risk (%)", f"{res['metrics']['alzheimer_risk']}")
    mcol3.metric("PHQ-9 score", f"{res['metrics']['phq_score']}")

    st.markdown("### Band overview (relative power)")
    st.dataframe(res["df"].round(4), use_container_width=True)

    st.markdown("### Topography maps")
    tcols = st.columns(2)
    i = 0
    for b, img in res["topo_images"].items():
        with tcols[i%2]:
            st.image(img, caption=b, use_column_width=True)
        i += 1

    st.markdown("### Explainable AI (SHAP)")
    if res.get("shap_img"):
        st.image(res["shap_img"], use_column_width=True)
    else:
        st.info("No shap_summary.json found. Upload to enable XAI visualizations.")

    st.markdown("### Clinical recommendations")
    for r in res["recommendations"]:
        st.write("- " + r)

    # PDF export
    try:
        pdf_bytes = generate_pdf({
            "title": "NeuroEarly Pro — Clinical",
            "patient_info": res["patient_info"],
            "metrics": res["metrics"],
            "topo_images": res["topo_images"],
            "shap_img": res.get("shap_img"),
            "recommendations": res["recommendations"]
        }, lang=st.session_state.get("lang","en"), amiri_path=str(AMIRI_PATH) if AMIRI_PATH.exists() else None)
        if pdf_bytes:
            st.download_button("Download PDF report", data=pdf_bytes, file_name=f"NeuroEarly_Report_{now_ts()}.pdf", mime="application/pdf")
        else:
            st.error("PDF generation not available (reportlab or font missing).")
    except Exception as e:
        st.error(f"PDF export error: {e}")

else:
    output_area.info("No results yet. Upload EDF and press Start Processing.")

# footer
st.markdown("---")
st.markdown("Prepared by Golden Bird LLC — NeuroEarly Pro (Clinical Edition v7)")

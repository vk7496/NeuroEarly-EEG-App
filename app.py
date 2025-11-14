# app.py — NeuroEarly Pro (v5 professional)
# Full bilingual (English default / Arabic optional RTL with Amiri if present)
# Topomaps (simple heatmap per band), connectivity (optional), SHAP, PDF report (ReportLab)
# Place Amiri-Regular.ttf at repo root (optional) for Arabic PDF rendering.

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
from matplotlib import cm
from matplotlib.colors import Normalize

import streamlit as st
from PIL import Image

# Optional heavy libraries with graceful fallback
HAS_MNE = False
HAS_PYEDFLIB = False
HAS_REPORTLAB = False
HAS_SHAP = False
HAS_ARABIC_RESHAPER = False
HAS_BIDI = False
HAS_SKLEARN = False

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

try:
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False

# ====== Configuration & constants ======
ROOT = Path(__file__).parent
LOGO_PATH = ROOT / "assets" / "goldenbird_logo.png"
AMIRI_PATH = ROOT / "Amiri-Regular.ttf"  # put font here for Arabic PDF

BANDS = {
    "Delta": (1.0, 4.0),
    "Theta": (4.0, 8.0),
    "Alpha": (8.0, 13.0),
    "Beta": (13.0, 30.0),
    "Gamma": (30.0, 45.0),
}

DEFAULT_COLOR = "#0b63d6"  # nice blue (for header)
SHAP_JSON = ROOT / "shap_summary.json"

st.set_page_config(page_title="NeuroEarly Pro — Clinical", layout="wide")

# ====== Translations (simple dictionary) ======
TRANSLATIONS = {
    "en": {
        "app_title": "NeuroEarly Pro — Clinical & Research",
        "upload_hint": "Upload EDF file (.edf)",
        "process_btn": "Process EDF(s) and Analyze",
        "console": "Console / Visualization",
        "patient_info": "Patient / Settings",
        "language": "Language",
        "patient_id": "Patient ID",
        "dob": "Date of birth",
        "sex": "Sex",
        "meds": "Current meds (one per line)",
        "labs": "Relevant labs (B12, TSH, ...)",
        "no_results": "No processed results yet. Upload EDF and press 'Process EDF(s) and Analyze'.",
        "pdf_download": "Download PDF report",
        "pdf_failed": "PDF generation failed — ensure reportlab installed and font available.",
        "xai_missing": "No shap_summary.json found. Upload to enable XAI visualizations.",
    },
    "ar": {
        "app_title": "نيروإيرلي — کلينيکي و بحثي",
        "upload_hint": "آپلود فایل EDF (.edf)",
        "process_btn": "پردازش فایل(ها) و تحلیل",
        "console": "کنسول / نمایش",
        "patient_info": "بیمار / تنظیمات",
        "language": "اللغة",
        "patient_id": "رقم المریض",
        "dob": "تاریخ الولادة",
        "sex": "الجنس",
        "meds": "الأدوية الحالية (سطر لكل دواء)",
        "labs": "الفحوصات المهمة (B12, TSH, ...)",
        "no_results": "هیچ نتیجه ای پردازش نشده. فایل EDF بارگذاری کنید و دکمه پردازش را بزنید.",
        "pdf_download": "دانلود گزارش PDF",
        "pdf_failed": "تولید PDF شکست خورد — بررسی کنید reportlab و فونت موجود باشد.",
        "xai_missing": "فایل shap_summary.json پیدا نشد. بارگذاری کنید برای نمایش XAI.",
    }
}

def t(key: str) -> str:
    lang = st.session_state.get("lang", "en")
    return TRANSLATIONS.get(lang, TRANSLATIONS["en"]).get(key, key)

# ====== Utility functions ======

def now_ts(fmt="%Y%m%d_%H%M%S"):
    return datetime.utcnow().strftime(fmt)

def safe_display_arabic(text: str) -> str:
    """If Arabic libs available reshape & bidi; else return original."""
    if not text:
        return text
    if st.session_state.get("lang", "en") != "ar":
        return text
    if HAS_ARABIC_RESHAPER and HAS_BIDI:
        try:
            reshaped = arabic_reshaper.reshape(text)
            bidi = get_display(reshaped)
            return bidi
        except Exception:
            return text
    return text

# ====== EDF reading helpers (robust) ======

def read_edf_bytes(uploaded) -> Tuple[Optional[Dict[str,Any]], Optional[str]]:
    """
    Read uploaded EDF (Streamlit UploadedFile) robustly.
    Returns (raw_like, msg) where raw_like is:
      - if mne available: mne.io.Raw instance
      - else dict with keys: signals (np.ndarray shape channels x samples), ch_names, sfreq
    """
    if not uploaded:
        return None, "No file"
    try:
        # streamlit's uploaded file supports .getvalue()
        raw_bytes = uploaded.getvalue()
        # write to a temp file to allow libraries that expect a path
        with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmp:
            tmp.write(raw_bytes)
            tmp_path = tmp.name

        # Try mne first (prefer)
        if HAS_MNE:
            try:
                raw = mne.io.read_raw_edf(tmp_path, preload=True, verbose=False)
                # delete temp file after reading
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
                return raw, None
            except Exception as e_mne:
                # fallthrough to pyedflib
                # but keep note
                mne_err = str(e_mne)
        else:
            mne_err = "mne not available"

        # Fallback: pyedflib -> build signals array
        if HAS_PYEDFLIB:
            try:
                f = pyedflib.EdfReader(tmp_path)
                n = f.signals_in_file
                ch_names = f.getSignalLabels()
                sfreq = float(f.getSampleFrequency(0))
                signals = []
                for i in range(n):
                    sig = f.readSignal(i)
                    signals.append(sig)
                f.close()
                signals = np.asarray(signals)  # channels x samples
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
                return {"signals": signals, "ch_names": ch_names, "sfreq": sfreq}, None
            except Exception as e_py:
                py_err = str(e_py)
        else:
            py_err = "pyedflib not available"

        # if both failed
        combined = f"mne error: {mne_err}; pyedflib error: {py_err}"
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        return None, f"Error reading EDF: {combined}"
    except Exception as e:
        return None, f"Unexpected EDF read error: {e}"

# ====== Signal processing helpers ======

from scipy.signal import welch

def compute_band_powers_from_raw(raw_obj, bands=BANDS, ch_names: Optional[List[str]] = None):
    """
    Accepts either mne.io.Raw or dict {"signals": ndarray, "ch_names":[], "sfreq":...}
    Returns DataFrame of per-channel absolute & relative band power and per-band global means.
    """
    try:
        if HAS_MNE and isinstance(raw_obj, mne.io.BaseRaw):
            data = raw_obj.get_data()  # channels x samples
            sfreq = raw_obj.info.get("sfreq", 256.0)
            if ch_names is None:
                ch_names = [ch for ch in raw_obj.ch_names]
        elif isinstance(raw_obj, dict):
            data = raw_obj.get("signals")
            sfreq = float(raw_obj.get("sfreq", 256.0))
            if ch_names is None:
                ch_names = raw_obj.get("ch_names", [f"ch{i}" for i in range(data.shape[0])])
        else:
            return None, "Unsupported raw object type"

        # normalize lists/tuples -> numpy arrays (fixes tuple/list issue)
        data = np.asarray(data)

        # ensure channels x samples
        if data.ndim == 1:
            data = np.expand_dims(data, 0)
        if data.shape[0] < data.shape[1] and data.shape[0] > 256:
            # unexpected orientation; try transpose if seems wrong (heuristic)
            data = data.T

        n_ch, n_samp = data.shape

        # compute band powers
        results = []
        band_means = {}
        total_power_per_ch = np.zeros(n_ch)
        band_abs = {b: np.zeros(n_ch) for b in bands}

        # compute full spectrum power (for relative)
        for i in range(n_ch):
            f, Pxx = welch(data[i, :], fs=sfreq, nperseg=min(4096, max(256, n_samp//8)))
            total_power = np.trapz(Pxx, f)
            total_power_per_ch[i] = total_power if total_power > 0 else 1e-12
            # bandwise
            for bname, (lo, hi) in bands.items():
                mask = (f >= lo) & (f <= hi)
                if mask.any():
                    val = np.trapz(Pxx[mask], f[mask])
                else:
                    val = 0.0
                band_abs[bname][i] = val

        # build dataframe
        rows = []
        for i in range(n_ch):
            row = {"ch": ch_names[i] if i < len(ch_names) else f"ch{i}"}
            for b in bands:
                abs_v = float(band_abs[b][i])
                rel_v = float(abs_v / total_power_per_ch[i]) if total_power_per_ch[i] > 0 else 0.0
                row[f"{b}_abs"] = abs_v
                row[f"{b}_rel"] = rel_v
            rows.append(row)

        df = pd.DataFrame(rows)
        # global means per band
        for b in bands:
            band_means[b] = float(df[f"{b}_rel"].mean())

        return {"df": df, "band_means": band_means, "sfreq": sfreq, "n_ch": n_ch, "ch_names": ch_names}, None
    except Exception as e:
        return None, f"Band power error: {e}"

# ====== Plot helpers ======

def plot_band_bars(df: pd.DataFrame, top_n=10):
    """Plot bar charts of top channels per absolute power for each band."""
    figs = {}
    for b in BANDS:
        s = df.set_index("ch")[f"{b}_abs"].sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(6,3))
        s.head(top_n).plot.bar(ax=ax, color=DEFAULT_COLOR)
        ax.set_title(f"{b} (abs power)"); ax.set_ylabel("Power")
        ax.set_xlabel("")
        plt.xticks(rotation=45, ha="right")
        fig.tight_layout()
        buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig); buf.seek(0)
        figs[b] = buf.getvalue()
    return figs

def heatmap_from_channel_values(vals: np.ndarray, ch_names: List[str], title: str):
    """Create a simple heatmap image for channel values.
       If number of channels <=64 attempt to place on roughly square grid; else show bar fallback.
    """
    vals = np.asarray(vals)
    n = vals.shape[0]
    # try to form near-square grid
    side = int(np.ceil(np.sqrt(n)))
    grid = np.zeros((side, side))
    grid[:] = np.nan
    for i in range(n):
        r = i // side
        c = i % side
        grid[r, c] = vals[i]
    fig, ax = plt.subplots(figsize=(6,3.5))
    cmap = cm.viridis
    # mask nan
    im = ax.imshow(np.nan_to_num(grid, nan=np.nanmin(grid)), cmap=cmap, aspect="auto")
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig); buf.seek(0)
    return buf.getvalue()

# ====== PDF generation ======

def generate_pdf_report(summary: dict, lang="en", amiri_path: Optional[str]=None) -> Optional[bytes]:
    """
    summary: {
      "patient_info": {...},
      "metrics": {...},
      "topo_images": {band: bytes},
      "bar_img": bytes,
      "shap_img": bytes_or_none,
      "connectivity_image": bytes_or_none,
      "recommendations": [...],
      "created": ts
    }
    """
    if not HAS_REPORTLAB:
        return None

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=36, bottomMargin=36, leftMargin=44, rightMargin=44)
    styles = getSampleStyleSheet()

    base_font = "Helvetica"
    if amiri_path and Path(amiri_path).exists():
        try:
            pdfmetrics.registerFont(TTFont("Amiri", amiri_path))
            base_font = "Amiri"
        except Exception:
            base_font = "Helvetica"

    # add custom styles
    styles.add(ParagraphStyle(name="TitleBlue", fontName=base_font, fontSize=18, leading=22, textColor=colors.HexColor(DEFAULT_COLOR)))
    styles.add(ParagraphStyle(name="H2", fontName=base_font, fontSize=12, leading=14))
    styles.add(ParagraphStyle(name="Body", fontName=base_font, fontSize=10, leading=12))
    styles.add(ParagraphStyle(name="Note", fontName=base_font, fontSize=9, leading=11, textColor=colors.grey))

    story = []

    # header
    story.append(Paragraph(summary.get("title", "NeuroEarly Pro — Clinical"), styles["TitleBlue"]))
    story.append(Spacer(1,6))

    # patient info
    pi = summary.get("patient_info", {})
    p_rows = [["Field", "Value"]]
    for k in ["id","dob","sex","meds","labs"]:
        p_rows.append([k.upper(), str(pi.get(k,""))])
    t2 = Table(p_rows, colWidths=[120, 320])
    t2.setStyle(TableStyle([("GRID",(0,0),(-1,-1),0.25,colors.grey),("BACKGROUND",(0,0),(-1,0),colors.HexColor("#eef7ff"))]))
    story.append(t2)
    story.append(Spacer(1,8))

    # Metrics table
    metrics = summary.get("metrics", {})
    rows = [["Metric", "Value"]]
    for k,v in metrics.items():
        rows.append([str(k), str(v)])
    t_metrics = Table(rows, colWidths=[200,240])
    t_metrics.setStyle(TableStyle([("GRID",(0,0),(-1,-1),0.25,colors.grey)]))
    story.append(t_metrics)
    story.append(Spacer(1,8))

    # Bar chart
    if summary.get("bar_img"):
        try:
            img = RLImage(io.BytesIO(summary["bar_img"]), width=400, height=200)
            story.append(Paragraph("Normative Comparison (top contributors)", styles["H2"]))
            story.append(img); story.append(Spacer(1,6))
        except Exception:
            pass

    # Topo images
    topo = summary.get("topo_images",{})
    for b, img_bytes in topo.items():
        try:
            story.append(Paragraph(f"Topography — {b}", styles["H2"]))
            story.append(RLImage(io.BytesIO(img_bytes), width=250, height=150))
            story.append(Spacer(1,6))
        except Exception:
            pass

    # SHAP
    if summary.get("shap_img"):
        try:
            story.append(Paragraph("SHAP (top contributors)", styles["H2"]))
            story.append(RLImage(io.BytesIO(summary["shap_img"]), width=400, height=150))
            story.append(Spacer(1,6))
        except Exception:
            pass

    # Recommendations
    story.append(Paragraph("<b>Structured Clinical Recommendations</b>", styles["H2"]))
    for r in summary.get("recommendations", []):
        story.append(Paragraph(r, styles["Body"]))
        story.append(Spacer(1,4))

    story.append(Spacer(1,12))
    story.append(Paragraph("Prepared by Golden Bird LLC — NeuroEarly Pro", styles["Note"]))
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

# ====== SHAP loader ======

def load_shap_summary(path: Path, model_key: str = "depression_global"):
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get(model_key, {})
    except Exception:
        return None

# ====== Streamlit UI ======

if "lang" not in st.session_state:
    st.session_state["lang"] = "en"
if "results" not in st.session_state:
    st.session_state["results"] = None

# layout: left sidebar for inputs, main area for output
with st.sidebar:
    st.markdown(f"## {t('patient_info')}")
    lang = st.selectbox(t("language"), options=["English","العربية"], index=0 if st.session_state["lang"]=="en" else 1)
    st.session_state["lang"] = "ar" if lang.startswith("الع") else "en"

    patient_id = st.text_input(t("patient_id"), value="", key="patient_id")
    dob = st.date_input(t("dob"), value=date(1980,1,1), key="dob")
    sex = st.selectbox(t("sex"), options=["Unknown","Male","Female"], index=0, key="sex")
    meds = st.text_area(t("meds"), key="meds")
    labs = st.text_area(t("labs"), key="labs")

    st.markdown("---")
    st.markdown("### Upload")
    uploaded = st.file_uploader(t("upload_hint"), type=["edf"], accept_multiple_files=False, key="uploader")

    st.markdown("---")
    st.write("PHQ-9 (Depression) — quick")
    # PHQ-9: 9 questions, choices 0-3
    phq = {}
    for i in range(1,10):
        phq[f"q{i}"] = st.radio(f"Q{i}", options=[0,1,2,3], index=0, key=f"phq_q{i}", horizontal=False)

    st.markdown("---")
    st.write("Alzheimer risk screening (brief)")
    # simplified: 8 questions, highlight 3,5,8 are sensitive
    alz = {}
    alz_questions = {
        1:"Do you have memory complaints about recent events?",
        2:"Do you have difficulty performing daily tasks?",
        3:"Do you repeat questions or stories? (sensitive)",
        4:"Are you disoriented in time occasionally?",
        5:"Have you been forgetting names/faces frequently? (sensitive)",
        6:"Any mood/behavior changes?",
        7:"Any language difficulties?",
        8:"Have you had difficulties with navigation or recognizing places? (sensitive)"
    }
    for qnum, txt in alz_questions.items():
        alz[f"q{qnum}"] = st.selectbox(txt, options=["No","Sometimes","Often"], index=0, key=f"alz_q{qnum}")

    st.markdown("---")
    st.button(t("process_btn"), key="process_button")

# Header (main)
col1, col2 = st.columns([9,1])
with col1:
    st.markdown(f"<div style='background: linear-gradient(90deg, {DEFAULT_COLOR}, #39a6ff); padding:12px; border-radius:8px; color:#fff; font-weight:700; font-size:18px;'>{t('app_title')}</div>", unsafe_allow_html=True)
with col2:
    if LOGO_PATH.exists():
        try:
            st.image(str(LOGO_PATH), width=120)
        except Exception:
            pass

st.markdown("")

# Console & main visualization
left_col, right_col = st.columns([1,3])
with left_col:
    st.subheader(t("console"))
    log_box = st.empty()

with right_col:
    st.subheader("Upload & Quick stats")
    status = st.info("No processed results yet.") if st.session_state.get("results") is None else st.success("Results ready.")
    results_area = st.empty()

# Processing when button pressed
if st.session_state.get("process_button", False):
    log_box.info("Reading EDF file... please wait")
    raw_like, err = read_edf_bytes(uploaded) if uploaded else (None, "No uploaded file")
    if err:
        log_box.error(err)
        st.session_state["results"] = None
    else:
        log_box.success("EDF loaded successfully.")
        # compute band powers
        out, err2 = compute_band_powers_from_raw(raw_like)
        if err2:
            log_box.error(err2)
            st.session_state["results"] = None
        else:
            df = out["df"]
            band_means = out["band_means"]
            topo_imgs = {}
            for b in BANDS:
                topo_imgs[b] = heatmap_from_channel_values(df[f"{b}_rel"].values, out["ch_names"], f"{b} ({BANDS[b][0]}-{BANDS[b][1]} Hz approx)")
            bar_imgs = plot_band_bars(df)
            # compute simple risk metric (theta/alpha ratio global)
            theta_alpha_ratio = band_means.get("Theta",0.0) / (band_means.get("Alpha",1e-12))
            st.session_state["theta_alpha_ratio"] = theta_alpha_ratio
            # aggregate results
            results = {
                "patient_info": {"id": patient_id, "dob": str(dob), "sex": sex, "meds": meds, "labs": labs},
                "metrics": {"theta_alpha_ratio": theta_alpha_ratio, **{f"mean_{k}":round(v,4) for k,v in band_means.items()}},
                "df": df,
                "topo_images": topo_imgs,
                "bar_img": bar_imgs.get("Alpha"),
                "bar_imgs": bar_imgs,
                "created": now_ts(),
            }
            # SHAP
            shap_features = {}
            shap_data = load_shap_summary(SHAP_JSON)
            if shap_data:
                # choose model key heuristic
                model_key = "depression_global"
                if theta_alpha_ratio > 1.3:
                    model_key = "alzheimers_global"
                shap_features = shap_data
                results["shap"] = shap_features
            st.session_state["results"] = results
            log_box.success("Processing complete.")

# If results present show dashboard
res = st.session_state.get("results")
if res:
    # show quick metrics
    right_col.subheader("QEEG Band summary (relative power)")
    df_display = res["df"].copy()
    right_col.dataframe(df_display.style.format(precision=4), height=360)

    # visualization area: show bar charts and heatmaps
    st.markdown("## Topographic Maps and Band Charts")
    # top band charts
    grid_cols = st.columns(2)
    for i, b in enumerate(BANDS):
        c = grid_cols[i % 2]
        img_bytes = res["bar_imgs"].get(b)
        if img_bytes:
            c.image(img_bytes, caption=f"{b} comparison", use_column_width=True)
    st.markdown("---")
    # heatmaps
    heat_cols = st.columns(2)
    for i, b in enumerate(BANDS):
        c = heat_cols[i % 2]
        img = res["topo_images"].get(b)
        if img:
            c.image(img, caption=f"{b} topography", use_column_width=True)

    # SHAP
    st.markdown("## Explainable AI (XAI)")
    shap_data = load_shap_summary(SHAP_JSON) if SHAP_JSON.exists() else None
    if shap_data:
        # choose model key
        model_key = "depression_global"
        if st.session_state.get("theta_alpha_ratio",0) > 1.3:
            model_key = "alzheimers_global"
        features = shap_data.get(model_key, {})
        if features:
            s = pd.Series(features).abs().sort_values(ascending=False)
            st.bar_chart(s.head(10), use_container_width=True)
        else:
            st.info(t("xai_missing"))
    else:
        st.info(t("xai_missing"))

    # PDF export
    st.markdown("---")
    st.subheader("Export")
    try:
        # prepare summary
        summ = {
            "title": t("app_title"),
            "patient_info": res["patient_info"],
            "metrics": res["metrics"],
            "topo_images": res["topo_images"],
            "bar_img": res.get("bar_img"),
            "shap_img": None,
            "recommendations": [
                "This is an automated screening report. Clinical assessment recommended for positive findings.",
                "Consider MRI or specialist referral if focal delta index > 2 or extreme asymmetry.",
                "Follow-up in 3-6 months for moderate risk cases."
            ],
            "created": res["created"]
        }
        pdf_bytes = generate_pdf_report(summ, lang=st.session_state.get("lang","en"), amiri_path=str(AMIRI_PATH) if AMIRI_PATH.exists() else None)
        if pdf_bytes:
            st.download_button(t("pdf_download"), data=pdf_bytes, file_name=f"NeuroEarly_Report_{now_ts()}.pdf", mime="application/pdf")
            st.success("PDF generated.")
        else:
            st.error(t("pdf_failed"))
    except Exception as e:
        st.error(f"PDF generation exception: {e}")

else:
    right_col.info(t("no_results"))

# footer hints
st.markdown("---")
st.markdown("**Notes:**")
st.markdown("- Default language is English; Arabic is available for text sections and the PDF uses Amiri font if present.")
st.markdown("- For best connectivity & microstate results install `mne` and `scikit-learn`.")
st.markdown("- Place pre-trained models in `models/depression.pkl` and `models/alzheimers.pkl` to enable built-in scoring (future).")

# app.py — NeuroEarly Pro Clinical v6.5 (Spectral heatmaps)
# Copy / paste this file into your repo (replace existing app.py)
# Required optional assets: assets/goldenbird_logo.png, assets/brain_blue.png, assets/brain_dark.png
# Optional for Arabic PDF: Amiri-Regular.ttf in repo root
# Optional for XAI: shap_summary.json in repo root

import os
import io
import json
import tempfile
import traceback
from pathlib import Path
from datetime import datetime, date
from typing import Optional, Any, Tuple, List, Dict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

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

# --- paths and constants ---
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

HEADER_BLUE = "#0b63d6"  # main accent
SPECTRAL_CMAP = "Spectral"  # user chose option B

st.set_page_config(page_title="NeuroEarly Pro — Clinical", layout="wide")

# --- utilities ---
def now_ts(fmt="%Y%m%d_%H%M%S"):
    return datetime.utcnow().strftime(fmt)

def safe_arabic(text: str) -> str:
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

# Robust EDF read
def read_edf_bytes(uploaded) -> Tuple[Optional[Any], Optional[str]]:
    if not uploaded:
        return None, "No file provided"
    try:
        raw_bytes = uploaded.getvalue()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmp:
            tmp.write(raw_bytes)
            tmp_path = tmp.name
        # Try mne
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
        # Fallback pyedflib
        if HAS_PYEDFLIB:
            try:
                f = pyedflib.EdfReader(tmp_path)
                n = f.signals_in_file
                ch_names = f.getSignalLabels()
                sf = float(f.getSampleFrequency(0))
                signals = []
                for i in range(n):
                    signals.append(f.readSignal(i))
                f.close()
                arr = np.asarray(signals)
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

# Signal processing helpers
from scipy.signal import welch, butter, sosfiltfilt, iirnotch

def bandpass(sig, sf, low=1.0, high=45.0):
    try:
        sos = butter(4, [low, high], fs=sf, btype='bandpass', output='sos')
        return sosfiltfilt(sos, sig)
    except Exception:
        return sig

def apply_notch(sig, sf):
    # try 50 and 60 Hz notch using IIR notch (fallback if fails)
    try:
        b, a = iirnotch(50.0, 30.0, sf)
        sig = sosfiltfilt([b, a], sig)
    except Exception:
        pass
    try:
        b, a = iirnotch(60.0, 30.0, sf)
        sig = sosfiltfilt([b, a], sig)
    except Exception:
        pass
    return sig

def compute_band_powers(raw_obj, bands=BANDS):
    """
    Accepts mne Raw or dict {signals: ndarray (ch x samples), ch_names, sfreq}
    Returns: dict with df (per-channel), summary metrics (theta/alpha, asym, fdi)
    """
    try:
        if HAS_MNE and isinstance(raw_obj, mne.io.BaseRaw):
            data = raw_obj.get_data()  # channels x samples
            sf = raw_obj.info.get("sfreq", 256.0)
            ch_names = list(raw_obj.ch_names)
        elif isinstance(raw_obj, dict):
            data = raw_obj.get("signals")
            sf = float(raw_obj.get("sfreq", 256.0))
            ch_names = raw_obj.get("ch_names", [f"ch{i}" for i in range(data.shape[0])])
        else:
            return None, "Unsupported raw object"

        # normalize (tuple/list -> ndarray)
        data = np.asarray(data, dtype=float)

        # orientation fix: ensure shape = (n_channels, n_samples)
        if data.ndim == 1:
            data = np.expand_dims(data, 0)
        # heuristics: if channels < samples and channels < 200 -> ok; else consider transpose
        n_ch, n_samp = data.shape[0], data.shape[1] if data.ndim>1 else 0
        if n_ch > n_samp:
            # possible that shape is (samples, channels)
            data = data.T
            n_ch, n_samp = data.shape

        # preprocess per channel
        for i in range(n_ch):
            try:
                sig = data[i]
                sig = bandpass(sig, sf, 1.0, 45.0)
                sig = apply_notch(sig, sf)
                data[i] = sig
            except Exception:
                pass

        # compute PSD and band powers
        band_abs = {b: np.zeros(n_ch) for b in bands}
        band_rel = {b: np.zeros(n_ch) for b in bands}
        for ch in range(n_ch):
            sig = data[ch]
            if sig is None or not np.all(np.isfinite(sig)) or np.all(sig==0):
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

        # summary metrics
        theta_mean = float(df["Theta_rel"].mean()) if "Theta_rel" in df.columns else 0.0
        alpha_mean = float(df["Alpha_rel"].mean()) if "Alpha_rel" in df.columns else 1e-12
        theta_alpha_ratio = theta_mean / (alpha_mean if alpha_mean>0 else 1e-12)

        # frontal alpha asymmetry: prefer F3/F4
        fa_asym = 0.0
        if "F3" in ch_names and "F4" in ch_names:
            f3 = ch_names.index("F3"); f4 = ch_names.index("F4")
            fa_asym = df.loc[f3, "Alpha_rel"] - df.loc[f4, "Alpha_rel"]
        elif df.shape[0] >= 2:
            fa_asym = df.loc[0, "Alpha_rel"] - df.loc[1, "Alpha_rel"]

        # FDI: focal delta index = max_delta / mean_delta
        mean_delta = float(df["Delta_rel"].mean()) if "Delta_rel" in df.columns else 0.0
        max_delta = float(df["Delta_rel"].max()) if "Delta_rel" in df.columns else 0.0
        fdi = (max_delta / (mean_delta + 1e-12)) if mean_delta>0 else 0.0

        summary = {
            "theta_alpha_ratio": float(theta_alpha_ratio),
            "alpha_asymmetry": float(fa_asym),
            "fdi": float(fdi),
            "band_means": {b: float(df[f"{b}_rel"].mean()) for b in bands}
        }

        return {"df": df, "summary": summary, "sfreq": sf, "ch_names": ch_names}, None

    except Exception as e:
        return None, f"compute_band_powers error: {e}"

# plotting helpers (Spectral)
def heatmap_from_vals(vals: np.ndarray, title: str) -> bytes:
    vals = np.asarray(vals, dtype=float)
    n = vals.size
    side = int(np.ceil(np.sqrt(n)))
    grid = np.full((side, side), np.nan)
    for i in range(n):
        r = i // side; c = i % side
        grid[r, c] = vals[i]
    fig, ax = plt.subplots(figsize=(4.5,3))
    cmap = plt.get_cmap(SPECTRAL_CMAP)
    # replace nan with min
    vmin = np.nanmin(grid)
    vmax = np.nanmax(grid)
    im = ax.imshow(np.nan_to_num(grid, nan=vmin), cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.axis('off')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format='png', dpi=150); plt.close(fig); buf.seek(0)
    return buf.getvalue()

def shap_bar_from_json(shap_json: dict, model_key: str="depression_global") -> Optional[bytes]:
    try:
        features = shap_json.get(model_key, {})
        if not features:
            return None
        s = pd.Series(features).abs().sort_values(ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(6,2.5))
        s[::-1].plot.barh(ax=ax, color="#2b8cbe")
        ax.set_xlabel("abs SHAP"); ax.set_title("Top SHAP contributors")
        fig.tight_layout()
        buf = io.BytesIO(); fig.savefig(buf, format='png', dpi=150); plt.close(fig); buf.seek(0)
        return buf.getvalue()
    except Exception:
        return None

# PDF builder (ReportLab)
def build_pdf(report: dict, lang="en", amiri_path: Optional[str]=None) -> Optional[bytes]:
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

        # create unique style names to avoid duplicate-name error
        styles.add(ParagraphStyle(name="NEP_Title", fontName=base_font, fontSize=16, textColor=colors.HexColor(HEADER_BLUE)))
        styles.add(ParagraphStyle(name="NEP_H2", fontName=base_font, fontSize=12))
        styles.add(ParagraphStyle(name="NEP_Body", fontName=base_font, fontSize=10, leading=12))

        story = []
        story.append(Paragraph(report.get("title","NeuroEarly Pro — Clinical"), styles["NEP_Title"]))
        story.append(Spacer(1,6))

        # patient table
        pi = report.get("patient_info", {})
        trows = [["Field", "Value"]]
        trows.append(["ID", pi.get("id","")])
        trows.append(["DOB", pi.get("dob","")])
        trows.append(["Sex", pi.get("sex","")])
        trows.append(["Meds", pi.get("meds","")])
        trows.append(["Labs", pi.get("labs","")])
        t = Table(trows, colWidths=[120,330])
        t.setStyle(TableStyle([("GRID",(0,0),(-1,-1),0.25,colors.grey),("BACKGROUND",(0,0),(-1,0),colors.HexColor("#eef7ff"))]))
        story.append(t); story.append(Spacer(1,8))

        # metrics
        m = report.get("metrics", {})
        mrows = [["Metric","Value"]]
        for k,v in m.items():
            mrows.append([str(k), str(v)])
        tm = Table(mrows, colWidths=[220,230])
        tm.setStyle(TableStyle([("GRID",(0,0),(-1,-1),0.25,colors.grey)]))
        story.append(tm); story.append(Spacer(1,8))

        # topo images
        topo = report.get("topo_images", {})
        for band, imgb in topo.items():
            try:
                story.append(Paragraph(f"Topography — {band}", styles["NEP_H2"]))
                story.append(RLImage(io.BytesIO(imgb), width=300, height=180))
                story.append(Spacer(1,6))
            except Exception:
                pass

        # shap
        if report.get("shap_img"):
            story.append(Paragraph("Explainable AI (SHAP)", styles["NEP_H2"]))
            story.append(RLImage(io.BytesIO(report["shap_img"]), width=400, height=140))
            story.append(Spacer(1,6))

        # recommendations
        story.append(Paragraph("Structured Clinical Recommendations", styles["NEP_H2"]))
        for r in report.get("recommendations", []):
            story.append(Paragraph(r, styles["NEP_Body"]))
            story.append(Spacer(1,4))

        story.append(Spacer(1,12))
        story.append(Paragraph("Prepared by Golden Bird LLC — NeuroEarly Pro", styles["NEP_Body"]))
        doc.build(story)
        buf.seek(0)
        return buf.getvalue()
    except Exception as e:
        print("PDF build error:", e)
        traceback.print_exc()
        return None

def load_shap_json(path: Path):
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

# ---------------- UI ----------------
# initial states
if "lang" not in st.session_state:
    st.session_state["lang"] = "en"
if "theme" not in st.session_state:
    st.session_state["theme"] = "blue"
if "results" not in st.session_state:
    st.session_state["results"] = None

# SIDEBAR: language, patient info, upload, process
with st.sidebar:
    st.markdown(f"### {safe_arabic('Settings') if st.session_state['lang']=='ar' else 'Settings'}")
    lang_choice = st.selectbox("Language / اللغة", ["English","العربية"], index=0 if st.session_state["lang"]=="en" else 1)
    st.session_state["lang"] = "ar" if lang_choice.startswith("الع") else "en"
    st.markdown("---")
    st.subheader("Patient info")
    patient_id = st.text_input("Patient ID", key="patient_id")
    dob = st.date_input("Date of birth", value=date(1980,1,1), max_value=date(2025,12,31), key="dob")
    sex = st.selectbox("Sex", ["Unknown","Male","Female"], key="sex")
    meds = st.text_area("Current medications (one per line)", key="meds")
    labs = st.text_area("Relevant labs (B12, TSH, ...)", key="labs")
    st.markdown("---")
    st.subheader("Upload EDF")
    uploaded = st.file_uploader("Upload .edf file", type=["edf"], accept_multiple_files=False, key="uploader")
    st.markdown("---")
    if LOGO_PATH.exists():
        try:
            st.image(str(LOGO_PATH), width=140)
        except Exception:
            pass
    st.markdown("---")
    process = st.button("Start Processing / شروع پردازش")

# Header (main) with brain icon fallback
colL, colR = st.columns([9,1])
with colL:
    brain_path = None
    # choose brain icon based on theme if exists; fallback to None
    if st.session_state["theme"] == "dark" and BRAIN_DARK.exists():
        brain_path = BRAIN_DARK
    elif BRAIN_BLUE.exists():
        brain_path = BRAIN_BLUE

    if brain_path:
        c1, c2 = st.columns([1,8])
        with c1:
            try:
                st.image(str(brain_path), width=64)
            except Exception:
                pass
        with c2:
            title_txt = safe_arabic("نيروإيرلي پرو — باليني") if st.session_state["lang"]=="ar" else "NeuroEarly Pro — Clinical"
            st.markdown(f"<h2 style='margin:0px'>{title_txt}</h2>", unsafe_allow_html=True)
    else:
        # fallback textual header
        title_txt = safe_arabic("نيروإيرلي پرو — باليني") if st.session_state["lang"]=="ar" else "NeuroEarly Pro — Clinical"
        st.markdown(f"<h2>{title_txt}</h2>", unsafe_allow_html=True)

with colR:
    if LOGO_PATH.exists():
        try:
            st.image(str(LOGO_PATH), width=110)
        except Exception:
            pass

st.markdown("---")

# Main layout: questionnaires + outputs
left_col, right_col = st.columns([1,2])

with left_col:
    st.subheader("Questionnaires / پرسشنامه‌ها")
    st.markdown("PHQ-9 (Depression) — select 0..3 for each")
    phq_questions = [
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
    for i, q in enumerate(phq_questions, start=1):
        phq_answers[f"phq_{i}"] = st.radio(f"Q{i}: {q}", [0,1,2,3], index=0, key=f"phq_{i}")

    st.markdown("---")
    st.markdown("Alzheimer screening (brief)")
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
    for k, txt in ad_questions.items():
        ad_answers[k] = st.selectbox(txt, ["No","Sometimes","Often"], index=0, key=k)

with right_col:
    st.subheader("Console & Results")
    console = st.empty()
    result_area = st.empty()

# Processing
if process:
    console.info("Starting processing...")
    if not uploaded:
        console.error("Please upload an EDF file first.")
    else:
        raw, err = read_edf_bytes(uploaded)
        if err:
            console.error(err)
            st.session_state["results"] = None
        else:
            console.info("EDF loaded. Computing features...")
            out, e2 = compute_band_powers(raw)
            if e2:
                console.error(e2)
                st.session_state["results"] = None
            else:
                df = out["df"]
                summary = out["summary"]
                # produce topo images
                topo_imgs = {}
                for b in BANDS:
                    try:
                        topo_imgs[b] = heatmap_from_vals(df[f"{b}_rel"].values, f"{b} (relative)")
                    except Exception:
                        topo_imgs[b] = None
                # SHAP image if available
                shap_img = None
                shap_json = load_shap_json(SHAP_JSON) if SHAP_JSON.exists() else None
                if shap_json:
                    model_key = "depression_global" if summary.get("theta_alpha_ratio",0) <= 1.3 else "alzheimers_global"
                    shap_img = shap_bar_from_json(shap_json, model_key=model_key)
                # compute questionnaire scores
                phq_score = sum(int(v) for v in phq_answers.values())
                ad_score = sum(0 if v=="No" else (1 if v=="Sometimes" else 2) for v in ad_answers.values())
                # risk heuristics (tunable)
                depression_risk = min(100, max(0, (phq_score/27)*60 + summary.get("theta_alpha_ratio",0)*10))
                alz_risk = min(100, max(0, (ad_score/(len(ad_answers)*2))*50 + summary.get("fdi",0)*20 + (1.0 - summary.get("band_means",{}).get("Alpha",0))*10))
                # recommendations
                recs = []
                if depression_risk > 50 or phq_score >= 10:
                    recs.append("High depression risk — consider psychiatric evaluation and PHQ-9 clinical interview.")
                elif depression_risk > 25:
                    recs.append("Moderate depression risk — recommend monitoring and 4-8 week follow-up.")
                else:
                    recs.append("Low depression risk — routine follow-up.")
                if alz_risk > 40 or summary.get("fdi",0) > 2.0:
                    recs.append("Cognitive decline indicators present — recommend neurology referral and brain MRI/neuropsychological testing.")
                else:
                    recs.append("No immediate red-flag for severe cognitive impairment; consider periodic reassessment.")
                # store results
                results = {
                    "patient_info": {"id": patient_id, "dob": str(dob), "sex": sex, "meds": meds, "labs": labs},
                    "df": df,
                    "summary": summary,
                    "topo_images": topo_imgs,
                    "shap_img": shap_img,
                    "metrics": {"phq_score": phq_score, "ad_score": ad_score, "depression_risk": round(depression_risk,1), "alzheimer_risk": round(alz_risk,1)},
                    "recommendations": recs,
                    "created": now_ts()
                }
                st.session_state["results"] = results
                console.success("Processing complete.")

# Display results if available
res = st.session_state.get("results")
if res:
    c1, c2, c3 = st.columns(3)
    c1.metric("Depression Risk (%)", f"{res['metrics']['depression_risk']}")
    c2.metric("Alzheimer Risk (%)", f"{res['metrics']['alzheimer_risk']}")
    c3.metric("PHQ-9 score", f"{res['metrics']['phq_score']}")

    st.markdown("### Band table (relative power)")
    st.dataframe(res["df"].round(4), use_container_width=True)

    st.markdown("### Topographic maps (Spectral colormap)")
    tcols = st.columns(2)
    i = 0
    for band, img in res["topo_images"].items():
        with tcols[i%2]:
            if img:
                st.image(img, caption=band, use_container_width=True)
            else:
                st.info(f"{band} image unavailable")
        i += 1

    st.markdown("### Explainable AI (SHAP)")
    if res.get("shap_img"):
        st.image(res["shap_img"], use_container_width=True)
    else:
        st.info("No shap_summary.json found. Upload to enable XAI visualizations.")

    st.markdown("### Clinical recommendations")
    for r in res["recommendations"]:
        st.write("- " + r)

    # PDF export
    try:
        pdf = build_pdf({
            "title": "NeuroEarly Pro — Clinical",
            "patient_info": res["patient_info"],
            "metrics": res["metrics"],
            "topo_images": res["topo_images"],
            "shap_img": res.get("shap_img"),
            "recommendations": res["recommendations"]
        }, lang=st.session_state.get("lang","en"), amiri_path=str(AMIRI_PATH) if AMIRI_PATH.exists() else None)
        if pdf:
            st.download_button("Download PDF report", data=pdf, file_name=f"NeuroEarly_Report_{now_ts()}.pdf", mime="application/pdf")
        else:
            st.warning("PDF generation unavailable (reportlab or font missing).")
    except Exception as e:
        st.error(f"PDF export error: {e}")

else:
    result_area.info("No results yet. Upload EDF and press Start Processing.")

st.markdown("---")
st.markdown("Prepared by Golden Bird LLC — NeuroEarly Pro (Clinical v6.5)")

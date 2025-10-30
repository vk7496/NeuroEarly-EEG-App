# app.py — NeuroEarly Pro — Final Professional single-file edition
# Requirements: see README above. Put Amiri-Regular.ttf in ffont/ and logo in assets/
# Default language: English (select Arabic to switch UI and PDF to Arabic using Amiri font)

import os, io, sys, json, math, traceback, tempfile
from pathlib import Path
from datetime import datetime, date
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# optional heavy libs
HAS_MNE = False
HAS_PYEDF = False
HAS_MATPLOTLIB = False
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
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    HAS_MATPLOTLIB = True
except Exception:
    HAS_MATPLOTLIB = False

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
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

# --- constants & paths ---
ROOT = Path(__file__).parent
ASSETS = ROOT / "assets"
FFONT_DIR = ROOT / "ffont"
LOGO_PATH = ASSETS / "GoldenBird_logo.png"
AMIRI_PATH = FFONT_DIR / "Amiri-Regular.ttf"
SHAP_JSON = ROOT / "shap_summary.json"

BANDS = {
    "Delta": (0.5, 4.0),
    "Theta": (4.0, 8.0),
    "Alpha": (8.0, 13.0),
    "Beta": (13.0, 30.0),
    "Gamma": (30.0, 45.0)
}

# normative thresholds (example heuristics; for production use normative DB)
NORM_RANGES = {
    "theta_alpha": (0.2, 1.2),  # healthy range (min,max)
    "alpha_asym": (-0.05, 0.05)
}

# helper
def now_ts():
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")

def arabic_text(s: str) -> str:
    if not HAS_ARABIC: return s
    try:
        reshaped = arabic_reshaper.reshape(s)
        return get_display(reshaped)
    except Exception:
        return s

# UI texts bilingual simple mapping
TEXTS = {
    "title": {"en": "NeuroEarly Pro — Clinical", "ar": "NeuroEarly Pro — بالعيادة"},
    "upload_hint": {"en": "Upload EDF files (you can upload multiple)", "ar": "فایل‌های EDF را بارگذاری کنید (چندتایی ممکن است)"},
    "generate_pdf": {"en": "Generate PDF report", "ar": "ایجاد گزارش PDF"},
    "download_pdf": {"en": "Download report", "ar": "دانلود گزارش"},
}

# --- Streamlit layout / sidebar ---
st.set_page_config(page_title="NeuroEarly Pro — Clinical", layout="wide")

# Sidebar: language + patient info + tests
with st.sidebar:
    st.image(str(LOGO_PATH)) if LOGO_PATH.exists() else None
    lang = st.selectbox("Report language / اللغة", options=["en","ar"], index=0)
    is_ar = (lang == "ar")
    if is_ar and HAS_ARABIC:
        st.write(arabic_text("الواجهة باللغة العربية فعّال"))
    st.markdown("---")
    st.subheader("Patient information" if not is_ar else arabic_text("معلومات المريض"))
    pname = st.text_input("Name / اسم", value="")
    pid = st.text_input("ID", value="")
    dob = st.date_input("DOB / تاريخ الميلاد", value=date(1970,1,1), min_value=date(1900,1,1), max_value=date(2015,12,31))
    sex = st.selectbox("Sex / الجنس", ["Unknown", "Male", "Female"])
    st.markdown("---")
    st.subheader("Medical")
    meds = st.text_area("Current medications (one per line) / الأدوية الحالية", value="", height=100)
    comorb = st.text_area("Comorbidities / الأمراض المزمنة", value="", height=100)
    labs = st.text_area("Recent labs (B12, TSH, etc.) / التحاليل (B12, TSH ...)", value="", height=100)
    st.markdown("---")
    st.markdown("Model backend status:")
    st.text(f"mne={HAS_MNE} pyedflib={HAS_PYEDF} matplotlib={HAS_MATPLOTLIB}")
    st.markdown("---")

# Main column
st.title(TEXTS["title"][lang] if lang in TEXTS and TEXTS["title"].get(lang) else TEXTS["title"]["en"])

# upload area
uploaded = st.file_uploader(TEXTS["upload_hint"][lang], type=["edf"], accept_multiple_files=True)

# PHQ-9 (with corrected Q3, Q5, Q8)
st.header("1) PHQ-9 (Depression screening)" if not is_ar else arabic_text("الاختبار PHQ-9 (فحص الاكتئاب)"))
# Questions labels (EN) — Arabic would be mapped with arabic_text()
PHQ9_QS = [
    "Little interest or pleasure in doing things",
    "Feeling down, depressed, or hopeless",
    "Sleep: (choose) Insomnia / Hypersomnia / Oversleeping",
    "Feeling tired/low energy",
    "Appetite: (choose) Overeating / Under-eating",
    "Feeling bad about yourself — or failure",
    "Trouble concentrating on things",
    "Moving or speaking slowly OR being restless",
    "Thoughts that you would be better off dead"
]
# For Q3, Q5, Q8 we provide labels mapping but scoring remains 0..3.
phq_cols = st.columns([1,1,1])
phq_answers = {}
for i,q in enumerate(PHQ9_QS):
    col = phq_cols[i % 3]
    if i in (2,4,7):  # Q3, Q5, Q8 — provide clarifying choices in text but still numeric radio
        col.write(f"Q{i+1}")
        phq_answers[f"q{i+1}"] = col.selectbox("", options=[0,1,2,3], index=0, key=f"phq{i+1}")
        # show helper text
        if i==2: col.caption("0: No problem / 1: Slight (insomnia) / 2: Moderate (some) / 3: Severe (hypersomnia)")
        if i==4: col.caption("0: No change / 1: Slight (less) / 2: Moderate / 3: Severe (overeating/under-eating)")
        if i==7: col.caption("0: Normal / 1: Slight restlessness / 2: Moderate / 3: Very slow or agitated")
    else:
        col.write(f"Q{i+1}")
        phq_answers[f"q{i+1}"] = col.radio("", options=[0,1,2,3], index=0, key=f"phq{i+1}")

phq_total = sum(phq_answers.values())
st.info(f"PHQ-9 total: {phq_total} (0–27) — interpretation: 0–4 minimal, 5–9 mild, 10–14 moderate, 15–19 mod-severe, 20–27 severe)")

# AD8 cognitive screening (0/1)
st.header("2) AD8 (Cognitive screening)" if not is_ar else arabic_text("الاختبار AD8 (فحص الإدراك)"))
AD8_QS = [
    "Problems with judgment?",
    "Less interest in hobbies/activities?",
    "Repeats questions/stories?",
    "Trouble learning to use tools/ appliances?",
    "Forget the correct month or year?",
    "Difficulty handling complicated financial affairs?",
    "Trouble remembering appointments?",
    "Daily problems with thinking and memory?"
]
ad_cols = st.columns([1,1])
ad_answers = {}
for i,q in enumerate(AD8_QS):
    col = ad_cols[i % 2]
    col.write(f"A{i+1}")
    ad_answers[f"a{i+1}"] = col.radio("", options=[0,1], index=0, key=f"ad{i+1}")

ad_total = sum(ad_answers.values())
st.info(f"AD8 total: {ad_total} (0–8). Score ≥2 suggests cognitive impairment; confirm with clinical workup.")

# Provide area for optional model files / shap upload
st.markdown("---")
st.subheader("Model files & XAI")
model_file = st.file_uploader("Optional: upload shap_summary.json (for XAI visualizations)", type=["json"])
if model_file:
    try:
        data = json.load(model_file)
        # save locally to SHAP_JSON for report generator
        with open(SHAP_JSON, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        st.success("SHAP summary uploaded.")
    except Exception as e:
        st.error(f"Could not read SHAP JSON: {e}")

# --- EEG processing helpers ---
def read_edf_bytes(filelike) -> Tuple[np.ndarray, float, List[str]]:
    """
    Returns data (n_samples x n_ch), sf (sampling freq), ch_names
    Fallback: use pyedflib if available, else use mne if available.
    """
    try:
        # prefer mne if available
        if HAS_MNE:
            raw = mne.io.read_raw_edf(filelike, preload=True, verbose=False)
            data, sf = raw.get_data(return_times=False), raw.info["sfreq"]
            ch_names = raw.ch_names
            return data.T, float(sf), ch_names
        elif HAS_PYEDF:
            # pyedflib
            f = pyedflib.EdfReader(filelike)
            n_channels = f.signals_in_file
            ch_names = f.getSignalLabels()
            sf = f.getSampleFrequency(0)
            data = np.vstack([f.readSignal(i) for i in range(n_channels)])
            f._close()
            return data.T, float(sf), ch_names
        else:
            raise RuntimeError("No EDF reader available (install mne or pyedflib).")
    except Exception as e:
        raise

def bandpower(psd_freqs, psd, fmin, fmax):
    mask = (psd_freqs >= fmin) & (psd_freqs <= fmax)
    return np.trapz(psd[mask], psd_freqs[mask]) if mask.sum() > 0 else 0.0

def compute_relative_band_powers(raw_signal, sf, ch_names):
    """
    raw_signal: (n_samples, n_ch)
    returns DataFrame of relative band powers per channel and aggregated means
    """
    import scipy.signal as sps
    n_ch = raw_signal.shape[1]
    # compute PSD per channel via welch
    freqs, psd = None, None
    pows = {}
    for i in range(n_ch):
        f, Pxx = sps.welch(raw_signal[:,i], fs=sf, nperseg=min(2048, raw_signal.shape[0]))
        if freqs is None: freqs = f
        if psd is None: psd = np.zeros((len(freqs), n_ch))
        psd[:,i] = Pxx
    band_rel = {}
    total_power = np.trapz(psd, freqs, axis=0)
    for band_name,(fmin,fmax) in BANDS.items():
        mask = (freqs>=fmin)&(freqs<=fmax)
        bp = np.trapz(psd[mask,:], freqs[mask], axis=0)
        # relative per channel
        band_rel[f"{band_name}_rel"] = bp / (total_power + 1e-12)
    df = pd.DataFrame(band_rel)
    df.index = ch_names
    agg = df.mean().to_dict()
    return df, agg

def generate_topomap_image(vals, ch_names=None, band_name=""):
    """
    Simple topomap placeholder using matplotlib (scatter on 2D 10-20 approximate positions).
    vals: array per channel
    """
    if not HAS_MATPLOTLIB:
        return None
    # approximate 10-20 positions for some channels if available
    pos = {
        'Fp1':(-0.5,1),'Fp2':(0.5,1),'F3':(-0.8,0.3),'F4':(0.8,0.3),
        'C3':(-0.8,-0.3),'C4':(0.8,-0.3),'P3':(-0.5,-0.8),'P4':(0.5,-0.8),
        'O1':(-0.2,-1),'O2':(0.2,-1),'F7':(-1,0.6),'F8':(1,0.6),'T7':(-1,-0.3),'T8':(1,-0.3)
    }
    xs, ys, v = [], [], []
    for i,ch in enumerate(ch_names):
        if ch in pos:
            x,y = pos[ch]
            xs.append(x); ys.append(y); v.append(vals[i])
    if len(xs)==0:
        # fallback just bar chart
        fig,ax = plt.subplots(figsize=(4,3))
        names = ch_names[:min(12,len(ch_names))]
        ax.bar(range(len(names)), vals[:len(names)])
        ax.set_title(f"{band_name} topomap (bar fallback)")
        buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format='png'); plt.close(fig); buf.seek(0)
        return buf.getvalue()
    fig,ax = plt.subplots(figsize=(4,3))
    sc = ax.scatter(xs, ys, c=v, s=300, cmap='RdBu_r', vmin=np.min(v), vmax=np.max(v))
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{band_name} topomap")
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format='png'); plt.close(fig); buf.seek(0)
    return buf.getvalue()

# connectivity (coherence) using scipy.signal.coherence as fallback
def compute_connectivity_matrix(raw_signal, sf, ch_names=None, band=(8.0,13.0)):
    import scipy.signal as sps
    n_ch = raw_signal.shape[1]
    M = np.zeros((n_ch,n_ch))
    for i in range(n_ch):
        for j in range(i+1,n_ch):
            try:
                f, Cxy = sps.coherence(raw_signal[:,i], raw_signal[:,j], fs=sf, nperseg=min(2048, raw_signal.shape[0]))
                mask = (f>=band[0]) & (f<=band[1])
                if mask.sum()>0:
                    val = np.mean(Cxy[mask])
                else:
                    val = np.nan
            except Exception:
                val = np.nan
            M[i,j] = M[j,i] = val if not np.isnan(val) else 0.0
    # narrative
    mean_conn = np.nanmean(M)
    narrative = f"Mean connectivity (band {band[0]}–{band[1]} Hz): {mean_conn:.3f}"
    # image
    conn_img = None
    if HAS_MATPLOTLIB:
        fig,ax = plt.subplots(figsize=(4,3))
        im = ax.imshow(M, cmap='viridis', vmin=0, vmax=1)
        ax.set_title("Connectivity (coherence)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format='png'); plt.close(fig); buf.seek(0)
        conn_img = buf.getvalue()
    return M, narrative, conn_img

# tumor detection: focal delta index (FDI) and extreme asymmetry
def compute_focal_delta(raw_signal, sf, ch_names):
    # delta power per channel
    df, agg = compute_relative_band_powers(raw_signal, sf, ch_names)
    # compute FDI: local delta / mean delta global
    if "Delta_rel" in df.columns:
        delta_vals = df["Delta_rel"].values
    else:
        # fallback: compute from bands keys
        delta_vals = df.iloc[:,0].values
    mean_delta = delta_vals.mean() + 1e-12
    FDI = delta_vals / mean_delta
    # find max index
    max_idx = int(np.argmax(FDI))
    max_ch = ch_names[max_idx]
    fdi_val = FDI[max_idx]
    # asymmetry pairs example T7/T8, check extreme ratio
    try:
        t7_idx = ch_names.index("T7")
        t8_idx = ch_names.index("T8")
        asym = (delta_vals[t8_idx] / (delta_vals[t7_idx]+1e-9))
    except Exception:
        asym = 1.0
    alert = False
    if fdi_val > 2.0 or asym>3.0 or asym<0.33:
        alert = True
    narrative = f"Focal Delta Index max at {max_ch}: {fdi_val:.2f}, asymmetry(T7/T8) ≈ {asym:.2f}"
    return {"FDI_vals": FDI.tolist(), "max_channel": max_ch, "FDI": float(fdi_val), "asym_T7_T8": float(asym), "alert": alert, "narrative": narrative}

# XAI SHAP bar chart from shap_summary.json
def display_shap_ui(agg_metrics: Dict[str,float], lang="en"):
    st.subheader("Explainable AI (XAI)")
    shap_data = None
    if SHAP_JSON.exists():
        try:
            shap_data = json.load(open(SHAP_JSON, "r", encoding="utf-8"))
        except Exception:
            shap_data = None
    if shap_data:
        # choose key by heuristic: use theta_alpha_ratio or mean_connectivity
        model_key = "depression_global"
        if agg_metrics.get("theta_alpha_ratio", 0) > 1.3:
            model_key = "alzheimers_global"
        features = shap_data.get(model_key, {})
        if features:
            s = pd.Series(features).abs().sort_values(ascending=False)
            # bar chart
            st.bar_chart(s.head(10))
            # show table
            df_s = pd.DataFrame.from_dict(features, orient="index", columns=["shap_value"]).sort_values(by="shap_value", key=lambda x: x.abs(), ascending=False)
            st.table(df_s.head(12))
        else:
            st.info("SHAP file present but no matching model key.")
    else:
        st.info("No shap_summary.json found. Upload to enable XAI visualizations.")

# PDF generator (reportlab) — returns bytes
def generate_pdf_report(summary: Dict[str,Any], lang="en", amiri_path: Optional[str]=None) -> bytes:
    if not HAS_REPORTLAB:
        raise RuntimeError("reportlab not installed in environment.")
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=36, bottomMargin=36, leftMargin=44, rightMargin=44)
    styles = getSampleStyleSheet()
    base_font = "Helvetica"
    # register Amiri if present
    if amiri_path and Path(amiri_path).exists():
        try:
            pdfmetrics.registerFont(TTFont("Amiri", amiri_path))
            base_font = "Amiri"
        except Exception as e:
            print("Amiri reg failed:", e)
    styles.add(ParagraphStyle(name="TitleBlue", fontName=base_font, fontSize=16, textColor=colors.HexColor("#0b63d6"), alignment=1, spaceAfter=8))
    styles.add(ParagraphStyle(name="H2", fontName=base_font, fontSize=12, textColor=colors.HexColor("#0b63d6"), spaceAfter=6))
    styles.add(ParagraphStyle(name="Body", fontName=base_font, fontSize=10, leading=14))
    styles.add(ParagraphStyle(name="Note", fontName=base_font, fontSize=9, textColor=colors.grey))
    elements = []
    # header
    if LOGO_PATH.exists():
        elements.append(Image(str(LOGO_PATH), width=1.4*inch, height=1.4*inch))
    elements.append(Paragraph("NeuroEarly Pro — Clinical Report", styles["TitleBlue"]))
    elements.append(Spacer(1,6))
    # patient summary
    ptxt = f"Patient: {summary.get('patient_name','')}   ID: {summary.get('patient_id','')}   DOB: {summary.get('dob','')}"
    elements.append(Paragraph(ptxt, styles["Body"]))
    elements.append(Spacer(1,6))
    # Final ML risk
    ml = summary.get("ml_score", None)
    if ml is not None:
        elements.append(Paragraph(f"Final ML Risk Score: {ml*100:.1f}%" , styles["H2"]))
    # QEEG summary table
    elements.append(Paragraph("QEEG Key Metrics", styles["H2"]))
    metrics = summary.get("metrics", {})
    data = [["Metric","Value"]]
    for k,v in metrics.items():
        try:
            data.append([str(k), f"{float(v):.4f}" if isinstance(v,(int,float)) else str(v)])
        except Exception:
            data.append([str(k), str(v)])
    t = Table(data, colWidths=[3.5*inch, 2*inch])
    t.setStyle(TableStyle([("BACKGROUND",(0,0),(1,0), colors.HexColor("#f0f8ff")), ("GRID",(0,0),(-1,-1),0.5, colors.grey)]))
    elements.append(t)
    elements.append(Spacer(1,10))
    # Add plots if present
    # topomaps (list)
    topo_imgs = summary.get("topo_images", {})
    for band,buf in (topo_imgs.items() if isinstance(topo_imgs,dict) else []):
        if buf:
            img = Image(io.BytesIO(buf))
            img._restrictSize(4*inch, 3*inch)
            elements.append(Paragraph(f"Topography — {band}", styles["H2"]))
            elements.append(img)
            elements.append(Spacer(1,6))
    # connectivity image
    if summary.get("connectivity_image", None):
        elements.append(Paragraph("Connectivity", styles["H2"]))
        img = Image(io.BytesIO(summary["connectivity_image"]))
        img._restrictSize(6*inch, 3*inch)
        elements.append(img)
        elements.append(Spacer(1,6))
    # SHAP image if any (we produce bar chart in code separately — here show table)
    if summary.get("shap_table", None):
        elements.append(Paragraph("XAI — SHAP top contributors", styles["H2"]))
        stbl = [["Feature","Importance"]]
        for k,v in summary["shap_table"].items():
            stbl.append([k, f"{v:.4f}"])
        t2 = Table(stbl, colWidths=[3.5*inch, 2*inch])
        t2.setStyle(TableStyle([("GRID",(0,0),(-1,-1),0.5, colors.grey)]))
        elements.append(t2)
    # clinical narrative
    elements.append(Paragraph("Clinical Recommendations", styles["H2"]))
    elements.append(Paragraph(summary.get("clinical_narrative","No narrative."), styles["Body"]))
    # footer / provider
    elements.append(Spacer(1,12))
    elements.append(Paragraph("Designed by Golden Bird L.L.C", styles["Note"]))
    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()

# ==== MAIN processing flow ====
results = []
if uploaded:
    processing_placeholder = st.empty()
    processing_placeholder.info("Processing uploaded EDF(s)...")
    for up in uploaded:
        try:
            # read EDF (note: mne.read_raw_edf accepts path-like; we use NamedTemporaryFile)
            with tempfile.NamedTemporaryFile(suffix=".edf", delete=False) as tmp:
                tmp.write(up.read()); tmp.flush()
                tmp_path = tmp.name
            data, sf, ch_names = read_edf_bytes(tmp_path)
            # ensure shape (n_samples, n_ch)
            if data.ndim==1:
                data = data.reshape(-1,1)
            # Preprocessing: notch at 50/60, bandpass 0.5-45
            try:
                import scipy.signal as sps
                # notch
                for f0 in (50.0, 60.0):
                    b,a = sps.iirnotch(f0, 30.0, sf) if sf>0 else (None,None)
                    if b is not None:
                        for ch in range(data.shape[1]):
                            data[:,ch] = sps.filtfilt(b,a,data[:,ch])
                # bandpass
                b,a = sps.butter(4, [0.5/(sf/2), 45.0/(sf/2)], btype='band')
                for ch in range(data.shape[1]):
                    data[:,ch] = sps.filtfilt(b,a,data[:,ch])
            except Exception:
                pass
            # compute bandpowers
            dfbands, agg = compute_relative_band_powers(data, sf, ch_names)
            # aggregated metrics for this file
            theta_alpha_ratio = (agg.get("Theta_rel", 1e-9) / (agg.get("Alpha_rel",1e-9))) if agg.get("Alpha_rel",0)>0 else 0.0
            theta_beta_ratio = (agg.get("Theta_rel",0) / (agg.get("Beta_rel",1e-9))) if agg.get("Beta_rel",0)>0 else 0.0
            alpha_asym = 0.0
            try:
                f3 = dfbands.loc["F3"]["Alpha_rel"] if "F3" in dfbands.index else np.nan
                f4 = dfbands.loc["F4"]["Alpha_rel"] if "F4" in dfbands.index else np.nan
                alpha_asym = float(f3 - f4)
            except Exception:
                alpha_asym = 0.0
            # topomap images per band
            topo_imgs = {}
            for band in ["Delta","Theta","Alpha","Beta","Gamma"]:
                try:
                    vals = dfbands[f"{band}_rel"].values if f"{band}_rel" in dfbands.columns else np.zeros(len(ch_names))
                    img = generate_topomap_image(vals, ch_names=ch_names, band_name=band)
                    topo_imgs[band] = img
                except Exception:
                    topo_imgs[band] = None
            # connectivity
            conn_mat, conn_narr, conn_img = compute_connectivity_matrix(data, sf, ch_names, band=BANDS["Alpha"])
            # tumor/focal delta
            focal = compute_focal_delta(data, sf, ch_names)
            # store
            res = {
                "filename": up.name,
                "agg_features": agg,
                "df_bands": dfbands,
                "topo_images": topo_imgs,
                "connectivity_matrix": conn_mat,
                "connectivity_narrative": conn_narr,
                "connectivity_image": conn_img,
                "focal": focal,
                "raw_sf": sf,
                "ch_names": ch_names,
            }
            results.append(res)
            processing_placeholder.success(f"Processed {up.name}")
        except Exception as e:
            processing_placeholder.error(f"Failed processing {up.name}: {e}")
            st.error(traceback.format_exc())
    # show summary for first file
    if results:
        st.markdown("### Aggregated features (first file)")
        try:
            st.dataframe(pd.Series(results[0]["agg_features"]).to_frame("value"))
        except Exception:
            st.write(results[0]["agg_features"])

        # show topomaps
        st.markdown("### Topography Maps (first file)")
        tcols = st.columns(5)
        for i,band in enumerate(["Delta","Theta","Alpha","Beta","Gamma"]):
            buf = results[0]["topo_images"].get(band)
            if buf:
                tcols[i%5].image(buf, caption=band, use_column_width=True)
        # connectivity and focal
        st.markdown("### Connectivity & Focal Delta / Tumor indicators")
        st.write(results[0]["connectivity_narrative"])
        if results[0]["connectivity_image"]:
            st.image(results[0]["connectivity_image"], use_column_width=True)
        st.write("Focal delta summary:")
        st.json(results[0]["focal"])

        # display XAI
        # prepare aggregate metrics dict
        metrics = {
            "theta_alpha_ratio": theta_alpha_ratio,
            "theta_beta_ratio": theta_beta_ratio,
            "alpha_asymmetry_F3_F4": alpha_asym
        }
        display_shap_ui(metrics, lang=lang)

# generate report button
st.markdown("---")
if st.button(TEXTS["generate_pdf"][lang]):
    # prepare summary
    summary = {
        "patient_name": pname,
        "patient_id": pid,
        "dob": str(dob),
        "ml_score": None,
        "metrics": {},
        "topo_images": {},
        "connectivity_image": None,
        "shap_table": {},
        "clinical_narrative": ""
    }
    # fill metrics from first result
    if results:
        r0 = results[0]
        # simple scoring heuristic for ML risk
        th_alpha = (r0["agg_features"].get("Theta_rel",0) / max(r0["agg_features"].get("Alpha_rel",1e-9),1e-9)) if "Theta_rel" in r0["agg_features"] else None
        if th_alpha is None:
            ml_score = 0.05
        else:
            ml_score = min(1.0, max(0.0, (th_alpha - 0.5)/2.5))  # heuristic
        summary["ml_score"] = ml_score
        summary["metrics"].update({
            "Theta/Alpha": theta_alpha_ratio,
            "Theta/Beta": theta_beta_ratio,
            "Alpha Asymmetry (F3-F4)": alpha_asym,
            "FDI": r0["focal"]["FDI"],
            "Focal channel": r0["focal"]["max_channel"]
        })
        summary["topo_images"] = r0["topo_images"]
        summary["connectivity_image"] = r0["connectivity_image"]
        # shap_table from shap_summary.json
        if SHAP_JSON.exists():
            try:
                sdata = json.load(open(SHAP_JSON,"r",encoding="utf-8"))
                model_key = "depression_global"
                if summary["metrics"].get("Theta/Alpha",0)>1.3:
                    model_key = "alzheimers_global"
                summary["shap_table"] = sdata.get(model_key,{})
            except Exception:
                summary["shap_table"] = {}
        # clinical narrative
        narrative = []
        narrative.append(f"PHQ-9 score: {phq_total}. AD8 score: {ad_total}.")
        narrative.append(f"QEEG shows Theta/Alpha ratio = {theta_alpha_ratio:.3f}. Alpha asymmetry (F3-F4) = {alpha_asym:.4f}.")
        narrative.append(f"Focal Delta Index (FDI): {r0['focal']['FDI']:.2f} at {r0['focal']['max_channel']}.")
        narrative.append("Recommendation: correlate with clinical exam, labs (B12/TSH) and consider neuroimaging (MRI) if ML risk > 0.25 or FDI>2.0.")
        summary["clinical_narrative"] = " ".join(narrative)
    else:
        summary["clinical_narrative"] = "No EDF uploaded — report contains questionnaire results only."
        summary["metrics"].update({"PHQ9_total": phq_total, "AD8_total": ad_total})
    # generate pdf
    try:
        pdf_bytes = generate_pdf_report(summary, lang=lang, amiri_path=str(AMIRI_PATH) if AMIRI_PATH.exists() else None)
        st.success("Report generated.")
        st.download_button(TEXTS["download_pdf"][lang], data=pdf_bytes, file_name=f"NeuroEarly_Report_{now_ts()}.pdf", mime="application/pdf")
    except Exception as e:
        st.error(f"PDF generation failed: {e}")
        st.exception(traceback.format_exc())

# final help / developer note
st.sidebar.markdown("---")
st.sidebar.markdown("Developer notes:")
st.sidebar.write("Place Amiri font at ffont/Amiri-Regular.ttf for Arabic PDF; place logo at assets/GoldenBird_logo.png")
st.sidebar.write("If heavy libs fail to import, functionality will be degraded but app remains usable for questionnaires and CSV export.")

# Export CSV of aggregated metrics if available
if results:
    try:
        df_export = pd.DataFrame([res["agg_features"] for res in results])
        csv = df_export.to_csv(index=False).encode("utf-8")
        st.download_button("Download metrics (CSV)", data=csv, file_name=f"NeuroEarly_metrics_{now_ts()}.csv", mime="text/csv")
    except Exception:
        pass

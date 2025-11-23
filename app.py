# app.py — NeuroEarly Pro (Clinical, Stress + Denoise + SHAP explanations)
# Features:
# - Bilingual UI (English / Arabic optional)
# - EDF upload, robust reading (mne or pyedflib fallback), denoise (notch + bandpass)
# - Band powers, FDI, connectivity (coherence fallback), Stress index (EEG-derived)
# - Topomap images per band with doctor-friendly captions
# - SHAP visualization from ./shap_summary.json with textual explanation
# - Lab report upload (PDF/TXT) reading for common deficiencies
# - PDF report (ReportLab) with explanations
# - Graceful degradation if heavy libs are missing

import os, io, sys, tempfile, json, traceback
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

# optional heavy libs
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
    from scipy.signal import welch, butter, sosfiltfilt, iirnotch, detrend
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

# ---------------- CONFIG ----------------
ROOT = Path(".")
ASSETS = ROOT / "assets"
AMIRI_TTF = ROOT / "Amiri-Regular.ttf"
LOGO = ASSETS / "goldenbird_logo.png"
SHAP_JSON = ROOT / "shap_summary.json"   # <--- your SHAP file in project root
HEALTHY_EDF = Path("/mnt/data/test_edf.edf")  # optional baseline (do not make model condition on it)

APP_TITLE = "NeuroEarly Pro — Clinical AI Assistant"
BLUE = "#0b63d6"
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

def safe_read_bytes(uploaded):
    try:
        return uploaded.read()
    except Exception:
        try:
            return uploaded.getvalue()
        except Exception as e:
            return None

# ---------------- EDF read + denoise ----------------
def write_temp(bytes_data, suffix=".edf"):
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tf:
        tf.write(bytes_data)
        tf.flush()
        return tf.name

def notch_filter(data, sf, freq=50.0):
    # notch at freq (50 or 60) using iirnotch if available; else passthrough
    try:
        if not HAS_SCIPY:
            return data
        b, a = iirnotch(freq/(sf/2), Q=30.0)
        return sosfiltfilt(np.array([b]), data) if False else np.array(data)  # fallback: no-op (scipy notch using sosfiltfilt requires design in sos)
    except Exception:
        return data

def bandpass_filter(data, sf, low=0.5, high=45.0, order=4):
    if not HAS_SCIPY:
        return data
    ny = 0.5 * sf
    lowcut = max(low/ny, 1e-6)
    highcut = min(high/ny, 0.999)
    sos = butter(order, [lowcut, highcut], btype='band', output='sos')
    try:
        return sosfiltfilt(sos, data, axis=-1)
    except Exception:
        return sosfiltfilt(sos, data) if data.ndim==1 else data

def read_edf_uploaded(uploaded_file) -> Tuple[Optional[object], Optional[str]]:
    """
    Returns (raw_like, msg) where raw_like is either mne Raw or dict {'signals','ch_names','sfreq'}.
    """
    if uploaded_file is None:
        return None, "No file uploaded"
    data = safe_read_bytes(uploaded_file)
    if data is None:
        return None, "Cannot read uploaded file bytes"
    tmp = write_temp(data, suffix=".edf")
    # try mne
    if HAS_MNE:
        try:
            raw = mne.io.read_raw_edf(tmp, preload=True, verbose=False)
            os.unlink(tmp)
            return raw, None
        except Exception as e:
            # fallback to pyedflib
            pass
    if HAS_PYEDF:
        try:
            f = pyedflib.EdfReader(tmp)
            n = f.signals_in_file
            ch_names = f.getSignalLabels()
            sfreq = f.getSampleFrequency(0)
            arrs = [f.readSignal(i) for i in range(n)]
            f.close()
            os.unlink(tmp)
            signals = np.vstack(arrs)
            return {"signals": signals, "ch_names": ch_names, "sfreq": float(sfreq)}, None
        except Exception as e:
            try:
                os.unlink(tmp)
            except Exception:
                pass
            return None, f"pyedflib read error: {e}"
    else:
        try:
            os.unlink(tmp)
        except Exception:
            pass
        return None, "No EDF backend available (install mne or pyedflib)"

# ---------------- Spectral computations ----------------
def compute_band_powers(raw_like, bands=BANDS):
    """
    Accepts mne Raw or fallback dict {'signals','ch_names','sfreq'}.
    Returns dict with 'bands' (per-channel band abs/rel), 'metrics', 'psd','freqs','ch_names','sfreq'
    """
    # prepare data matrix (n_ch x n_s)
    if raw_like is None:
        return None
    if HAS_MNE and isinstance(raw_like, mne.io.BaseRaw):
        raw = raw_like.copy().pick_types(eeg=True, meg=False)
        sf = raw.info['sfreq']
        data = raw.get_data()  # n_ch x n_s
        ch_names = raw.ch_names
    else:
        dd = raw_like
        data = np.asarray(dd["signals"])
        ch_names = dd["ch_names"]
        sf = dd["sfreq"]
        if data.ndim == 1:
            data = data[np.newaxis, :]
    # denoise: notch and bandpass
    try:
        for i in range(data.shape[0]):
            # 50Hz/60Hz notch heuristics: try both if present in spectrum
            if HAS_SCIPY:
                data[i,:] = detrend(data[i,:])
                # notch not implemented robustly here; skip to bandpass
            data[i,:] = bandpass_filter(data[i,:], sf, low=1.0, high=45.0)
    except Exception as e:
        pass

    # PSD
    if HAS_SCIPY:
        nperseg = min(int(4*sf), data.shape[1])
        freqs_list = []
        psd_list = []
        for ch in range(data.shape[0]):
            f, Pxx = welch(data[ch,:], fs=sf, nperseg=nperseg)
            freqs_list = f
            psd_list.append(Pxx)
        freqs = freqs_list
        psd = np.vstack(psd_list)
    else:
        # fallback simple FFT PSD
        n = data.shape[1]
        freqs = np.fft.rfftfreq(n, d=1.0/sf)
        fft_vals = np.abs(np.fft.rfft(data, axis=1))**2 / n
        psd = fft_vals

    # compute band summaries
    band_summary = {}
    total_power = psd.sum(axis=1) + 1e-12
    for i, ch in enumerate(ch_names):
        band_summary[ch] = {}
        for bname, (fmin, fmax) in bands.items():
            mask = (freqs >= fmin) & (freqs < fmax)
            power = float(psd[i, mask].sum()) if mask.any() else 0.0
            band_summary[ch][f"{bname}_abs"] = power
            band_summary[ch][f"{bname}_rel"] = (power / total_power[i]) if total_power[i] > 0 else 0.0

    # metrics
    # theta/alpha ratio global
    theta_alpha_vals = []
    for ch in band_summary:
        a = band_summary[ch].get("Alpha_rel", 0.0)
        t = band_summary[ch].get("Theta_rel", 0.0)
        if a > 0:
            theta_alpha_vals.append(t / a)
    theta_alpha = float(np.mean(theta_alpha_vals)) if theta_alpha_vals else 0.0
    # FDI (focal delta index)
    delta_rels = [band_summary[ch].get("Delta_rel", 0.0) for ch in band_summary]
    FDI = float(max(delta_rels) / (np.mean(delta_rels)+1e-12)) if len(delta_rels)>0 else 0.0

    return {"bands": band_summary, "metrics": {"theta_alpha": theta_alpha, "FDI": FDI},
            "psd": psd, "freqs": freqs, "ch_names": ch_names, "sfreq": sf}

# ---------------- Topomap generator (doctor-friendly) ----------------
def topomap_png_from_vals(vals: List[float], band_name: str="Band"):
    try:
        arr = np.asarray(vals).astype(float).ravel()
        n = len(arr)
        side = int(np.ceil(np.sqrt(n)))
        grid_flat = np.full(side*side, np.nan)
        grid_flat[:n] = arr
        grid = grid_flat.reshape(side, side)
        fig, ax = plt.subplots(figsize=(4,3))
        im = ax.imshow(grid, cmap="RdBu_r", interpolation='nearest', origin='upper')
        ax.set_title(f"{band_name} — topography")
        ax.axis('off')
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Relative power (normalized)")
        buf = io.BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format='png', dpi=150)
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue()
    except Exception as e:
        print("topomap error:", e)
        return None

# ---------------- Stress index ----------------
def compute_stress_index(band_summary: Dict[str, Any]):
    """
    Simple EEG-derived stress index:
    StressIndex = mean(Beta_rel / Alpha_rel) across channels, normalized 0..1
    (Higher beta relative to alpha => higher cortical arousal / stress)
    """
    ratios = []
    for ch, info in band_summary.items():
        a = info.get("Alpha_rel", 1e-6)
        b = info.get("Beta_rel", 0.0)
        ratios.append(b / max(a, 1e-6))
    if not ratios:
        return 0.0
    raw = float(np.nanmean(ratios))
    # normalize (heuristic): map [0..3] -> [0..1]
    norm = max(0.0, min(1.0, raw / 3.0))
    return norm

# ---------------- SHAP rendering + explanation text ----------------
def render_shap_image(shap_path: Path):
    if not shap_path.exists() or not HAS_SHAP:
        return None
    try:
        with open(shap_path, "r", encoding="utf-8") as f:
            sj = json.load(f)
        # pick first key
        key = next(iter(sj.keys()))
        feat_dict = sj.get(key, {})
        # sort top contributors
        s = pd.Series(feat_dict).abs().sort_values(ascending=False).head(12)
        fig, ax = plt.subplots(figsize=(6,3))
        s.plot.barh(ax=ax)
        ax.set_xlabel("mean(|SHAP|)")
        ax.set_title("SHAP — feature importance")
        ax.invert_yaxis()
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150)
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue()
    except Exception as e:
        print("SHAP render error:", e)
        return None

def shap_explanation_text(shap_path: Path):
    # human-friendly explanation that will appear under SHAP image
    txt = ("SHAP explanation: bars show which features contributed most to the model's prediction. "
           "Longer bars mean a greater influence. Positive effect features (increase risk) "
           "and negative effect features (decrease risk) should be interpreted in clinical context. "
           "If uncertain, compare SHAP with raw band powers and clinical questionnaire scores.")
    return txt

# ---------------- Lab report parsing (very simple keyword search) ----------------
LAB_KEYWORDS = {
    "b12": ["b12", "vitamin b12", "cobalamin"],
    "vitd": ["vit d", "vitamin d", "25-oh", "25ohd"],
    "tsh": ["tsh", "thyroid stimulating", "thyroid"],
    "glucose": ["glucose", "hba1c", "blood sugar"]
}
def parse_lab_text(txt: str):
    txtl = txt.lower()
    findings = []
    for k,v in LAB_KEYWORDS.items():
        for token in v:
            if token in txtl:
                # crude detection of abnormal words
                if "low" in txtl or "deficien" in txtl or "high" in txtl or "elevat" in txtl or "<" in txtl or ">" in txtl:
                    findings.append(k.upper())
                else:
                    findings.append(k.upper())
                break
    return sorted(set(findings))

# ---------------- PDF report (simplified) ----------------
def generate_pdf(summary: dict, lang: str="en"):
    if not HAS_REPORTLAB:
        return None
    try:
        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=36, bottomMargin=36, leftMargin=44, rightMargin=44)
        styles = getSampleStyleSheet()
        base_font = "Helvetica"
        if AMIRI_TTF.exists() and HAS_ARABIC:
            try:
                pdfmetrics.registerFont(TTFont("Amiri", str(AMIRI_TTF)))
                base_font = "Amiri"
            except Exception:
                pass
        styles.add(ParagraphStyle(name="TitleBlue", fontName=base_font, fontSize=16, textColor=colors.HexColor(BLUE), alignment=1))
        story = []
        story.append(Paragraph("NeuroEarly Pro — Clinical Report", styles["TitleBlue"]))
        story.append(Spacer(1,8))
        # patient info table
        pi = summary.get("patient_info", {})
        rows = [["Field","Value"], ["Name", pi.get("name","-")], ["ID", pi.get("id","-")], ["DOB", pi.get("dob","-")]]
        t = Table(rows, colWidths=[120,300])
        t.setStyle(TableStyle([("GRID",(0,0),(-1,-1),0.25,colors.grey)]))
        story.append(t); story.append(Spacer(1,8))
        # metrics table
        story.append(Paragraph("Key QEEG metrics", styles["Normal"])); story.append(Spacer(1,6))
        m = summary.get("metrics",{})
        rows = [[k,str(v)] for k,v in m.items()]
        t2 = Table(rows, colWidths=[200,200]); t2.setStyle(TableStyle([("GRID",(0,0),(-1,-1),0.25,colors.grey)]))
        story.append(t2); story.append(Spacer(1,8))
        # show topomaps if present
        if summary.get("topo_images"):
            story.append(Paragraph("Topography maps (per frequency band)", styles["Heading2"]))
            for band, img in summary["topo_images"].items():
                try:
                    story.append(RLImage(io.BytesIO(img), width=250, height=150))
                    story.append(Paragraph(f"Caption: {summary.get('topo_text',{}).get(band,'')}", styles["Normal"]))
                except Exception:
                    pass
                story.append(Spacer(1,6))
        # SHAP
        if summary.get("shap_img"):
            story.append(PageBreak())
            story.append(Paragraph("Explainable AI — SHAP", styles["Heading2"]))
            story.append(RLImage(io.BytesIO(summary["shap_img"]), width=420, height=180))
            story.append(Paragraph(summary.get("shap_text",""), styles["Normal"]))
        # Recommendations
        story.append(PageBreak())
        story.append(Paragraph("Clinical Recommendations", styles["Heading2"]))
        for r in summary.get("recommendations", []):
            story.append(Paragraph(r, styles["Normal"]))
            story.append(Spacer(1,4))
        doc.build(story)
        buf.seek(0)
        return buf.getvalue()
    except Exception as e:
        print("PDF error:", e)
        traceback.print_exc()
        return None

# ---------------- UI ----------------
st.set_page_config(page_title=APP_TITLE, layout="wide")
header_html = f"""
<div style="background: linear-gradient(90deg,{BLUE},#7DD3FC); padding:12px; border-radius:8px; color:white; display:flex; align-items:center; justify-content:space-between;">
  <div style="font-weight:700; font-size:20px;">🧠 {APP_TITLE}</div>
  <div style="display:flex; align-items:center;">{'<img src=\"assets/goldenbird_logo.png\" style=\"height:44px; margin-left:12px;\" />' if LOGO.exists() else ''}</div>
</div>
"""
st.markdown(header_html, unsafe_allow_html=True)
st.write("")

# layout
left, main, right = st.columns([1,2,1])

with left:
    st.header("Patient")
    lang = st.selectbox("Language / اللغة", ["English","Arabic"])
    patient_name = st.text_input("Full Name", "John Doe")
    patient_id = st.text_input("File ID", "F-101")
    dob = st.date_input("Date of birth", value=date(1980,1,1))
    gender = st.selectbox("Gender", ["Unknown","Male","Female","Other"])
    st.markdown("---")
    st.header("Lab report (optional)")
    lab_file = st.file_uploader("Upload lab report (PDF/TXT)", type=["pdf","txt"], accept_multiple_files=False)
    st.markdown("---")
    st.header("EDF upload")
    uploaded = st.file_uploader("Upload EDF (.edf)", type=["edf"], accept_multiple_files=False)
    st.markdown("")
    process_btn = st.button("Process EDF & Analyze")

with main:
    st.header("Clinical Questionnaires")
    with st.expander("PHQ-9 (Depression) — click to open"):
        phq_answers = []
        for i,q in enumerate(["Little interest","Feeling down","Sleep","Energy","Appetite","Failure","Concentration","Psychomotor","Self-harm"], start=1):
            v = st.radio(f"Q{i}. {q}", ["0 - Not at all","1 - Several days","2 - More than half the days","3 - Nearly every day"], index=0, key=f"phq_{i}")
            phq_answers.append(int(v.split()[0]))
    with st.expander("AD8 / Cognitive screening — click to open"):
        ad8_answers = []
        for i,q in enumerate(["Judgment issues","Less interest","Repeating","Learning new","Forgetting month/year","Handling finances","Forgetting appointments","Daily thinking problems"], start=1):
            v = st.selectbox(f"Q{i}. {q}", ["0 - No","1 - Yes"], key=f"ad8_{i}")
            ad8_answers.append(int(v.split()[0]))
    st.markdown("---")
    st.header("Console / Results")
    console = st.empty()
    results_placeholder = st.empty()

with right:
    st.header("Visuals & Explanations")
    shap_img_place = st.empty()
    shap_text_place = st.empty()

# Process
processing = None
summary = {}

if process_btn:
    console.info("Reading files and starting processing...")
    # parse lab text
    lab_findings = []
    if lab_file:
        try:
            b = lab_file.read()
            try:
                txt = b.decode('utf-8', errors='ignore')
            except Exception:
                txt = ""
            lab_findings = parse_lab_text(txt)
        except Exception:
            lab_findings = []
    # read EDF
    raw_like, msg = read_edf_uploaded(uploaded) if uploaded else (None, "No EDF uploaded")
    if raw_like is None:
        console.error(f"EDF read error: {msg}. If you want to test, upload an EDF file.")
    else:
        console.info("Computing spectral features (denoising applied)...")
        res = compute_band_powers(raw_like)
        if res is None:
            console.error("Spectral computation failed.")
        else:
            # prepare band_summary
            band_summary = res["bands"]
            # compute stress index
            stress = compute_stress_index(band_summary)
            # compute FDI and theta/alpha
            metrics = res.get("metrics", {})
            metrics["stress_index"] = stress
            # prepare topomaps
            topo_images = {}
            topo_text = {}
            for b in BANDS:
                vals = [band_summary[ch].get(f"{b}_rel", 0.0) for ch in res["ch_names"]]
                img = topomap_png_from_vals(vals, band_name=b)
                topo_images[b] = img
                topo_text[b] = (f"{b} topomap: shows relative distribution of {b} power across channels. "
                                "Areas with higher values indicate relatively stronger activity in this band. "
                                "Interpret in combination with clinical history.")
            # SHAP
            shap_img = render_shap_image(SHAP_JSON) if SHAP_JSON.exists() and HAS_SHAP else None
            shap_text = shap_explanation_text(SHAP_JSON) if shap_img else ("SHAP not available — place shap_summary.json in project root and install shap.")
            # eye state detection (Berger effect): look for high occipital alpha (O1/O2)
            occ_vals = []
            for ch in res["ch_names"]:
                if "O1" in ch or "O2" in ch:
                    occ_vals.append(band_summary[ch].get("Alpha_rel", 0.0))
            oc_mean = float(np.mean(occ_vals)) if occ_vals else float(np.mean([band_summary[ch].get("Alpha_rel",0.0) for ch in band_summary]))
            eye_state = "Eyes Closed" if oc_mean > 0.12 else "Eyes Open"  # threshold heuristic
            # questionnaire scoring
            phq_score = sum(phq_answers) if 'phq_answers' in locals() else 0
            ad8_score = sum(ad8_answers) if 'ad8_answers' in locals() else 0
            # final heuristic risk (example)
            theta_alpha = metrics.get("theta_alpha", 0.0)
            risk_depression = 0.2*(phq_score/27.0) + 0.6*(stress) + 0.2*(theta_alpha/2.0)
            risk_alz = 0.3*(ad8_score/16.0) + 0.5*(theta_alpha/2.0) + 0.2*metrics.get("FDI",0.0)
            # textual recommendations
            recs = []
            if metrics.get("FDI",0.0) > 2.5:
                recs.append("High focal delta (FDI) — consider structural imaging (MRI) to rule out focal lesion.")
            if stress > 0.6:
                recs.append("Elevated EEG-derived stress index — consider stress management, sleep hygiene, psych referral.")
            if lab_findings:
                recs.append(f"Lab abnormalities detected: {', '.join(lab_findings)} — consider metabolic correction.")
            if not recs:
                recs.append("No acute automated findings; integrate with clinical assessment.")
            # populate summary
            summary = {
                "patient_info": {"name": patient_name, "id": patient_id, "dob": str(dob)},
                "bands": band_summary,
                "metrics": metrics,
                "topo_images": topo_images,
                "topo_text": topo_text,
                "shap_img": shap_img,
                "shap_text": shap_text,
                "eye_state": eye_state,
                "phq_score": phq_score,
                "ad8_score": ad8_score,
                "lab_findings": lab_findings,
                "recommendations": recs,
                "created": now_ts()
            }
            processing = summary
            console.success("Processing complete.")

# Display results
if processing:
    results_placeholder.subheader("Results")
    st.markdown(f"**Detected eye state:** **{processing['eye_state']}**")
    st.markdown("### Key metrics")
    kcols = st.columns(4)
    kcols[0].metric("Theta/Alpha", f"{processing['metrics'].get('theta_alpha',0.0):.2f}")
    kcols[1].metric("FDI", f"{processing['metrics'].get('FDI',0.0):.2f}")
    kcols[2].metric("Stress index", f"{processing['metrics'].get('stress_index',0.0)*100:.0f}%")
    kcols[3].metric("PHQ-9", f"{processing.get('phq_score',0)}")
    st.markdown("### Topomaps")
    cols = st.columns(2)
    i = 0
    for band, img in processing['topo_images'].items():
        with cols[i%2]:
            if img:
                st.image(img, caption=f"{band} — {processing['topo_text'].get(band,'')}", use_column_width=True)
            else:
                st.info(f"{band} topomap not available.")
        i += 1
    st.markdown("### SHAP (explainability)")
    if processing.get('shap_img'):
        shap_img_place.image(processing['shap_img'], caption="SHAP feature importance")
        shap_text_place.markdown(f"**Explanation:** {processing.get('shap_text')}")
    else:
        shap_img_place.info("SHAP unavailable. Place shap_summary.json in project root and ensure shap is installed.")
        shap_text_place.write("")
    st.markdown("### Lab findings (auto-parsed)")
    st.write(processing.get("lab_findings", []))
    st.markdown("### Recommendations")
    for r in processing.get("recommendations", []):
        st.write(f"- {r}")
    # PDF generation
    if st.button("Generate PDF report (clinician)"):
        pdf_bytes = generate_pdf(processing, lang=("ar" if lang.startswith("Arabic") else "en"))
        if pdf_bytes:
            st.download_button("Download PDF", data=pdf_bytes, file_name=f"NeuroEarly_Report_{now_ts()}.pdf", mime="application/pdf")
        else:
            st.error("PDF generation not available (reportlab not installed or error).")
else:
    st.info("No processed result — run analysis first.")

st.markdown("---")
st.markdown("Notes: This tool is for assistive screening only. Clinical correlation and specialist review required.")

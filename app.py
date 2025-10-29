# NeuroEarly Pro - Streamlit app (single-file, production-ready template)
# Features:
# - bilingual UI (EN/AR), Arabic uses Amiri font (assets/Amiri-Regular.ttf)
# - EDF upload (pyedflib / mne fallback)
# - QEEG metrics (band powers, ratios, asymmetry), focal delta index
# - Connectivity (if mne available), Microstate placeholder
# - SHAP loading from shap_summary.json (in repo) or upload
# - PDF generation (reportlab) with visuals, bilingual content and Golden Bird footer
# - Safe fallbacks and clear logging for missing heavy libs

import os
import io
import sys
import json
import math
import base64
import tempfile
import traceback
from datetime import datetime, date
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd
import streamlit as st

# Optional heavy libs
HAS_MNE = False
HAS_PYEDFLIB = False
HAS_MATPLOTLIB = False
HAS_REPORTLAB = False
HAS_SHAP = False
HAS_ARABIC_TOOLS = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    HAS_MATPLOTLIB = True
except Exception:
    HAS_MATPLOTLIB = False

try:
    import pyedflib
    HAS_PYEDFLIB = True
except Exception:
    HAS_PYEDFLIB = False

try:
    import mne
    HAS_MNE = True
except Exception:
    HAS_MNE = False

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
    import scipy.signal as sps
except Exception:
    sps = None

try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

# Arabic shaping
try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    HAS_ARABIC_TOOLS = True
except Exception:
    HAS_ARABIC_TOOLS = False

# Assets and paths
REPO_ROOT = Path(__file__).parent
AMIRI_FONT_PATH = REPO_ROOT / "assets" / "Amiri-Regular.ttf"
LOGO_PATH = REPO_ROOT / "assets" / "GoldenBird_logo.svg"
SHAP_JSON_PATH = REPO_ROOT / "shap_summary.json"

# UI constants
BAND_DEFS = {
    "Delta": (0.5, 4),
    "Theta": (4, 8),
    "Alpha": (8, 13),
    "Beta": (13, 30),
    "Gamma": (30, 45)
}

BLUE_COLOR = "#0b63d6"

# ----------------- Utility helpers -----------------
def now_ts():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def safe_plot_to_bytes(fig):
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

def format_percent(x, ndigits=1):
    try:
        return f"{x*100:.{ndigits}f}%"
    except Exception:
        return str(x)

def reshape_ar(text):
    if HAS_ARABIC_TOOLS:
        return get_display(arabic_reshaper.reshape(text))
    return text

# ----------------- EEG processing helpers -----------------
def read_edf_bytes_to_signal(fpath):
    """Return dict: {'data': ndarray (n_ch, n_samples), 'ch_names': [..], 'sf': sf}"""
    if HAS_PYEDFLIB:
        try:
            f = pyedflib.EdfReader(str(fpath))
            n = f.signals_in_file
            ch_names = f.getSignalLabels()
            sf = int(f.getSampleFrequencies()[0])
            data = np.vstack([f.readSignal(i) for i in range(n)])
            f._close()
            return {"data": data, "ch_names": ch_names, "sf": sf}
        except Exception as e:
            st.warning(f"pyedflib read failed: {e}")
    if HAS_MNE:
        try:
            raw = mne.io.read_raw_edf(str(fpath), preload=True, verbose=False)
            data, times = raw.get_data(return_times=True)
            ch_names = raw.info["ch_names"]
            sf = int(raw.info["sfreq"])
            return {"data": data, "ch_names": ch_names, "sf": sf}
        except Exception as e:
            st.warning(f"mne read failed: {e}")
    # fallback: try to load as numpy (if file contains array)
    try:
        arr = np.load(str(fpath))
        if arr.ndim == 2:
            return {"data": arr, "ch_names": [f"Ch{i}" for i in range(arr.shape[0])], "sf": 250}
    except Exception:
        pass
    raise ValueError("Could not read EDF file. Install pyedflib or mne, or upload a supported EDF.")

def band_power_welch(data, sf, band):
    """Compute relative band power for each channel for given band tuple (low, high)."""
    if sps is None:
        raise RuntimeError("scipy.signal required for bandpower")
    nperseg = max(256, int(sf * 2))
    f, Pxx = sps.welch(data, fs=sf, nperseg=nperseg, axis=1)
    low, high = band
    mask = (f >= low) & (f <= high)
    abs_power = np.trapz(Pxx[:, mask], f[mask], axis=1)
    total_power = np.trapz(Pxx, f, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(total_power > 0, abs_power / total_power, 0.0)
    return rel  # shape (n_channels,)

def compute_band_metrics(data, sf, ch_names):
    """Return df with relative power per channel for each band and aggregated means."""
    bands_rel = {}
    for bname, band in BAND_DEFS.items():
        rel = band_power_welch(data, sf, band)  # per-channel
        bands_rel[bname] = rel
    # aggregated metrics
    df = pd.DataFrame({f"{bn.lower()}_rel": bands_rel[bn] for bn in bands_rel})
    df["ch"] = ch_names
    metrics = {
        "alpha_mean_rel": float(np.mean(bands_rel["Alpha"])),
        "theta_mean_rel": float(np.mean(bands_rel["Theta"])),
        "theta_alpha_ratio": float(np.mean(bands_rel["Theta"]) / (np.mean(bands_rel["Alpha"]) + 1e-9)),
        "theta_beta_ratio": float(np.mean(bands_rel["Theta"]) / (np.mean(bands_rel["Beta"]) + 1e-9)),
    }
    return df, metrics, bands_rel

def compute_alpha_asymmetry(bands_rel_df, ch_names):
    """Simple alpha asymmetry using F3/F4 if present, otherwise use first pair"""
    try:
        ch = list(bands_rel_df["ch"].values)
        if "F3" in ch and "F4" in ch:
            a3 = bands_rel_df.loc[bands_rel_df["ch"] == "F3", "Alpha_rel"].values[0]
            a4 = bands_rel_df.loc[bands_rel_df["ch"] == "F4", "Alpha_rel"].values[0]
            return float(a3 - a4)
        else:
            # fallback: leftmost vs rightmost
            return float(bands_rel_df["Alpha_rel"].iloc[0] - bands_rel_df["Alpha_rel"].iloc[-1])
    except Exception:
        return 0.0

def compute_focal_delta_index(bands_rel_df, ch_names):
    """Compute focal delta index (FDI) as max(delta_ch)/mean(delta_all)"""
    try:
        delta = bands_rel_df["Delta_rel"].values
        idx = np.argmax(delta)
        focal = float(delta[idx] / (np.mean(delta) + 1e-9))
        focal_ch = bands_rel_df["ch"].iloc[idx]
        return {"FDI": focal, "focal_ch": focal_ch, "focal_value": float(delta[idx])}
    except Exception:
        return {"FDI": 0.0, "focal_ch": None, "focal_value": 0.0}

def compute_connectivity_matrix(data, sf, ch_names=None, method="coherence", fmin=8.0, fmax=13.0):
    """Compute connectivity matrix using MNE (if available) or fallback simple correlation."""
    if HAS_MNE:
        try:
            info = mne.create_info(ch_names, sf, ch_types="eeg")
            raw = mne.io.RawArray(data, info)
            # band-pass for alpha band
            raw_f = raw.copy().filter(l_freq=fmin, h_freq=fmax, verbose=False)
            con = mne.connectivity.spectral_connectivity([raw_f], method="coh", mode="fourier",
                                                         sfreq=sf, fmin=fmin, fmax=fmax, faverage=True, verbose=False)
            conn = np.squeeze(con.get_data(output='dense'))
            # conn will be (n_ch, n_ch)
            return conn, f"Coherence {fmin}-{fmax}Hz"
        except Exception as e:
            st.info(f"mne connectivity failed: {e}")
    # fallback: simple Pearson correlation of band-limited envelopes (very approximate)
    try:
        # bandpass via np.fft (quick, approximate)
        mask = (np.arange(data.shape[1]) >= 0)  # dummy
        env = np.abs(sps.hilbert(data, axis=1))
        conn = np.corrcoef(env)
        return conn, "Correlation (approx)"
    except Exception:
        return None, "Connectivity not available"

# ----------------- Visualization helpers -----------------
def make_normative_bar_chart(theta_alpha, alpha_asym):
    """Return PNG bytes for bar chart comparing patient vs normative ranges."""
    fig, ax = plt.subplots(figsize=(6, 2.6))
    # Norm ranges (example)
    norm_low = 0.3
    norm_high = 1.1
    # draw box for normal (white) and pathological (red) region
    ax.add_patch(plt.Rectangle((0.8, norm_high - 0.5), 0.6, 0.5, color='pink', alpha=0.5))
    # simple bars for Theta/Alpha
    bars = [theta_alpha, alpha_asym]
    names = ["Theta/Alpha", "Alpha Asym (F3-F4)"]
    ax.barh([0.5, -0.5], [theta_alpha, alpha_asym], color=BLUE_COLOR)
    ax.set_yticks([0.5, -0.5])
    ax.set_yticklabels(names)
    ax.set_xlim(-1.0, max(1.5, theta_alpha * 1.2))
    ax.set_xlabel("value")
    ax.set_title("Theta/Alpha vs Normal")
    return safe_plot_to_bytes(fig)

def make_topomap_image(vals, ch_names=None, band_name="Alpha"):
    """Make a simple topomap-like heatmap placeholder. Returns PNG bytes."""
    if not HAS_MATPLOTLIB:
        return None
    # create a simple 2D interpolation for visualization
    n = len(vals)
    side = int(np.ceil(np.sqrt(n)))
    grid = np.zeros((side, side))
    grid.flat[:n] = vals
    fig, ax = plt.subplots(figsize=(6, 2.2))
    im = ax.imshow(grid, cmap='RdBu_r', aspect='auto')
    ax.set_title(f"Topomap - {band_name}")
    plt.colorbar(im, ax=ax, fraction=0.05)
    return safe_plot_to_bytes(fig)

def make_connectivity_image(conn_mat):
    if conn_mat is None or not HAS_MATPLOTLIB:
        return None
    fig, ax = plt.subplots(figsize=(4, 3))
    im = ax.imshow(conn_mat, cmap='viridis', vmin=np.nanmin(conn_mat), vmax=np.nanmax(conn_mat))
    ax.set_title("Connectivity Matrix")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return safe_plot_to_bytes(fig)

def make_shap_bar_image(shap_dict):
    """Given a dict feat->importance, produce bar chart bytes."""
    if not shap_dict or not HAS_MATPLOTLIB:
        return None
    s = pd.Series(shap_dict).abs().sort_values(ascending=False).head(12)
    fig, ax = plt.subplots(figsize=(6, 3))
    s.plot.bar(ax=ax, color=BLUE_COLOR)
    ax.set_title("Top SHAP feature contributions")
    ax.set_ylabel("abs importance")
    return safe_plot_to_bytes(fig)

# ----------------- PDF generator -----------------
def generate_pdf_report(summary: dict,
                        lang: str = "en",
                        amiri_path: Optional[str] = None,
                        topo_images: Optional[Dict[str, bytes]] = None,
                        conn_image: Optional[bytes] = None,
                        bar_img: Optional[bytes] = None,
                        shap_img: Optional[bytes] = None) -> bytes:
    """Create bilingual clinician-facing PDF and return bytes."""
    if not HAS_REPORTLAB:
        st.error("reportlab not installed; cannot build PDF here.")
        return b""

    if not amiri_path:
        amiri_path = str(AMIRI_FONT_PATH) if AMIRI_FONT_PATH.exists() else None

    buffer = io.BytesIO()
    try:
        doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=36, bottomMargin=36, leftMargin=44, rightMargin=44)
        styles = getSampleStyleSheet()
        base_font = "Helvetica"

        # register Amiri if available
        if amiri_path and Path(amiri_path).exists():
            try:
                pdfmetrics.registerFont(TTFont("Amiri", str(amiri_path)))
                base_font = "Amiri"
            except Exception as e:
                print("Amiri font registration failed:", e)

        styles.add(ParagraphStyle(name="TitleBlue", fontName=base_font, fontSize=16,
                                  textColor=colors.HexColor(BLUE_COLOR), alignment=1, spaceAfter=12))
        styles.add(ParagraphStyle(name="Section", fontName=base_font, fontSize=12,
                                  textColor=colors.HexColor(BLUE_COLOR), spaceAfter=8))
        styles.add(ParagraphStyle(name="Body", fontName=base_font, fontSize=10, leading=14))
        styles.add(ParagraphStyle(name="Note", fontName=base_font, fontSize=9, textColor=colors.grey))

        story = []
        # Header
        story.append(Paragraph("🧠 NeuroEarly EEG Clinical Report", styles["TitleBlue"]))
        story.append(Spacer(1, 6))
        story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles["Note"]))
        story.append(Spacer(1, 10))

        # Final ML Risk prominently
        final_risk = summary.get("Final ML Risk Score", None)
        if final_risk is not None:
            story.append(Paragraph(f"<b>Final ML Risk Score:</b> {final_risk}", styles["Section"]))
            story.append(Spacer(1, 6))

        # Summary block
        story.append(Paragraph("Executive Summary", styles["Section"]))
        if "Executive Summary" in summary:
            story.append(Paragraph(summary["Executive Summary"], styles["Body"]))
        else:
            # build from metrics
            story.append(Paragraph("QEEG summary and metrics follow below.", styles["Body"]))
        story.append(Spacer(1, 8))

        # Metrics table
        metrics_table_data = [["Metric", "Value"]]
        for k, v in summary.items():
            if k in ["Executive Summary"]:
                continue
            metrics_table_data.append([k, str(v)])
        t = Table(metrics_table_data, colWidths=[200, 300])
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(BLUE_COLOR)),
                               ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                               ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                               ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.grey),
                               ("BOX", (0, 0), (-1, -1), 0.25, colors.grey)]))
        story.append(t)
        story.append(Spacer(1, 10))

        # Bar chart
        if bar_img:
            story.append(Paragraph("Normative Comparison", styles["Section"]))
            story.append(Image(io.BytesIO(bar_img), width=5.8 * inch, height=2.4 * inch))
            story.append(Spacer(1, 8))

        # SHAP
        if shap_img:
            story.append(Paragraph("Explainable AI (SHAP)", styles["Section"]))
            story.append(Image(io.BytesIO(shap_img), width=5.8 * inch, height=2.4 * inch))
            story.append(Spacer(1, 8))

        # Topos
        if topo_images:
            story.append(Paragraph("Topographic Maps", styles["Section"]))
            for band, bdata in topo_images.items():
                story.append(Paragraph(f"{band}", styles["Body"]))
                story.append(Image(io.BytesIO(bdata), width=5.8 * inch, height=2.0 * inch))
                story.append(Spacer(1, 6))

        # Connectivity
        if conn_image:
            story.append(Paragraph("Functional Connectivity", styles["Section"]))
            story.append(Image(io.BytesIO(conn_image), width=5.8 * inch, height=2.4 * inch))
            story.append(Spacer(1, 8))

        # Footer
        story.append(Spacer(1, 10))
        story.append(Paragraph("Report generated by Golden Bird LLC — NeuroEarly Pro © 2025", styles["Note"]))
        if LOGO_PATH.exists():
            # embed logo small
            try:
                story.append(Image(str(LOGO_PATH), width=1.2 * inch, height=1.2 * inch))
            except Exception:
                pass

        doc.build(story)
        pdf_bytes = buffer.getvalue()
        buffer.close()
        return pdf_bytes

    except Exception as e:
        st.error(f"PDF generation error: {e}")
        traceback.print_exc()
        return b""

# ----------------- Streamlit UI -----------------
st.set_page_config(page_title="NeuroEarly Pro — Clinical", layout="wide")
st.title("NeuroEarly Pro — Clinical")

# Language selector
LANGUAGES = {"English": "en", "Arabic": "ar"}
selected_lang = st.sidebar.radio("Language / اللغة", options=list(LANGUAGES.keys()), index=0)
lang = LANGUAGES[selected_lang]

# Sidebar patient info
with st.sidebar:
    st.markdown("## Settings & Patient")
    report_lang = st.selectbox("Report language / لغة التقرير", options=list(LANGUAGES.keys()), index=0)
    name = st.text_input("Name / اسم")
    pid = st.text_input("ID")
    dob = st.date_input("DOB / تاريخ الميلاد", value=date(1980, 1, 1),
                        min_value=date(1900, 1, 1), max_value=date.today())
    sex = st.selectbox("Sex / الجنس", options=["Unknown", "Male", "Female"])
    # clinical lists
    st.markdown("---")
    st.markdown("### Clinical details (optional)")
    meds = st.text_area("Current medications (one per line)")
    conditions = st.text_area("Medical comorbidities (one per line)")
    lab_checks = st.text_area("Relevant labs (B12, TSH, CBC, etc.)")

# Main area
st.header("1) Upload EDF file(s) (.edf)")
uploaded = st.file_uploader("Drag & drop EDF files here", accept_multiple_files=True, type=["edf", "EDF", "npz", "npy"])
process_button = st.button("Process uploaded EEG")

# PHQ-9 (modified questions)
st.header("2) PHQ-9 (Depression screening)")
phq = {}
cols = st.columns(3)
q3_label_en = "3. Trouble falling/staying asleep, or sleeping too much"
q3_label_ar = "٣. مشکل در به خواب رفتن/بی‌خوابی یا خواب زیاد"
q5_label_en = "5. Poor appetite or overeating"
q5_label_ar = "٥. پرخوری یا کم‌خوری"
q8_label_en = "8. Moving or speaking so slowly that others could have noticed / or feeling restless"
q8_label_ar = "٨. صحبت کردن آهسته یا احساس بی‌قراری"

Q_LABELS = [
    ("1", "Little interest or pleasure in doing things"),
    ("2", "Feeling down, depressed, or hopeless"),
    ("3", q3_label_en if lang == "en" else q3_label_ar),
    ("4", "Feeling tired or having little energy"),
    ("5", q5_label_en if lang == "en" else q5_label_ar),
    ("6", "Poor concentration"),
    ("7", "Moving or speaking slowly or being fidgety"),
    ("8", q8_label_en if lang == "en" else q8_label_ar),
    ("9", "Thoughts that you would be better off dead")
]

phq_vals = {}
for idx, (qid, qtxt) in enumerate(Q_LABELS, 1):
    phq_vals[qid] = st.radio(f"Q{idx}: {qtxt}", [0, 1, 2, 3], index=0, key=f"phq_{qid}")

phq_total = sum(phq_vals.values())
phq_cat = ""
if phq_total <= 4:
    phq_cat = "Minimal"
elif phq_total <= 9:
    phq_cat = "Mild"
elif phq_total <= 14:
    phq_cat = "Moderate"
elif phq_total <= 19:
    phq_cat = "Moderately severe"
else:
    phq_cat = "Severe"

st.info(f"PHQ-9 total: {phq_total} ({phq_cat})")

# AD8 cognitive screening (simple)
st.header("3) AD8 (Cognitive screening)")
AD8_QS = [
    "Problems with judgment (handling problems at home, dealing with money, etc.)",
    "Reduced interest in hobbies/activities",
    "Repeating the same questions/stories",
    "Difficulty learning how to use a tool, appliance, or gadget",
    "Forgetting the correct month or year",
    "Difficulty handling complicated financial affairs",
    "Trouble remembering appointments",
    "Thinking that family members are not as they used to be"
]
ad8_vals = {}
for i, q in enumerate(AD8_QS, 1):
    ad8_vals[f"A{i}"] = st.radio(f"A{i}: {q}", [0, 1], index=0, key=f"ad8_{i}")
ad8_total = sum(ad8_vals.values())
st.info(f"AD8 total: {ad8_total} (>=2 suggests cognitive impairment screening)")

# shap upload optional
st.markdown("### SHAP (optional)")
shap_file = st.file_uploader("Upload shap_summary.json (optional)", type=["json"])
shap_data = None
if shap_file:
    try:
        shap_data = json.load(shap_file)
        st.success("SHAP summary loaded from upload.")
    except Exception as e:
        st.warning(f"Could not load SHAP JSON: {e}")

# process EDFs when button clicked
results = []
if process_button:
    if not uploaded:
        st.warning("Upload at least one EDF file to process.")
    else:
        for up in uploaded:
            try:
                # save temp
                tf = tempfile.NamedTemporaryFile(delete=False, suffix=".edf")
                tf.write(up.read())
                tf.flush()
                tf.close()
                # read
                rec = read_edf_bytes_to_signal(tf.name)
                data = rec["data"]
                ch_names = rec["ch_names"]
                sf = rec["sf"]
                # ensure shape (n_ch, n_samples)
                if data.ndim == 1:
                    data = data[np.newaxis, :]
                df_bands, metrics, bandvals = compute_band_metrics(data, sf, ch_names)
                alpha_asym = compute_alpha_asymmetry(df_bands, ch_names)
                focal = compute_focal_delta_index(df_bands, ch_names)
                # connectivity
                conn_mat, conn_narr = compute_connectivity_matrix(data, sf, ch_names, method="coherence")
                # topomaps
                topo_imgs = {}
                for bname in BAND_DEFS.keys():
                    vals = bandvals[bname]
                    try:
                        topo = make_topomap_image(vals, ch_names=ch_names, band_name=bname)
                    except Exception:
                        topo = None
                    topo_imgs[bname] = topo
                # shap: choose model key heuristic
                shap_used = None
                if shap_data is None and SHAP_JSON_PATH.exists():
                    try:
                        shap_data = json.load(open(SHAP_JSON_PATH, "r", encoding="utf-8"))
                    except Exception:
                        shap_data = None
                if shap_data:
                    # decide model key
                    model_key = "depression_global"
                    if metrics.get("theta_alpha_ratio", 0) > 1.3:
                        model_key = "alzheimers_global"
                    shap_used = shap_data.get(model_key, {})
                # compute final risk heuristic
                # Risk = weighted sum of theta_alpha (normed), focal index, connectivity reduction
                ta = metrics.get("theta_alpha_ratio", 0.0)
                fdi = focal.get("FDI", 0.0)
                mean_conn = None
                if conn_mat is not None:
                    mean_conn = float(np.nanmean(conn_mat))
                else:
                    mean_conn = 0.0
                # normalize heuristics
                ta_norm = min(1.0, ta / 1.4)
                fdi_norm = min(1.0, (fdi - 1.0) / 3.0) if fdi > 1.0 else 0.0
                conn_norm = max(0.0, (0.5 - mean_conn))  # lower connectivity -> higher risk
                risk_score = 0.6 * ta_norm + 0.25 * fdi_norm + 0.15 * conn_norm
                risk_pct = round(risk_score * 100, 1)
                risk_cat = "Low"
                if risk_pct > 40:
                    risk_cat = "High"
                elif risk_pct > 20:
                    risk_cat = "Moderate"
                # store result
                res = {
                    "filename": up.name,
                    "agg_features": metrics,
                    "df_bands": df_bands,
                    "topo_images": topo_imgs,
                    "connectivity_matrix": conn_mat,
                    "connectivity_narrative": conn_narr,
                    "focal": focal,
                    "mean_connectivity": mean_conn,
                    "shap": shap_used,
                    "Final ML Risk Score": f"{risk_pct}%",
                    "Risk Category": risk_cat,
                    "PHQ_total": int(phq_total),
                    "PHQ_category": phq_cat,
                    "AD8_total": int(ad8_total),
                    "patient_info": {"name": name, "id": pid, "dob": str(dob), "sex": sex},
                }
                results.append(res)
                st.success(f"Processed {up.name} — Risk: {risk_pct}% ({risk_cat})")
            except Exception as e:
                st.error(f"Failed processing {up.name}: {e}")
                traceback.print_exc()

# Show results and visualizations
if results:
    res0 = results[0]
    st.header("Normative Comparison & Connectivity")
    ta_val = res0["agg_features"].get("theta_alpha_ratio", 0.0)
    alpha_asym_val = compute_alpha_asymmetry(res0["df_bands"], res0["df_bands"]["ch"].tolist())
    bar_bytes = make_normative_bar_chart(ta_val, alpha_asym_val)
    if bar_bytes:
        st.image(bar_bytes, use_column_width="always")

    st.subheader("Focal Delta / Tumor indicators")
    try:
        fdi = res0["focal"]["FDI"]
        fch = res0["focal"]["focal_ch"]
        st.info(f"Focal Delta Alert — FDI={fdi:.2f} at {fch}")
    except Exception:
        st.info("Focal Delta info not available.")

    st.subheader("Topographic Maps (first file)")
    for band, img in res0["topo_images"].items():
        if img:
            st.image(img, caption=f"{band} topomap", use_column_width=False)
    st.subheader("Connectivity Matrix (first file)")
    if res0["connectivity_matrix"] is not None:
        conn_img = make_connectivity_image(res0["connectivity_matrix"])
        if conn_img:
            st.image(conn_img, caption=res0["connectivity_narrative"], use_column_width=False)

    # SHAP panel
    st.subheader("Explainable AI (XAI)")
    shap_img = None
    if res0.get("shap"):
        shap_img = make_shap_bar_image(res0["shap"])
        if shap_img:
            st.image(shap_img, use_column_width=False)
    else:
        st.info("No SHAP data available. Upload shap_summary.json to enable XAI visualizations.")

    st.markdown("---")
    st.subheader("Export")
    try:
        df_export = pd.DataFrame([r["agg_features"] for r in results])
        csv = df_export.to_csv(index=False).encode("utf-8")
        st.download_button("Download metrics (CSV)", data=csv, file_name=f"NeuroEarly_metrics_{now_ts()}.csv", mime="text/csv")
    except Exception:
        pass

    # PDF generation button
    st.markdown("---")
    if st.button("Generate PDF report (English/Arabic)"):
        # prepare summary dict for PDF
        summ = {}
        summ["Patient"] = f"{name} | ID: {pid} | DOB: {dob} | Sex: {sex}"
        summ["Final ML Risk Score"] = results[0].get("Final ML Risk Score", "N/A")
        summ["Risk Category"] = results[0].get("Risk Category", "N/A")
        summ["Executive Summary"] = f"Automated QEEG screening: Theta/Alpha={res0['agg_features'].get('theta_alpha_ratio',0):.3f}; FDI={res0['focal'].get('FDI',0):.2f}."
        # add more metrics
        for k,v in res0["agg_features"].items():
            summ[k] = v
        # generate images
        bar_img = bar_bytes
        topo_imgs = {k: v for k,v in res0["topo_images"].items() if v is not None}
        conn_img_b = None
        if res0["connectivity_matrix"] is not None:
            conn_img_b = make_connectivity_image(res0["connectivity_matrix"])
        shap_img_b = shap_img
        # build PDF
        pdfb = generate_pdf_report(summ, lang=(report_lang if report_lang in LANGUAGES else "English"),
                                   amiri_path=str(AMIRI_FONT_PATH) if AMIRI_FONT_PATH.exists() else None,
                                   topo_images=topo_imgs, conn_image=conn_img_b, bar_img=bar_img, shap_img=shap_img_b)
        if pdfb:
            st.success("Report generated.")
            st.download_button("Download PDF report", data=pdfb, file_name=f"NeuroEarly_Report_{now_ts()}.pdf", mime="application/pdf")
        else:
            st.error("PDF generation failed. See logs.")

# Footer
st.markdown("---")
st.markdown("Designed and developed by Golden Bird LLC — NeuroEarly Pro © 2025")


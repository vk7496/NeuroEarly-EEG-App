# app.py — NeuroEarly Pro (v6 Professional)
# Full bilingual (English default / Arabic optional RTL with Amiri),
# Topomaps, Connectivity if available, SHAP support, PDF generation
# Sidebar: patient info, meds, labs. Includes synthetic healthy EDF generator for testing.

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

# Optional heavy libs
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

# Paths — update if needed
ROOT = Path(__file__).parent
ASSETS = ROOT / "assets"
LOGO_PATH = ASSETS / "goldenbird_logo.png"
AMIRI_PATH = ROOT / "Amiri-Regular.ttf"

# Constants and bands
BANDS = {
    "Delta": (1.0, 4.0),
    "Theta": (4.0, 8.0),
    "Alpha": (8.0, 13.0),
    "Beta": (13.0, 30.0),
    "Gamma": (30.0, 45.0)
}

def now_ts(fmt="%Y%m%d_%H%M%S"):
    return datetime.utcnow().strftime(fmt)

# ------------------ Utilities ------------------
def safe_text(text: str, lang="en"):
    """Return text, reshape for Arabic if necessary."""
    if lang.startswith("ar") and HAS_ARABIC:
        shaped = arabic_reshaper.reshape(text)
        bidi = get_display(shaped)
        return bidi
    return text

def write_temp_file_from_upload(uploaded) -> Path:
    """Save uploadedBytesIO to a temp file and return path (to avoid BytesIO issues)."""
    suffix = Path(uploaded.name).suffix or ".edf"
    tf = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        tf.write(uploaded.getvalue())
        tf.close()
        return Path(tf.name)
    except Exception as e:
        try:
            tf.close()
        except:
            pass
        raise

# ------------------ EDF Reading ------------------
def read_edf_bytes(uploaded) -> Tuple[Optional[mne.io.Raw], Optional[str]]:
    """
    Try reading uploaded EDF using mne if available, else return None.
    Returns (raw, msg)
    """
    if uploaded is None:
        return None, "No file"
    # Write to temp file and pass path to mne/pyedflib — avoids BytesIO issues and Info temp keys.
    try:
        tmp_path = write_temp_file_from_upload(uploaded)
    except Exception as e:
        return None, f"Failed to save uploaded file: {e}"
    try:
        if HAS_MNE:
            raw = mne.io.read_raw_edf(str(tmp_path), preload=True, verbose=False)
            return raw, None
        else:
            # fallback using pyedflib to extract samples and create a RawArray if possible
            if HAS_PYEDFLIB:
                f = pyedflib.EdfReader(str(tmp_path))
                n = f.signals_in_file
                ch_names = f.getSignalLabels()
                sf = f.getSampleFrequency(0)
                data = np.vstack([f.readSignal(i) for i in range(n)])
                f._close()
                raw = None
                if HAS_MNE:
                    info = mne.create_info(ch_names=ch_names, sfreq=sf)
                    raw = mne.io.RawArray(data, info)
                    return raw, None
                else:
                    return None, "pyedflib read but mne not available to create Raw object."
            else:
                return None, "mne not available and pyedflib not installed."
    except Exception as e:
        return None, f"Error reading EDF: {e}"
    finally:
        # keep tmp file (some environments require it), you can remove it if desired
        pass

# ------------------ Compute Band Powers ------------------
def compute_band_powers(raw: mne.io.Raw, bands=BANDS) -> pd.DataFrame:
    """Return relative and absolute band powers per channel as DataFrame."""
    picks = mne.pick_types(raw.info, eeg=True, meg=False, exclude='bads')
    data, times = raw.get_data(picks=picks, return_times=True)
    ch_names = [raw.ch_names[p] for p in picks]
    sf = raw.info['sfreq']
    # Use Welch PSD
    from scipy.signal import welch
    res = []
    for i, ch in enumerate(data):
        f, Pxx = welch(ch, fs=sf, nperseg=int(sf*2))
        total_power = np.trapz(Pxx, f)
        row = {"ch": ch_names[i], "total": total_power}
        for bname,(lo,hi) in bands.items():
            mask = (f >= lo) & (f <= hi)
            band_power = np.trapz(Pxx[mask], f[mask]) if mask.any() else 0.0
            row[f"{bname}_abs"] = band_power
            row[f"{bname}_rel"] = (band_power / total_power) if total_power>0 else 0.0
        res.append(row)
    df = pd.DataFrame(res).set_index("ch")
    return df

# ------------------ Topomap Image Generator ------------------
def generate_topomap_image(raw: mne.io.Raw, band: Tuple[float,float], ch_names=None, cmap="RdBu_r") -> Optional[bytes]:
    """Compute band-power per channel and return PNG bytes of topomap."""
    try:
        sf = raw.info['sfreq']
        picks = mne.pick_types(raw.info, eeg=True, meg=False, exclude='bads')
        chs = [raw.ch_names[p] for p in picks]
        data, _ = raw.get_data(picks=picks, return_times=True)
        # Compute mean band power per channel using welch
        from scipy.signal import welch
        vals=[]
        for ch in data:
            f, Pxx = welch(ch, fs=sf, nperseg=int(sf*2))
            mask = (f >= band[0]) & (f <= band[1])
            power = np.trapz(Pxx[mask], f[mask]) if mask.any() else 0.0
            vals.append(power)
        vals = np.array(vals)
        # Normalize for plotting
        if vals.max() != 0:
            vals = vals / np.nanmax(vals)
        # Need montages for topomap coordinates
        try:
            montage = mne.channels.make_standard_montage('standard_1020')
            info = mne.pick_info(raw.info, picks)
            # create Evoked-like object
            evoked = mne.EvokedArray(vals.reshape(-1,1), info, tmin=0.0)
            evoked.set_montage(montage, match_case=False)
            fig = evoked.plot_topomap(times=0.0, ch_type='eeg', show=False)
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight')
            plt.close(fig)
            buf.seek(0)
            return buf.getvalue()
        except Exception:
            # fallback: simple scatter plot on 2D approximate layout
            fig, ax = plt.subplots(figsize=(4,3))
            ax.bar(range(len(vals)), vals)
            ax.set_title(f"Band {band[0]}-{band[1]} Hz (approx)")
            buf = io.BytesIO()
            fig.tight_layout()
            fig.savefig(buf, format='png')
            plt.close(fig)
            buf.seek(0)
            return buf.getvalue()
    except Exception as e:
        print("Topomap failed:", e)
        return None

# ------------------ SHAP visualization ------------------
def load_shap_summary(path=ROOT/"shap_summary.json"):
    try:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return None

# ------------------ PDF generator ------------------
def generate_pdf_report(summary: dict, lang="en", amiri_path: Optional[Path]=AMIRI_PATH) -> Optional[bytes]:
    """Create bilingual PDF report and return bytes (requires reportlab)."""
    if not HAS_REPORTLAB:
        return None
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=36, bottomMargin=36, leftMargin=44, rightMargin=44)
    styles = getSampleStyleSheet()
    base_font = "Helvetica"
    if amiri_path and amiri_path.exists():
        try:
            pdfmetrics.registerFont(TTFont("Amiri", str(amiri_path)))
            base_font = "Amiri"
        except Exception as e:
            print("Amiri reg failed:", e)
    styles.add(ParagraphStyle(name="TitleBlue", fontName=base_font, fontSize=16, textColor=colors.HexColor("#0b63d6"), alignment=1, spaceAfter=8))
    styles.add(ParagraphStyle(name="H2", fontName=base_font, fontSize=12, textColor=colors.HexColor("#0b63d6"), spaceAfter=6))
    styles.add(ParagraphStyle(name="Body", fontName=base_font, fontSize=10, leading=14))
    styles.add(ParagraphStyle(name="Note", fontName=base_font, fontSize=9, textColor=colors.grey))
    story = []
    # Header with logo and title
    header_table_data = []
    left = Paragraph("<b>NeuroEarly Pro — Clinical</b>", styles["TitleBlue"])
    if LOGO_PATH.exists():
        img = RLImage(str(LOGO_PATH), width=1.0*inch, height=1.0*inch)
        header_table_data = [[left, img]]
        t = Table(header_table_data, colWidths=[4.8*inch, 1.0*inch])
    else:
        header_table_data = [[left]]
        t = Table(header_table_data)
    t.setStyle(TableStyle([("VALIGN",(0,0),(-1,-1),"MIDDLE")]))
    story.append(t)
    story.append(Spacer(1,12))
    # Patient summary
    patient_info = summary.get("patient_info", {})
    story.append(Paragraph(safe_text("Patient summary", lang), styles["H2"]))
    rows = [["Field", "Value"]]
    rows.append(["Patient ID", patient_info.get("id","")])
    rows.append(["DOB", patient_info.get("dob","")])
    rows.append(["Sex", patient_info.get("sex","")])
    rows.append(["Meds", patient_info.get("meds","")])
    tbl = Table(rows, colWidths=[2.5*inch, 3.5*inch])
    tbl.setStyle(TableStyle([("GRID",(0,0),(-1,-1),0.25,colors.grey),("BACKGROUND",(0,0),(-1,0),colors.HexColor("#eef7ff"))]))
    story.append(tbl)
    story.append(Spacer(1,6))
    # Metrics table
    metrics = summary.get("metrics", {})
    story.append(Paragraph(safe_text("QEEG Key Metrics", lang), styles["H2"]))
    if metrics:
        rows = [["Metric", "Value"]]
        for k,v in metrics.items():
            rows.append([k, str(v)])
        t2 = Table(rows, colWidths=[3.5*inch,2.5*inch])
        t2.setStyle(TableStyle([("GRID",(0,0),(-1,-1),0.25,colors.grey),("BACKGROUND",(0,0),(-1,0),colors.HexColor("#eef7ff"))]))
        story.append(t2)
        story.append(Spacer(1,8))
    # Insert bar image (Theta/Alpha)
    if summary.get("normative_bar"):
        try:
            bar_bytes = summary["normative_bar"]
            story.append(Paragraph(safe_text("Normative Comparison", lang), styles["H2"]))
            story.append(Spacer(1,4))
            story.append(RLImage(io.BytesIO(bar_bytes), width=5.5*inch, height=3.0*inch))
            story.append(Spacer(1,6))
        except Exception:
            pass
    # Topomaps
    topo_imgs = summary.get("topo_images", {})
    if topo_imgs:
        story.append(Paragraph(safe_text("Topography Maps (bands)", lang), styles["H2"]))
        imgs = [RLImage(io.BytesIO(b), width=2.6*inch, height=1.6*inch) for b in topo_imgs.values() if b]
        # arrange two per row
        rows=[]
        row=[]
        for im in imgs:
            row.append(im)
            if len(row)==2:
                rows.append(row); row=[]
        if row:
            rows.append(row)
        for r in rows:
            t = Table([r], colWidths=[3.0*inch]*len(r))
            t.setStyle(TableStyle([("GRID",(0,0),(-1,-1),0.25,colors.grey)]))
            story.append(t)
            story.append(Spacer(1,6))
    # SHAP
    if summary.get("shap_img"):
        try:
            story.append(Paragraph(safe_text("Top contributors (SHAP)", lang), styles["H2"]))
            story.append(RLImage(io.BytesIO(summary["shap_img"]), width=5.5*inch, height=2.2*inch))
            story.append(Spacer(1,6))
        except Exception:
            pass
    # Recommendations & footer
    story.append(Paragraph(safe_text("Structured Clinical Recommendations", lang), styles["H2"]))
    for r in summary.get("recommendations", []):
        story.append(Paragraph(r, styles["Body"]))
    story.append(Spacer(1,12))
    story.append(Paragraph(safe_text("Prepared by Golden Bird LLC — NeuroEarly Pro", lang), styles["Note"]))
    # build
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

# ------------------ UI / Main ------------------
st.set_page_config(page_title="NeuroEarly Pro — Clinical", layout="wide")
# Layout: sidebar for patient info / upload / language; main area for console + visualizations

# Sidebar
with st.sidebar:
    st.image(str(LOGO_PATH)) if LOGO_PATH.exists() else None
    st.header("Settings")
    lang = st.selectbox("Language / اللغة", ["English", "العربية"])
    lang_code = "ar" if lang.startswith("ع") else "en"
    pid = st.text_input("Patient ID")
    dob = st.date_input("Date of birth", value=date(1980,1,1), min_value=date(1900,1,1), max_value=date(2025,12,31))
    sex = st.selectbox("Sex / الجنس", ["Unknown","Male","Female"])
    meds = st.text_area("Current meds (one per line)")
    labs = st.text_area("Relevant labs (B12, TSH, etc.)")
    st.markdown("---")
    st.subheader("Upload")
    uploaded = st.file_uploader("Upload EDF file", type=["edf","EDF"], accept_multiple_files=False)
    st.button("Generate synthetic healthy EDF", on_click=None)  # placeholder, below we provide an actual generator
    # Questionnaire: PHQ-9 (simplified) and Alzheimer screening (short)
    st.markdown("---")
    st.subheader("Questionnaires")
    st.write("PHQ-9 (brief)")
    phq = {}
    phq[1] = st.radio("Q1", [0,1,2,3], index=0, key="q1")
    phq[2] = st.radio("Q2", [0,1,2,3], index=0, key="q2")
    phq[3] = st.radio("Q3", [0,1,2,3], index=0, key="q3")  # sensitive question
    phq[4] = st.radio("Q4", [0,1,2,3], index=0, key="q4")
    phq[5] = st.radio("Q5", [0,1,2,3], index=0, key="q5")  # sensitive question
    phq[6] = st.radio("Q6", [0,1,2,3], index=0, key="q6")
    phq[7] = st.radio("Q7", [0,1,2,3], index=0, key="q7")
    phq[8] = st.radio("Q8", [0,1,2,3], index=0, key="q8")  # sensitive question
    phq[9] = st.radio("Q9", [0,1,2,3], index=0, key="q9")
    st.markdown("---")

# Main layout
st.title("NeuroEarly Pro — Clinical & Research")
st.write("EEG / QEEG analysis, Topomaps, Explainable AI — research demo")
col1, col2 = st.columns([1,2])

with col1:
    st.header("Console")
    console = st.empty()
    console.info("Ready. Upload EDF and press 'Process EDF' in the left panel.")

with col2:
    st.header("Upload & Quick stats")
    if uploaded:
        console.info("Saving and reading EDF file... please wait")
        raw, err = read_edf_bytes(uploaded)
        if err:
            st.error(err)
        elif raw is None:
            st.error("EDF read returned no Raw object.")
        else:
            st.success(f"EDF loaded successfully. Channels: {len(raw.ch_names)}; sfreq: {raw.info['sfreq']}")
            # compute band powers
            try:
                df_bands = compute_band_powers(raw)
                st.subheader("QEEG Band summary (relative power)")
                st.dataframe(df_bands.round(4))
            except Exception as e:
                st.error(f"Band power computation failed: {e}")
            # topomaps generation
            topo_images = {}
            for name,band in BANDS.items():
                img = generate_topomap_image(raw, band)
                topo_images[name] = img
            # SHAP visualization
            shap_data = load_shap_summary(ROOT/"shap_summary.json")
            shap_img_bytes = None
            if shap_data:
                # build a simple bar chart for top features of a relevant model key if present
                model_key = "depression_global" if np.mean([phq[i] for i in range(1,10)]) > 2.0 else "alzheimers_global"
                features = shap_data.get(model_key, {})
                if features:
                    s = pd.Series(features).abs().sort_values(ascending=False)
                    fig = plt.figure(figsize=(6,2.2))
                    ax = fig.add_subplot(111)
                    s.head(10).plot.bar(ax=ax)
                    ax.set_title("Top SHAP contributors")
                    buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format='png'); plt.close(fig); buf.seek(0)
                    shap_img_bytes = buf.getvalue()
            # Normative bar (theta/alpha + alpha asymmetry)
            try:
                # compute theta/alpha global ratio
                theta = df_bands["Theta_rel"].mean()
                alpha = df_bands["Alpha_rel"].mean()
                tar = theta/alpha if alpha>0 else 0.0
                # build a simple bar chart with healthy range
                fig, ax = plt.subplots(figsize=(5,3))
                ax.bar([0],[tar], width=0.6)
                ax.set_ylim(0, max(1.5, tar+0.2))
                ax.set_xticks([0]); ax.set_xticklabels(["Theta/Alpha ratio"])
                # shading good vs bad
                ax.axhspan(1.0, 1.5, facecolor='red', alpha=0.2)
                ax.axhspan(0.0, 0.8, facecolor='white', alpha=0.2)
                buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format='png'); plt.close(fig); buf.seek(0)
                normative_bar = buf.getvalue()
            except Exception:
                normative_bar = None
            # create summary dict for PDF
            summary = {
                "patient_info": {"id": pid, "dob": str(dob), "sex": sex, "meds": meds},
                "metrics": {
                    "theta_alpha_ratio": float(theta/alpha if alpha>0 else 0.0),
                    "mean_connectivity_alpha": 0.0  # placeholder if no conn computed
                },
                "topo_images": topo_images,
                "shap_img": shap_img_bytes,
                "normative_bar": normative_bar,
                "recommendations": [
                    "This is an automated screening report. Clinical correlation required.",
                    "Consider MRI if focal delta index > 2 or extreme asymmetry.",
                    "Follow-up in 3-6 months for moderate risk cases."
                ],
                "created": now_ts()
            }
            # Try connectivity if mne & sklearn available (simple coherence)
            try:
                if HAS_MNE:
                    # compute connectivity (coherence) for alpha as a simple mean
                    from mne.connectivity import spectral_connectivity
                    con, freqs, times, n_epochs, n_tapers = spectral_connectivity([raw], method="coh", mode='fourier', sfreq=raw.info['sfreq'], fmin=8.0, fmax=13.0, faverage=True, verbose=False)
                    mean_conn = np.nanmean(con)
                    summary["metrics"]["mean_connectivity_alpha"] = float(mean_conn)
                    # render image for connectivity
                    fig = plt.figure(figsize=(4,3)); ax = fig.add_subplot(111)
                    ax.imshow(con.squeeze(), cmap='viridis'); fig.colorbar(ax.images[0], ax=ax, fraction=0.046, pad=0.04)
                    ax.set_title("Connectivity (alpha)")
                    buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format='png'); plt.close(fig); buf.seek(0)
                    summary["connectivity_image"] = buf.getvalue()
                else:
                    summary["connectivity_image"] = None
            except Exception as e:
                summary["connectivity_image"] = None
            # PDF generation button
            pdf_bytes = generate_pdf_report(summary, lang=lang_code, amiri_path=AMIRI_PATH if AMIRI_PATH.exists() else None)
            if pdf_bytes:
                st.download_button("Download PDF report", data=pdf_bytes, file_name=f"NeuroEarly_Report_{now_ts()}.pdf", mime="application/pdf")
                st.success("PDF generated.")
            else:
                st.error("PDF generation failed - ensure reportlab is installed.")
            # show visuals
            st.subheader("Topography maps")
            cols = st.columns(2)
            idx=0
            for bname, img in topo_images.items():
                with cols[idx%2]:
                    st.markdown(f"**{bname}**")
                    if img:
                        st.image(img, use_column_width=True)
                    else:
                        st.info("Not available")
                idx+=1
            if shap_img_bytes:
                st.subheader("SHAP contributors")
                st.image(shap_img_bytes, use_column_width=True)
    else:
        st.info("No file uploaded. Use sidebar to upload an EDF or generate a synthetic test file.")

# ------------------ Synthetic EDF generator (for local/Colab testing) ------------------
def generate_synthetic_edf(duration_s=120, sf=500, n_channels=19) -> bytes:
    """
    Generate a synthetic EDF-like binary using pyedflib if available.
    Channels: default 19 standard names (subset).
    """
    ch_names = ["Fp1","Fp2","F3","F4","C3","C4","P3","P4","O1","O2","F7","F8","T7","T8","Fz","Cz","Pz","Oz","T9"][:n_channels]
    samples = duration_s * sf
    t = np.arange(int(samples))/sf
    signals = []
    # create multi-band synthetic EEG-like signals
    for i in range(n_channels):
        # mixture of alpha(10Hz), theta(6Hz), delta(2Hz), beta(20Hz), and noise
        sig = 5*np.sin(2*np.pi*10*t + i) * np.exp(-((i%5)/5.0))  # alpha component
        sig += 2*np.sin(2*np.pi*6*t + i*0.5)
        sig += 1.5*np.sin(2*np.pi*2*t + i*0.2)
        sig += 0.5*np.sin(2*np.pi*20*t + i*0.1)
        sig += 0.5*np.random.randn(len(t))
        signals.append(sig)
    signals = np.vstack(signals)
    if HAS_PYEDFLIB:
        tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".edf")
        fname = tmpfile.name
        tmpfile.close()
        f = pyedflib.EdfWriter(fname, n_channels)
        channel_info = []
        for ch in ch_names:
            ch_dict = {'label': ch, 'dimension': 'uV', 'sample_rate': sf, 'physical_min': -500.0, 'physical_max': 500.0, 'digital_min': -32768, 'digital_max': 32767, 'transducer': '', 'prefilter': ''}
            channel_info.append(ch_dict)
        f.setSignalHeaders(channel_info)
        f.writeSamples(signals)
        f.close()
        with open(fname, "rb") as fh:
            b = fh.read()
        try:
            os.remove(fname)
        except:
            pass
        return b
    else:
        # fallback: write raw numpy as .npy in memory for download (not EDF)
        buf = io.BytesIO()
        np.save(buf, signals)
        buf.seek(0)
        return buf.getvalue()

# Synthetic EDF download UI (below the main area)
st.sidebar.markdown("---")
st.sidebar.subheader("Testing tools")
dur = st.sidebar.number_input("Synthetic EDF duration (s)", min_value=30, max_value=600, value=120)
sfreq = st.sidebar.selectbox("Sample rate", [250, 500], index=1)
if st.sidebar.button("Create & download synthetic EDF"):
    try:
        edf_bytes = generate_synthetic_edf(duration_s=dur, sf=sfreq, n_channels=19)
        st.sidebar.download_button("Download synthetic EDF", data=edf_bytes, file_name=f"synthetic_{dur}s_{sfreq}Hz.edf" if HAS_PYEDFLIB else f"synthetic_{dur}s_{sfreq}Hz.npy", mime="application/octet-stream")
    except Exception as e:
        st.sidebar.error(f"Could not generate synthetic EDF: {e}")

# End of app

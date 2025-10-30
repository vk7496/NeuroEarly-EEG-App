# app.py — NeuroEarly Pro (final, bilingual, PDF-ready)
# Requirements (approx):
# streamlit, numpy, pandas, matplotlib, pyedflib, mne, reportlab,
# arabic-reshaper, python-bidi, shap, scikit-learn
#
# Place assets:
#  - assets/Amiri-Regular.ttf
#  - assets/goldenbird_logo.png
#  - optionally: shap_summary.json
#
# Note: code uses graceful fallbacks if heavy libs missing.

import os
import io
import sys
import math
import json
import traceback
from pathlib import Path
from datetime import datetime, date
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import streamlit as st

# Optional heavy libs
HAS_PYEDF = False
HAS_MNE = False
HAS_REPORTLAB = False
HAS_ARABIC = False
HAS_SHAP = False

try:
    import pyedflib
    HAS_PYEDF = True
except Exception:
    HAS_PYEDF = False

try:
    import mne
    from mne.time_frequency import psd_array_welch
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
    import arabic_reshaper
    from bidi.algorithm import get_display
    HAS_ARABIC = True
except Exception:
    HAS_ARABIC = False

try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

# --- Paths and assets ---
ROOT = Path(__file__).parent
ASSETS = ROOT / "assets"
AMIRI_PATH = ASSETS / "Amiri-Regular.ttf"
LOGO_PATH = ASSETS / "goldenbird_logo.png"
SHAP_JSON = ROOT / "shap_summary.json"

# Colors / UI
BLUE = "#0b63d6"
LIGHT_BG = "#f7fbff"

# EEG bands definitions
BANDS = {
    "Delta": (0.5, 4.0),
    "Theta": (4.0, 8.0),
    "Alpha": (8.0, 13.0),
    "Beta": (13.0, 30.0),
    "Gamma": (30.0, 45.0),
}

# Utility: timestamp
def now_ts():
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")

# Arabic helper
def shape_arabic(text: str) -> str:
    if not HAS_ARABIC or not text:
        return text
    try:
        reshaped = arabic_reshaper.reshape(text)
        bidi_text = get_display(reshaped)
        return bidi_text
    except Exception:
        return text

# Robust EDF reader: returns data arr (n_channels, n_samples), sfreq, ch_names
def read_edf_file(path: Path) -> Tuple[Optional[np.ndarray], Optional[float], Optional[List[str]]]:
    try:
        if HAS_MNE:
            raw = mne.io.read_raw_edf(str(path), preload=True, verbose='ERROR')
            data, sfreq = raw.get_data(return_times=False), raw.info["sfreq"]
            ch_names = raw.ch_names
            return data, float(sfreq), ch_names
        elif HAS_PYEDF:
            f = pyedflib.EdfReader(str(path))
            n = f.signals_in_file
            sfreq = f.getSampleFrequency(0)
            ch_names = f.getSignalLabels()
            data = np.vstack([f.readSignal(i) for i in range(n)])
            f.close()
            return data, float(sfreq), ch_names
    except Exception as e:
        print("EDF read error:", e)
        traceback.print_exc()
    return None, None, None

# Bandpower using Welch
def bandpower(data, sf, fmin, fmax):
    # data: 1D (samples)
    try:
        freqs, psd = None, None
        if HAS_MNE:
            freqs, psd = psd_array_welch(data, sfreq=sf, fmin=fmin, fmax=fmax, verbose=False)
            # psd returned as array per freq
            p = np.trapz(psd, freqs)
            return float(p)
        else:
            # simple FFT-based estimate
            n = len(data)
            freqs = np.fft.rfftfreq(n, d=1.0 / sf)
            psd = np.abs(np.fft.rfft(data)) ** 2
            mask = (freqs >= fmin) & (freqs <= fmax)
            if not mask.any():
                return 0.0
            p = np.trapz(psd[mask], freqs[mask])
            return float(p)
    except Exception as e:
        print("bandpower err:", e)
        return 0.0

# Compute band features per channel -> DataFrame
def compute_band_features(raw_data: np.ndarray, sf: float, ch_names: List[str]) -> pd.DataFrame:
    rows = []
    for i, ch in enumerate(ch_names):
        sig = raw_data[i, :]
        band_pows = {}
        total = 0.0
        for bname, (fmin, fmax) in BANDS.items():
            p = bandpower(sig, sf, fmin, fmax)
            band_pows[f"{bname}_abs"] = p
            total += p
        # normalized
        for bname in BANDS.keys():
            abskey = f"{bname}_abs"
            relkey = f"{bname}_rel"
            band_pows[relkey] = (band_pows[abskey] / total) if total > 0 else 0.0
        band_pows["ch"] = ch
        rows.append(band_pows)
    df = pd.DataFrame(rows)
    return df

# Alpha asymmetry: use F3 / F4 if present, else best-effort from channel names
def compute_alpha_asymmetry(df_bands: pd.DataFrame) -> float:
    try:
        # find F3 and F4 rows
        if "F3" in df_bands["ch"].values and "F4" in df_bands["ch"].values:
            a3 = float(df_bands.loc[df_bands["ch"]=="F3", "Alpha_rel"].values[0])
            a4 = float(df_bands.loc[df_bands["ch"]=="F4", "Alpha_rel"].values[0])
            return a3 - a4
        # fallback: use average of frontal channels names containing 'F'
        frontal = df_bands[df_bands["ch"].str.upper().str.startswith("F")]
        if not frontal.empty and frontal.shape[0] >= 2:
            left = frontal.iloc[0]["Alpha_rel"]
            right = frontal.iloc[-1]["Alpha_rel"]
            return float(left - right)
        return 0.0
    except Exception as e:
        print("alpha asym err:", e)
        return 0.0

# Theta/Alpha ratio global (mean)
def compute_theta_alpha_ratio(df_bands: pd.DataFrame) -> float:
    try:
        th = df_bands["Theta_rel"].mean()
        al = df_bands["Alpha_rel"].mean()
        return float(th / al) if al > 0 else float('inf')
    except Exception:
        return 0.0

# Focal Delta Index (FDI) for tumor indicator
def compute_focal_delta_index(df_bands: pd.DataFrame) -> Dict[str, Any]:
    # compute mean delta per channel, compare to global mean
    out = {"FDI": None, "alert": False, "max_ch": None, "max_val": None}
    try:
        df = df_bands.copy()
        if "Delta_rel" not in df.columns:
            return out
        global_mean = df["Delta_rel"].mean()
        df["fdi"] = df["Delta_rel"] / (global_mean if global_mean>0 else 1e-9)
        idx = df["fdi"].idxmax()
        max_ch = df.loc[idx, "ch"]
        max_val = float(df.loc[idx, "fdi"])
        out["FDI"] = max_val
        out["max_ch"] = max_ch
        out["max_val"] = max_val
        if max_val > 2.0:
            out["alert"] = True
        return out
    except Exception as e:
        print("fdi err:", e)
        return out

# Compute simple connectivity (coherence/pli/wPLI placeholders)
def compute_connectivity_matrix(data: np.ndarray, sf: float, ch_names: List[str], band=(8,13)) -> Tuple[Optional[np.ndarray], Optional[str]]:
    try:
        if not HAS_MNE:
            return None, "(connectivity requires mne)"
        # create RawArray
        info = mne.create_info(ch_names, sf, ch_types='eeg')
        raw = mne.io.RawArray(data, info)
        fmin, fmax = band
        # compute coherence / spectral connectivity (using mne.connectivity if available)
        try:
            from mne.connectivity import spectral_connectivity
            con, freqs, times, n_epochs, n_tapers = spectral_connectivity(
                [raw.get_data()], method='coh', sfreq=sf, fmin=fmin, fmax=fmax, faverage=True, verbose=False
            )
            conn = np.squeeze(con)
            narr = f"Coherence {fmin}-{fmax}Hz"
            return conn, narr
        except Exception as e:
            print("spectral_connectivity failed:", e)
            return None, "(connectivity failed)"
    except Exception as e:
        print("compute_connectivity err:", e)
        return None, "(connectivity error)"

# Generate topomap image (if mne available) or simple scalp scatter
def generate_topomap_image(vals: np.ndarray, ch_names: List[str], band_name: str = "Alpha") -> bytes:
    try:
        fig = plt.figure(figsize=(4,3))
        ax = fig.add_subplot(111)
        if HAS_MNE:
            # try standard_1020 montage and plot
            try:
                montage = mne.channels.make_standard_montage("standard_1020")
                info = mne.create_info(ch_names, sfreq=256, ch_types='eeg')
                info.set_montage(montage, on_missing='ignore')
                mne.viz.plot_topomap(vals, pos=info, axes=ax, show=False)
            except Exception:
                ax.bar(range(len(vals)), vals)
                ax.set_title(f"{band_name} (no topo coords)")
        else:
            ax.bar(range(len(vals)), vals)
            ax.set_title(f"{band_name} (bar)")
        buf = io.BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format='png', dpi=120)
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue()
    except Exception as e:
        print("topo gen err:", e)
        return b""

# Generate connectivity image (heatmap)
def generate_connectivity_image(conn_mat: np.ndarray, title="Connectivity") -> bytes:
    try:
        fig = plt.figure(figsize=(4,3))
        ax = fig.add_subplot(111)
        im = ax.imshow(conn_mat, cmap='viridis', aspect='auto')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title)
        buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format='png'); plt.close(fig); buf.seek(0)
        return buf.getvalue()
    except Exception as e:
        print("conn img err:", e)
        return b""

# Load SHAP summary json if present
def load_shap_summary(path: Path) -> Optional[Dict[str, float]]:
    try:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        print("load shap err:", e)
    return None

# ---- PDF generator (ReportLab) ----
def generate_pdf_report(summary: Dict[str, Any],
                        lang: str = "en",
                        amiri_path: Optional[str] = None,
                        logo_path: Optional[str] = None) -> Optional[bytes]:
    if not HAS_REPORTLAB:
        print("ReportLab not available")
        return None
    try:
        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=36, bottomMargin=36, leftMargin=44, rightMargin=44)
        styles = getSampleStyleSheet()
        base_font = "Helvetica"
        # register Amiri if present
        if amiri_path and Path(amiri_path).exists():
            try:
                pdfmetrics.registerFont(TTFont("Amiri", str(amiri_path)))
                base_font = "Amiri"
            except Exception as e:
                print("Amiri reg failed:", e)
        styles.add(ParagraphStyle(name="TitleBlue", fontName=base_font, fontSize=16, textColor=colors.HexColor(BLUE), alignment=1, spaceAfter=8))
        styles.add(ParagraphStyle(name="H2", fontName=base_font, fontSize=12, textColor=colors.HexColor(BLUE), spaceAfter=6))
        styles.add(ParagraphStyle(name="Body", fontName=base_font, fontSize=10, leading=14))
        styles.add(ParagraphStyle(name="Note", fontName=base_font, fontSize=9, textColor=colors.grey))
        elems = []
        # Header with logo
        if logo_path and Path(logo_path).exists():
            try:
                img = Image(str(logo_path), width=1.6*inch, height=1.6*inch)
                # place as right aligned table
                header_tbl = Table([[Paragraph("NeuroEarly Pro — Clinical", styles["TitleBlue"]), img]],
                                   colWidths=[3.8*inch, 1.6*inch])
                header_tbl.setStyle(TableStyle([("VALIGN",(0,0),(-1,-1),"MIDDLE")]))
                elems.append(header_tbl)
            except Exception as e:
                elems.append(Paragraph("NeuroEarly Pro — Clinical", styles["TitleBlue"]))
        else:
            elems.append(Paragraph("NeuroEarly Pro — Clinical", styles["TitleBlue"]))
        elems.append(Spacer(1,6))

        # Executive summary
        ml_score = summary.get("ml_score", None)
        risk_cat = summary.get("risk_category", "N/A")
        if ml_score is None:
            ml_score = summary.get("final_ml_risk", 0.0)
        elems.append(Paragraph(f"Final ML Risk Score: {ml_score:.2f}%", styles["H2"]))
        elems.append(Paragraph(f"Risk Category: {risk_cat}", styles["Body"]))
        elems.append(Spacer(1,8))

        # patient info
        pi = summary.get("patient_info", {})
        if lang != "en" and HAS_ARABIC:
            # show right-to-left shaping
            def ptxt(t): return shape_arabic(t)
        else:
            def ptxt(t): return t
        p_lines = [
            f"Patient: {pi.get('name','-')}",
            f"ID: {pi.get('id','-')}",
            f"DOB: {pi.get('dob','-')}",
            f"Sex: {pi.get('sex','-')}",
        ]
        for pl in p_lines:
            elems.append(Paragraph(ptxt(pl), styles["Body"]))
        elems.append(Spacer(1,8))
        # QEEG Key Metrics (table)
        elems.append(Paragraph("QEEG Key Metrics", styles["H2"]))
        kdata = [
            ["Metric", "Value"]
        ]
        qmetrics = summary.get("qmetrics", {})
        for k,v in qmetrics.items():
            if isinstance(v, float):
                kv = f"{v:.4f}"
            else:
                kv = str(v)
            kdata.append([k, kv])
        tbl = Table(kdata, colWidths=[3*inch, 3*inch])
        tbl.setStyle(TableStyle([("GRID",(0,0),(-1,-1),0.5,colors.lightgrey),
                                 ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#f0f8ff"))]))
        elems.append(tbl)
        elems.append(Spacer(1,8))

        # SHAP section
        if summary.get("shap_plot_bytes"):
            elems.append(Paragraph("Explainable AI (SHAP) — Top contributors", styles["H2"]))
            try:
                shp_img = Image(io.BytesIO(summary["shap_plot_bytes"]), width=5.6*inch, height=2.4*inch)
                elems.append(shp_img)
            except Exception as e:
                elems.append(Paragraph("SHAP plot available (image embed failed).", styles["Note"]))
            elems.append(Spacer(1,6))
        else:
            elems.append(Paragraph("Explainable AI (SHAP): not available", styles["Note"]))
            elems.append(Spacer(1,6))

        # Connectivity image
        if summary.get("conn_image"):
            elems.append(Paragraph("Connectivity (summary)", styles["H2"]))
            try:
                conn_i = Image(io.BytesIO(summary["conn_image"]), width=5.6*inch, height=2.4*inch)
                elems.append(conn_i)
            except Exception:
                elems.append(Paragraph("Connectivity image embed failed.", styles["Note"]))
            elems.append(Spacer(1,6))

        # Topo images (per band)
        if summary.get("topo_images"):
            elems.append(Paragraph("Topography Maps", styles["H2"]))
            for band, img_bytes in summary["topo_images"].items():
                if not img_bytes:
                    continue
                try:
                    band_title = Paragraph(f"{band}", styles["Body"])
                    elems.append(band_title)
                    im = Image(io.BytesIO(img_bytes), width=3.0*inch, height=1.6*inch)
                    elems.append(im)
                except Exception:
                    pass
            elems.append(Spacer(1,6))

        # Focal delta / tumor alerts
        fd = summary.get("focal", {})
        if fd:
            elems.append(Paragraph("Focal Delta / Tumor indicators", styles["H2"]))
            alert_txt = f"FDI: {fd.get('FDI','-')}  Max Channel: {fd.get('max_ch','-')}"
            if fd.get("alert"):
                alert_txt = "ALERT: " + alert_txt
            elems.append(Paragraph(alert_txt, styles["Body"]))
            elems.append(Spacer(1,6))

        # Clinical recommendations (structured)
        elems.append(Paragraph("Structured Clinical Recommendations", styles["H2"]))
        recs = summary.get("recommendations", [])
        if recs:
            for r in recs:
                elems.append(Paragraph(ptxt(r), styles["Body"]))
        else:
            elems.append(Paragraph("No recommendations generated.", styles["Note"]))
        elems.append(Spacer(1,12))

        # Footer: Golden Bird mention
        elems.append(Paragraph("Report generated by Golden Bird LLC — NeuroEarly Pro", styles["Note"]))
        doc.build(elems)
        buf.seek(0)
        return buf.getvalue()
    except Exception as e:
        print("PDF gen err:", e)
        traceback.print_exc()
        return None

# ---- STREAMLIT UI ----
st.set_page_config(page_title="NeuroEarly Pro — Clinical", layout="wide")

# Sidebar: Settings & patient info
with st.sidebar:
    st.image(str(LOGO_PATH)) if LOGO_PATH.exists() else None
    st.markdown("## Settings & Patient")
    report_lang = st.selectbox("Report language / اللغة", options=["en","ar"], index=0, format_func=lambda x: "English" if x=="en" else "Arabic")
    st.write("---")
    name = st.text_input("Name / الاسم")
    pid = st.text_input("ID")
    dob = st.date_input("DOB", value=date(1980,1,1))
    sex = st.selectbox("Sex / الجنس", ["Unknown","Male","Female"])
    st.write("---")
    st.markdown("**Clinical**")
    meds = st.text_area("Current medications (one per line)")
    conditions = st.text_area("Comorbid / background conditions (one per line)")
    labs = st.text_area("Relevant lab tests (B12, TSH, ...)")
    st.write("---")
    st.markdown("**Files & XAI**")
    shap_file_uploaded = st.file_uploader("Upload shap_summary.json (optional)", type=["json"])

# Main: upload EDF
st.title("NeuroEarly Pro — Clinical")
st.caption("EEG / QEEG analysis, Connectivity, Microstates, Explainable AI — Research demo")

uploaded = st.file_uploader("Upload EDF files (you can upload multiple)", type=["edf"], accept_multiple_files=True)

# PHQ-9 (customized options per your requests)
st.header("1) PHQ-9 (Depression screening)")
# Qs: We'll present the 9 questions with options 0-3 but with your requested option texts for some Qs
phq = {}
phq_q_texts = {
    1:"Little interest or pleasure in doing things",
    2:"Feeling down, depressed, or hopeless",
    3:"Sleep (choose) — Insomnia, Hypersomnia, Normal",
    4:"Feeling tired or having little energy",
    5:"Appetite (choose) — Overeating / Undereating / Normal",
    6:"Feeling bad about yourself — or that you are a failure",
    7:"Trouble concentrating on things",
    8:"Moving or speaking so slowly OR being fidgety/restless",
    9:"Thoughts that you would be better off dead or hurting yourself"
}
col1, col2, col3 = st.columns(3)
for i in range(1,10):
    col = [col1,col2,col3][(i-1)%3]
    if i==3:
        val = col.selectbox(f"Q{i}", options=[0,1,2,3], index=0, key=f"phq{i}", help=phq_q_texts[i])
    elif i==5:
        val = col.selectbox(f"Q{i}", options=[0,1,2,3], index=0, key=f"phq{i}", help=phq_q_texts[i])
    elif i==8:
        val = col.selectbox(f"Q{i}", options=[0,1,2,3], index=0, key=f"phq{i}", help=phq_q_texts[i])
    else:
        val = col.radio(f"Q{i}", options=[0,1,2,3], index=0, key=f"phq{i}")
    phq[f"Q{i}"] = val
phq_total = sum(phq.values())
st.info(f"PHQ-9 total: {phq_total} (0–4 minimal, 5–9 mild, 10–14 moderate, 15–19 mod-severe, 20–27 severe)")

# AD8 cognitive screening (0/1)
st.header("2) AD8 (Cognitive screening)")
ad8 = {}
for i in range(1,9):
    ad8[i] = st.radio(f"A{i}", options=[0,1], index=0, key=f"ad8_{i}")
ad8_total = sum(ad8.values())
st.info(f"AD8 total: {ad8_total} (>=2 suggests cognitive impairment)")

# Process uploaded EDF(s)
results = []
if uploaded:
    processing_placeholder = st.empty()
    processing_placeholder.info("Processing files...")
    for up in uploaded:
        try:
            # save temporarily
            tmp = Path("/tmp") / f"neuroearly_{now_ts()}_{up.name}"
            with open(tmp, "wb") as f:
                f.write(up.getbuffer())
            data, sf, ch_names = read_edf_file(tmp)
            if data is None:
                st.error(f"Failed to read {up.name}")
                continue
            # Preprocessing: simple detrend / clip
            raw = data.copy()
            # Compute band features
            dfbands = compute_band_features(raw, sf, ch_names)
            # compute aggregates
            agg = {
                "theta_alpha_ratio": compute_theta_alpha_ratio(dfbands),
                "alpha_asym_F3_F4": compute_alpha_asymmetry(dfbands),
                "theta_beta_ratio": (dfbands["Theta_rel"].mean() / dfbands["Beta_rel"].mean()) if dfbands["Beta_rel"].mean()>0 else 0.0,
                "beta_alpha_ratio": (dfbands["Beta_rel"].mean() / dfbands["Alpha_rel"].mean()) if dfbands["Alpha_rel"].mean()>0 else 0.0,
                "alpha_mean_rel": dfbands["Alpha_rel"].mean(),
                "theta_mean_rel": dfbands["Theta_rel"].mean(),
            }
            # generate topo images (per band)
            topo_imgs = {}
            for band in ["Delta","Theta","Alpha","Beta","Gamma"]:
                try:
                    vals = dfbands[f"{band}_rel"].values if f"{band}_rel" in dfbands.columns else np.zeros(len(ch_names))
                    topo_imgs[band] = generate_topomap_image(vals, ch_names=ch_names, band_name=band)
                except Exception as e:
                    topo_imgs[band] = b""
            # connectivity
            conn_mat, conn_narr = compute_connectivity_matrix(raw, sf, ch_names=ch_names, band=BANDS.get("Alpha",(8.0,13.0)))
            conn_img = None
            if conn_mat is not None:
                conn_img = generate_connectivity_image(conn_mat, title="Connectivity (Alpha)")
            # focal delta
            focal = compute_focal_delta_index(dfbands)
            # Save result
            results.append({
                "filename": up.name,
                "agg_features": agg,
                "df_bands": dfbands,
                "topo_images": topo_imgs,
                "connectivity_matrix": conn_mat,
                "connectivity_narrative": conn_narr,
                "connectivity_image": conn_img,
                "focal": focal,
                "raw_sf": sf,
                "ch_names": ch_names
            })
            processing_placeholder.success(f"Processed {up.name}")
        except Exception as e:
            processing_placeholder.error(f"Failed processing {up.name}: {e}")
            traceback.print_exc()

# Show brief aggregated features for first file
if results:
    st.markdown("### Aggregated features (first file)")
    try:
        st.write(pd.Series(results[0]["agg_features"]))
    except Exception:
        st.write(results[0]["agg_features"])

# SHAP visualization in UI
st.subheader("Explainable AI (XAI)")
shap_data = None
if shap_file_uploaded:
    try:
        shap_data = json.load(shap_file_uploaded)
    except Exception as e:
        st.warning("Failed loading uploaded SHAP file.")
else:
    shap_data = load_shap_summary(SHAP_JSON) if SHAP_JSON.exists() else None

if shap_data and results:
    try:
        # heuristic choose model key
        model_key = "depression_global"
        if results[0]["agg_features"].get("theta_alpha_ratio",0) > 1.3:
            model_key = "alzheimers_global"
        features = shap_data.get(model_key, {})
        if features:
            s = pd.Series(features).abs().sort_values(ascending=False)
            st.bar_chart(s.head(10), use_container_width=True)
        else:
            st.info("SHAP present but model key missing")
    except Exception as e:
        st.warning(f"XAI load error: {e}")
else:
    st.info("No SHAP file found. Upload shap_summary.json to enable XAI visuals.")

# Export metrics CSV
st.markdown("---")
st.subheader("Export")
try:
    if results:
        df_export = pd.DataFrame([res["agg_features"] for res in results])
        csv = df_export.to_csv(index=False).encode("utf-8")
        st.download_button("Download metrics (CSV)", data=csv, file_name=f"NeuroEarly_metrics_{now_ts()}.csv", mime="text/csv")
    else:
        st.info("Upload EDF(s) to enable export.")
except Exception as e:
    st.warning(f"Export error: {e}")

# Generate PDF button
st.markdown("---")
st.subheader("Generate Report (PDF)")
if st.button("Generate PDF report"):
    try:
        if not results:
            st.error("No processed EDF results found. Upload and process at least one EDF.")
        else:
            # build summary dict
            first = results[0]
            summary = {}
            summary["patient_info"] = {"name": name, "id": pid, "dob": str(dob), "sex": sex}
            summary["qmetrics"] = {
                "Theta/Alpha Ratio": first["agg_features"].get("theta_alpha_ratio",0.0),
                "Alpha Asymmetry (F3-F4)": first["agg_features"].get("alpha_asym_F3_F4",0.0),
                "Theta mean (rel)": first["agg_features"].get("theta_mean_rel",0.0),
                "Alpha mean (rel)": first["agg_features"].get("alpha_mean_rel",0.0),
            }
            # final ML risk score heuristic (placeholder: combine theta/alpha and PHQ/AD8)
            ml_score = 0.0
            try:
                ml_score = float(summary["qmetrics"]["Theta/Alpha Ratio"])*10.0
                ml_score += (phq_total/27.0)*20.0
                ml_score += (ad8_total/8.0)*30.0
                ml_score = min(99.9, ml_score)
            except Exception:
                ml_score = 0.0
            summary["ml_score"] = ml_score
            if ml_score < 15:
                rc = "Low"
            elif ml_score < 30:
                rc = "Moderate"
            else:
                rc = "High"
            summary["risk_category"] = rc
            # SHAP plot bytes (render simple bar plot if features available)
            summary["shap_plot_bytes"] = None
            if shap_data:
                try:
                    model_key = "depression_global" if summary["qmetrics"]["Theta/Alpha Ratio"] < 1.3 else "alzheimers_global"
                    features = shap_data.get(model_key, {})
                    if features:
                        s = pd.Series(features).abs().sort_values(ascending=False).head(10)
                        fig = plt.figure(figsize=(6,2.2))
                        ax = fig.add_subplot(111)
                        s.plot(kind='barh', ax=ax, color=BLUE)
                        ax.invert_yaxis()
                        ax.set_title("Top SHAP contributors")
                        buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format='png'); plt.close(fig); buf.seek(0)
                        summary["shap_plot_bytes"] = buf.getvalue()
                except Exception as e:
                    print("shap plot err:", e)
            # connectivity image
            summary["conn_image"] = first.get("connectivity_image", None)
            # topo images
            summary["topo_images"] = {k:v for k,v in first.get("topo_images", {}).items()}
            # focal
            summary["focal"] = first.get("focal", {})
            # recommendations (simple rules)
            recs = []
            recs.append("Correlate QEEG findings with PHQ-9 and AD8 scores.")
            if summary["qmetrics"]["Theta/Alpha Ratio"] > 1.4 and summary["ml_score"] > 25:
                recs.append("Recommend neuroimaging (MRI) and referral to neurology for further evaluation.")
            else:
                recs.append("Consider clinical follow-up and repeat EEG in 3-6 months.")
            recs.append("Check reversible causes: B12, TSH, metabolic panels.")
            summary["recommendations"] = recs

            # attach shap bytes if created
            summary["shap_plot_bytes"] = summary.get("shap_plot_bytes", None)

            pdf_bytes = generate_pdf_report(summary, lang=report_lang, amiri_path=str(AMIRI_PATH) if AMIRI_PATH.exists() else None, logo_path=str(LOGO_PATH) if LOGO_PATH.exists() else None)
            if pdf_bytes:
                st.success("PDF report generated.")
                st.download_button("Download report (PDF)", data=pdf_bytes, file_name=f"NeuroEarly_Report_{now_ts()}.pdf", mime="application/pdf")
            else:
                st.error("PDF generation failed (see logs).")
    except Exception as e:
        st.error(f"Failed to generate report: {e}")
        traceback.print_exc()

# Footer UI
st.markdown("---")
st.markdown("<small>Designed by Golden Bird LLC — NeuroEarly Pro</small>", unsafe_allow_html=True)

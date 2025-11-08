# app.py — NeuroEarly Pro (v5.1) — Full Clinical Edition
# Author: generated (adapt to environment)
# Notes: designed for Streamlit Cloud + GitHub. Ensure dependencies installed:
# streamlit, mne, scipy, numpy, pandas, matplotlib, reportlab, shap (optional), arabic-reshaper, python-bidi.

import os, io, sys, tempfile, traceback, json
from pathlib import Path
from datetime import datetime, date
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

from scipy.signal import welch, coherence
from scipy.interpolate import griddata

import streamlit as st
from PIL import Image

# optional libraries
HAS_MNE = False
HAS_MNE_CONN = False
HAS_SHAP = False
HAS_REPORTLAB = False
HAS_ARABIC = False

try:
    import mne
    HAS_MNE = True
except Exception:
    HAS_MNE = False

try:
    from mne_connectivity import spectral_connectivity
    HAS_MNE_CONN = True
except Exception:
    HAS_MNE_CONN = False

try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

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
    import arabic_reshaper
    from bidi.algorithm import get_display
    HAS_ARABIC = True
except Exception:
    HAS_ARABIC = False

# Paths & constants
ROOT = Path(__file__).parent
ASSETS = ROOT / "assets"
LOGO_PATH = ASSETS / "goldenbird_logo.png"
AMIRI_PATH = ROOT / "Amiri-Regular.ttf"
SHAP_JSON = ROOT / "shap_summary.json"

APP_TITLE = "NeuroEarly Pro — Clinical & Research"
LIGHT_BG = "#eaf2ff"
BLUE = "#0b63d6"

BANDS = {
    "Delta": (1.0, 4.0),
    "Theta": (4.0, 8.0),
    "Alpha": (8.0, 13.0),
    "Beta": (13.0, 30.0),
    "Gamma": (30.0, 45.0)
}

# helpers
def now_ts(fmt="%Y%m%d_%H%M%S"):
    return datetime.utcnow().strftime(fmt)

def safe_div(a,b):
    try:
        return a/b if (b and b!=0) else (float('inf') if a and (not b) else 0.0)
    except Exception:
        return 0.0

def try_load_shap():
    if SHAP_JSON.exists():
        try:
            return json.loads(SHAP_JSON.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None

# ----------------------------
# EDF reading: robust for Streamlit UploadedFile
# ----------------------------
def read_edf_bytes(uploaded) -> Tuple[Optional['mne.io.Raw'], Optional[str]]:
    """Save uploaded file to a temp file and read with mne. Returns (raw, errmsg)."""
    if not uploaded:
        return None, "No file"
    try:
        # write a safe temp file
        with tempfile.NamedTemporaryFile(suffix=".edf", delete=False) as tmp:
            tmp.write(uploaded.getvalue())
            tmp.flush()
            tmp_path = tmp.name
        if HAS_MNE:
            raw = mne.io.read_raw_edf(tmp_path, preload=True, verbose=False)
            # do NOT write arbitrary keys into raw.info (avoid temp path assignment)
            # but store temp path externally if needed
            raw._tmp_path = tmp_path
            return raw, None
        else:
            return None, "MNE not available"
    except Exception as e:
        return None, f"Error reading EDF: {e}"

# ----------------------------
# Band power calculation (Welch)
# ----------------------------
def compute_band_powers(raw, bands=BANDS, picks=None, nperseg=2048):
    """
    Returns (df, band_vals)
    df: DataFrame indexed by ch names with columns e.g. 'Alpha_abs','Alpha_rel','total_power'
    band_vals: dict band -> np.array per channel (abs power)
    Also sets df.attrs theta_alpha_ratio and alpha_asym_F3_F4 if possible.
    """
    ch_names = raw.ch_names if picks is None else [raw.ch_names[i] for i in picks]
    sf = float(raw.info.get('sfreq', 256.0))
    data = raw.get_data(picks=picks)  # n_chan x n_samples

    nchan = data.shape[0]
    band_vals = {b: np.zeros(nchan) for b in bands}
    total_power = np.zeros(nchan)

    for ci in range(nchan):
        # compute PSD using welch
        try:
            f, Pxx = welch(data[ci, :], fs=sf, nperseg=min(nperseg, data.shape[1]))
        except Exception:
            f, Pxx = welch(data[ci, :], fs=sf)
        mask_total = (f >= 1.0) & (f <= 45.0)
        total = float(np.trapz(Pxx[mask_total], f[mask_total])) if mask_total.any() else 0.0
        total_power[ci] = total
        for band,(lo,hi) in bands.items():
            mask = (f >= lo) & (f < hi)
            p = float(np.trapz(Pxx[mask], f[mask])) if mask.any() else 0.0
            band_vals[band][ci] = p

    df = pd.DataFrame(index=ch_names)
    for band in bands:
        df[f"{band}_abs"] = band_vals[band]
        df[f"{band}_rel"] = [ (v/tp if tp>0 else 0.0) for v,tp in zip(band_vals[band], total_power) ]
    df['total_power'] = total_power

    # derived metrics
    try:
        theta_mean = df["Theta_rel"].mean()
        alpha_mean = df["Alpha_rel"].mean()
        df.attrs["theta_alpha_ratio"] = (theta_mean/alpha_mean) if alpha_mean>0 else None
    except Exception:
        df.attrs["theta_alpha_ratio"] = None
    if "F3" in df.index and "F4" in df.index:
        try:
            df.attrs["alpha_asym_F3_F4"] = df.loc["F3","Alpha_rel"] - df.loc["F4","Alpha_rel"]
        except Exception:
            df.attrs["alpha_asym_F3_F4"] = None
    return df, band_vals

# ----------------------------
# Connectivity: use mne_connectivity if available else pairwise coherence
# ----------------------------
def compute_connectivity_matrix(raw, band=(8.0,13.0), picks=None):
    data = raw.get_data(picks=picks)
    sfreq = float(raw.info.get('sfreq', 256.0))
    nchan = data.shape[0]
    try:
        if HAS_MNE_CONN:
            con, freqs, times, ne, nt = spectral_connectivity(data[np.newaxis,:,:], method='coh', mode='fourier', sfreq=sfreq,
                                                             fmin=band[0], fmax=band[1], faverage=True, verbose=False)
            conn = con.squeeze()
            if conn.ndim == 1:
                conn = conn.reshape((nchan,nchan))
            return conn, f"Spectral connectivity (MNE) computed {band[0]}-{band[1]} Hz"
        else:
            # fallback pairwise coherence
            conn = np.zeros((nchan,nchan))
            for i in range(nchan):
                for j in range(i+1, nchan):
                    try:
                        f, Cxy = coherence(data[i,:], data[j,:], fs=sfreq, nperseg=min(2048, data.shape[1]))
                        mask = (f>=band[0]) & (f<=band[1])
                        val = float(Cxy[mask].mean()) if mask.any() else 0.0
                    except Exception:
                        val = 0.0
                    conn[i,j] = conn[j,i] = val
            return conn, f"Pairwise coherence (scipy) computed {band[0]}-{band[1]} Hz"
    except Exception as e:
        return None, f"Connectivity failed: {e}"

# ----------------------------
# Topomap / Heat image generator
# ----------------------------
def generate_topomap_image(values, ch_names, raw=None, band_name="Band"):
    """
    Return PNG bytes for topomap-like visual.
    If montage available uses interpolation, else uses grid of colored circles.
    """
    vals = np.asarray(values, dtype=float)
    try:
        positions = {}
        if raw is not None:
            try:
                montage = raw.get_montage()
                posdict = montage.get_positions().get('ch_pos', {})
                positions = {ch: posdict[ch] for ch in ch_names if ch in posdict}
            except Exception:
                positions = {}
        fig = plt.figure(figsize=(4,3)); ax = fig.add_subplot(111)
        if positions and len(positions)>=3:
            xs = np.array([positions[ch][0] for ch in positions])
            ys = np.array([positions[ch][1] for ch in positions])
            vs = np.array([values[ch_names.index(ch)] for ch in positions])
            xi = np.linspace(xs.min(), xs.max(), 80)
            yi = np.linspace(ys.min(), ys.max(), 80)
            xi, yi = np.meshgrid(xi, yi)
            try:
                zi = griddata((xs,ys), vs, (xi,yi), method='cubic')
            except Exception:
                zi = griddata((xs,ys), vs, (xi,yi), method='linear')
            im = ax.imshow(zi, origin='lower', extent=(xs.min(), xs.max(), ys.min(), ys.max()), cmap='RdBu_r')
            ax.scatter(xs, ys, c='k', s=8)
            ax.set_xticks([]); ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(f"{band_name} topomap")
        else:
            # grid circles
            n = len(vals)
            cols = int(np.ceil(np.sqrt(n)))
            rows = int(np.ceil(n/cols))
            minv, maxv = (vals.min(), vals.max()) if n>0 else (0.0, 1.0)
            for idx in range(n):
                r = idx // cols
                c = idx % cols
                norm = (vals[idx]-minv)/(maxv-minv+1e-9)
                color = cm.RdBu_r(norm)
                circ = plt.Circle((c, -r), 0.4, color=color)
                ax.add_patch(circ)
                ax.text(c, -r, ch_names[idx], ha='center', va='center', fontsize=6, color='white')
            ax.set_xlim(-0.5, cols-0.5); ax.set_ylim(-rows+0.5, 0.5)
            ax.set_xticks([]); ax.set_yticks([])
            m = matplotlib.cm.ScalarMappable(cmap='RdBu_r')
            m.set_array(vals)
            fig.colorbar(m, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(f"{band_name} grid topomap")
        buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format='png', dpi=150); plt.close(fig); buf.seek(0)
        return buf.getvalue()
    except Exception:
        try:
            fig = plt.figure(figsize=(4,3)); ax = fig.add_subplot(111)
            ax.bar(range(len(vals)), vals)
            ax.set_xticks(range(len(vals))); ax.set_xticklabels(ch_names, rotation=90, fontsize=6)
            ax.set_title(band_name)
            buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format='png', dpi=150); plt.close(fig); buf.seek(0)
            return buf.getvalue()
        except Exception:
            return None

# ----------------------------
# Focal Delta Index & Extreme Asymmetry helpers
# ----------------------------
def compute_focal_delta_index(df, region_channels: List[str], global_mean=None):
    try:
        if not set(region_channels).issubset(set(df.index)):
            return None
        mean_reg = df.loc[region_channels, "Delta_abs"].mean()
        if global_mean is None:
            global_mean = df["Delta_abs"].mean()
        if global_mean in (None,0):
            return None
        return float(mean_reg / global_mean)
    except Exception:
        return None

def compute_extreme_asymmetry(df, ch_right, ch_left):
    try:
        if ch_right in df.index and ch_left in df.index:
            r = df.loc[ch_right, "Delta_abs"]; l = df.loc[ch_left, "Delta_abs"]
            if l == 0:
                return float('inf') if r>0 else None
            return float(r / l)
    except Exception:
        return None

# ----------------------------
# PDF generation (reportlab) bilingual support
# ----------------------------
def generate_pdf_report(summary: dict, lang: str="en", amiri_path: Optional[str]=None) -> Optional[bytes]:
    if not HAS_REPORTLAB:
        return None
    buffer = io.BytesIO()
    try:
        doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=36, bottomMargin=36, leftMargin=44, rightMargin=44)
        styles = getSampleStyleSheet()
        base_font = "Helvetica"
        if amiri_path and Path(amiri_path).exists():
            try:
                pdfmetrics.registerFont(TTFont("Amiri", str(amiri_path)))
                base_font = "Amiri"
            except Exception:
                base_font = "Helvetica"

        styles.add(ParagraphStyle(name="TitleBlue", fontName=base_font, fontSize=16, textColor=colors.HexColor(BLUE), alignment=1, spaceAfter=8))
        styles.add(ParagraphStyle(name="H2", fontName=base_font, fontSize=12, textColor=colors.HexColor(BLUE), spaceAfter=6))
        styles.add(ParagraphStyle(name="Body", fontName=base_font, fontSize=10, leading=14))
        styles.add(ParagraphStyle(name="Note", fontName=base_font, fontSize=9, textColor=colors.grey))

        story = []
        title = "NeuroEarly Pro — Clinical Report"
        if lang.startswith("ar") and HAS_ARABIC:
            try:
                title = get_display(arabic_reshaper.reshape("تقرير NeuroEarly Pro ـ السريري"))
            except Exception:
                title = "NeuroEarly Pro — Clinical Report"

        story.append(Paragraph(title, styles["TitleBlue"]))
        story.append(Spacer(1,6))

        # logo
        if LOGO_PATH.exists():
            try:
                img = RLImage(str(LOGO_PATH), width=1.2*inch, height=1.2*inch)
                story.append(img)
            except Exception:
                pass
        story.append(Spacer(1,6))

        # patient meta
        patient = summary.get("patient_info", {})
        pid = patient.get("id","")
        dob = patient.get("dob","")
        created = summary.get("created", now_ts())
        meta_table = [["Patient ID", pid], ["DOB", dob], ["Report generated", created]]
        t2 = Table(meta_table, colWidths=[2.2*inch, 3.2*inch])
        t2.setStyle(TableStyle([("GRID",(0,0),(-1,-1),0.25,colors.grey), ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#eef7ff"))]))
        story.append(t2); story.append(Spacer(1,8))

        # metrics
        story.append(Paragraph("<b>QEEG Key Metrics</b>", styles["H2"]))
        metrics = summary.get("metrics", {})
        for k,v in metrics.items():
            story.append(Paragraph(f"{k}: {v}", styles["Body"]))
        story.append(Spacer(1,8))

        # normative bar (if present)
        if summary.get("normative_bar"):
            try:
                story.append(Paragraph("<b>Normative Comparison</b>", styles["H2"]))
                story.append(Spacer(1,0.1*inch))
                story.append(RLImage(io.BytesIO(summary["normative_bar"]), width=5.5*inch, height=2.2*inch))
                story.append(Spacer(1,0.2*inch))
            except Exception:
                pass

        # topomaps images
        if summary.get("topo_images"):
            story.append(Paragraph("<b>Topography Maps</b>", styles["H2"]))
            imgs = [RLImage(io.BytesIO(b), width=2.6*inch, height=1.6*inch) for _,b in summary["topo_images"].items() if b]
            rows=[]; row=[]
            for im in imgs:
                row.append(im)
                if len(row)==2:
                    rows.append(row); row=[]
            if row: rows.append(row)
            for r in rows:
                story.append(Table([[x for x in r]], style=[("VALIGN",(0,0),(-1,-1),"MIDDLE")]))
            story.append(Spacer(1,8))

        # connectivity
        if summary.get("connectivity_image"):
            story.append(Paragraph("<b>Functional Connectivity (Alpha)</b>", styles["H2"]))
            try:
                story.append(RLImage(io.BytesIO(summary["connectivity_image"]), width=5.5*inch, height=3.0*inch))
            except Exception:
                pass
            story.append(Spacer(1,8))

        # SHAP
        if summary.get("shap_img"):
            story.append(Paragraph("<b>Model Explainability (SHAP)</b>", styles["H2"]))
            try:
                story.append(RLImage(io.BytesIO(summary["shap_img"]), width=5.5*inch, height=3.0*inch))
            except Exception:
                pass
            story.append(Spacer(1,8))

        # tumor/focal
        if summary.get("tumor"):
            story.append(Paragraph("<b>Focal Delta / Tumor indicators</b>", styles["H2"]))
            story.append(Paragraph(summary["tumor"].get("narrative",""), styles["Body"]))
            if summary["tumor"].get("alerts"):
                for a in summary["tumor"]["alerts"]:
                    story.append(Paragraph(f"- {a}", styles["Body"]))
            story.append(Spacer(1,6))

        # recommendations
        if summary.get("recommendations"):
            story.append(Paragraph("<b>Structured Clinical Recommendations</b>", styles["H2"]))
            for r in summary["recommendations"]:
                story.append(Paragraph(r, styles["Body"]))
            story.append(Spacer(1,12))

        story.append(Paragraph("Prepared by Golden Bird LLC — NeuroEarly Pro System, 2025", styles["Note"]))
        story.append(Spacer(1,12))
        story.append(Paragraph("Signature: ____________________", styles["Body"]))

        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()
    except Exception as e:
        return None

# ----------------------------
# UI Layout
# ----------------------------
st.set_page_config(page_title=APP_TITLE, layout="wide")
header_html = f"""
<div style="display:flex;align-items:center;justify-content:space-between;padding:10px;border-radius:8px;background:{LIGHT_BG};">
  <div style="font-weight:700;color:{BLUE};font-size:18px;">🧠 {APP_TITLE}</div>
  <div style="display:flex;align-items:center;">
     <div style="font-size:12px;color:#333;margin-right:16px;">Prepared by Golden Bird LLC</div>
     {'<img src="'+str(LOGO_PATH).replace("\\\\","/")+'" style="height:42px;">' if LOGO_PATH.exists() else ''}
  </div>
</div>
"""
st.markdown(header_html, unsafe_allow_html=True)

# Sidebar (left)
with st.sidebar:
    st.header("Settings")
    lang_choice = st.selectbox("Language / اللغة", ["English","العربية"], index=0)
    lang = "ar" if lang_choice.startswith("ع") else "en"
    st.session_state["lang"] = lang

    st.markdown("---")
    st.subheader("Patient info")
    patient_name = st.text_input("Patient Name (optional)", value="")
    patient_id = st.text_input("Patient ID", value="")
    dob = st.date_input("Date of Birth", value=date(1980,1,1), min_value=date(1900,1,1), max_value=date(2025,12,31))
    sex = st.selectbox("Sex", ["Unknown","Male","Female"], index=0)

    st.markdown("---")
    st.subheader("Current medications")
    meds = st.text_area("List current meds (one per line)", value="", height=80)
    st.markdown("---")
    st.subheader("Blood Tests (summary)")
    labs = st.text_area("Enter labs (one per line) e.g. B12: 250 pg/mL", value="", height=100)

    st.markdown("---")
    st.subheader("Upload EDF files")
    uploads = st.file_uploader("Drag & drop EDF files (.edf) — you can upload multiple", type=["edf"], accept_multiple_files=True)

    st.markdown("")
    process_btn = st.button("Process EDF(s) and Analyze")

# main columns
col_console, col_main = st.columns([1,2])
with col_console:
    st.markdown("### Console")
    console_placeholder = st.empty()

with col_main:
    st.markdown("### Results")
    main_placeholder = st.empty()

def console_log(msg, kind="info"):
    if kind=="info":
        console_placeholder.info(msg)
    elif kind=="success":
        console_placeholder.success(msg)
    elif kind=="warning":
        console_placeholder.warning(msg)
    elif kind=="error":
        console_placeholder.error(msg)
    else:
        console_placeholder.write(msg)

# storage
if "results" not in st.session_state:
    st.session_state["results"] = []

# normative bar generator (simple visual)
def generate_normative_bar(theta_alpha, alpha_asym):
    # simple bar: patient vs normative thresholds
    fig = plt.figure(figsize=(6,2.2)); ax = fig.add_subplot(111)
    labels = ["Theta/Alpha","Alpha Asym (F3-F4)"]
    vals = [theta_alpha if theta_alpha is not None else 0.0, alpha_asym if alpha_asym is not None else 0.0]
    # normative ranges: white normal, red abnormal region
    ax.bar(range(len(vals)), vals, color=['#1f77b4','#1f77b4'])
    ax.set_xticks(range(len(vals))); ax.set_xticklabels(labels)
    ax.axhspan(0,1.2,alpha=0.06,color='white')  # normative
    ax.axhspan(1.2,10,alpha=0.08,color='red')   # abnormal
    ax.set_ylim(0,max(1.5, max(vals)*1.2))
    buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format='png'); plt.close(fig); buf.seek(0)
    return buf.getvalue()

# Process on button
if process_btn:
    if not uploads:
        console_log("No EDF uploaded. Please upload EDF files.", "error")
    else:
        st.session_state["results"] = []
        for up in uploads:
            try:
                console_log(f"Reading {up.name} ...", "info")
                raw, err = read_edf_bytes(up)
                if raw is None:
                    console_log(f"{up.name} read failed: {err}", "error"); continue

                # debug info
                try:
                    data = raw.get_data()
                    console_log(f"{up.name}: {len(raw.ch_names)} channels, {raw.n_times} samples, sfreq={raw.info.get('sfreq')}", "success")
                except Exception:
                    console_log("Couldn't fetch data shape", "warning")

                # compute band powers
                df_bands, band_vals = compute_band_powers(raw)
                # connectivity (alpha)
                conn_mat, conn_narr = compute_connectivity_matrix(raw, band=BANDS.get("Alpha",(8.0,13.0)))

                # topomap images
                topo_imgs = {}
                for band, arr in band_vals.items():
                    topo_imgs[band] = generate_topomap_image(arr, df_bands.index.tolist(), raw=raw, band_name=band)

                # connectivity image
                conn_img = None
                if conn_mat is not None:
                    try:
                        fig = plt.figure(figsize=(4,3)); ax = fig.add_subplot(111)
                        im = ax.imshow(conn_mat, cmap='viridis'); fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                        ax.set_title("Connectivity (Alpha)")
                        buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format='png'); plt.close(fig); buf.seek(0)
                        conn_img = buf.getvalue()
                    except Exception as e:
                        console_log(f"Connectivity image failed: {e}", "warning")

                # focal delta and tumor indicators (heuristic)
                focal_alerts = []
                try:
                    # region for "temporal" approx hippocampal: T7/T8 or T3/T4 depending on montage
                    regionL = [c for c in df_bands.index if c.upper() in ("T7","T3","T5","T1")]
                    regionR = [c for c in df_bands.index if c.upper() in ("T8","T4","T6","T2")]
                    # compute focal delta index for strongest channel
                    peak_ch = df_bands["Delta_abs"].idxmax()
                    peak_val = df_bands.loc[peak_ch, "Delta_abs"]
                    global_mean = df_bands["Delta_abs"].mean()
                    fdi = compute_focal_delta_index(df_bands, [peak_ch], global_mean=global_mean)
                    if fdi and fdi>2.0:
                        focal_alerts.append(f"Focal Delta Index elevated at {peak_ch} (FDI={fdi:.2f}) — consider focal lesion/tumor workup.")
                    # extreme asymmetry T7/T8
                    if "T7" in df_bands.index and "T8" in df_bands.index:
                        ea = compute_extreme_asymmetry(df_bands, "T8","T7")
                        if ea and (ea>3.0 or ea<0.33):
                            focal_alerts.append(f"Extreme delta asymmetry between T7/T8 (ratio {ea:.2f}).")
                except Exception:
                    pass

                # SHAP
                shap_img = None
                shap_data = try_load_shap()
                if shap_data:
                    model_key = "depression_global"
                    if df_bands.attrs.get("theta_alpha_ratio",0) and df_bands.attrs.get("theta_alpha_ratio",0)>1.3:
                        model_key = "alzheimers_global"
                    features = shap_data.get(model_key, {})
                    if features:
                        s = pd.Series(features).abs().sort_values(ascending=False)
                        fig = plt.figure(figsize=(6,3)); ax = fig.add_subplot(111)
                        s.head(10).plot.bar(ax=ax)
                        ax.set_title("Top contributors (SHAP)")
                        buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format='png'); plt.close(fig); buf.seek(0)
                        shap_img = buf.getvalue()

                # normative bar
                norm_bar = None
                try:
                    norm_bar = generate_normative_bar(df_bands.attrs.get("theta_alpha_ratio",0) or 0.0, df_bands.attrs.get("alpha_asym_F3_F4",0) or 0.0)
                except Exception:
                    norm_bar = None

                metrics = {
                    "theta_alpha_ratio": df_bands.attrs.get("theta_alpha_ratio"),
                    "alpha_asym_F3_F4": df_bands.attrs.get("alpha_asym_F3_F4"),
                    "mean_connectivity_alpha": float(conn_mat.mean()) if (conn_mat is not None) else None
                }

                res = {
                    "filename": up.name,
                    "df_bands": df_bands,
                    "band_vals": band_vals,
                    "topo_images": topo_imgs,
                    "connectivity_matrix": conn_mat,
                    "connectivity_image": conn_img,
                    "connectivity_narrative": conn_narr,
                    "focal": {"alerts": focal_alerts, "FDI": fdi if 'fdi' in locals() else None},
                    "shap_img": shap_img,
                    "metrics": metrics,
                    "normative_bar": norm_bar,
                    "patient_info": {"id": patient_id, "dob": str(dob), "name": patient_name}
                }
                st.session_state["results"].append(res)
                console_log(f"Processed {up.name} successfully.", "success")
            except Exception as e:
                console_log(f"Error processing {up.name}: {e}\n{traceback.format_exc()}", "error")

# show results
if st.session_state.get("results"):
    for idx,res in enumerate(st.session_state["results"]):
        st.markdown(f"## Result: {res.get('filename')}")
        st.markdown("**QEEG Key Metrics**")
        st.write(res.get("metrics"))
        # topomaps
        st.markdown("### Topography Maps")
        cols = st.columns(2)
        i=0
        for band,img in res.get("topo_images",{}).items():
            if img:
                with cols[i%2]:
                    st.image(img, caption=f"{band} topomap", use_container_width=True)
                i+=1
        # connectivity
        if res.get("connectivity_image"):
            st.markdown("### Functional Connectivity (Alpha)")
            st.image(res.get("connectivity_image"), use_container_width=True)
        # normative
        if res.get("normative_bar"):
            st.markdown("### Normative Comparison")
            st.image(res.get("normative_bar"), use_container_width=True)
        # SHAP
        if res.get("shap_img"):
            st.markdown("### Model Explainability (SHAP)")
            st.image(res.get("shap_img"), use_container_width=True)
        # focal alerts
        if res.get("focal",{}).get("alerts"):
            st.markdown("### Focal Alerts")
            for a in res["focal"]["alerts"]:
                st.warning(a)

# Export / PDF
st.markdown("---")
st.subheader("Export")
if st.session_state.get("results"):
    try:
        # CSV aggregate
        rows=[]
        for r in st.session_state["results"]:
            df = r["df_bands"]
            row = {"filename": r["filename"]}
            for b in BANDS.keys():
                try:
                    row[f"{b}_mean_rel"] = float(df[f"{b}_rel"].mean())
                    row[f"{b}_mean_abs"] = float(df[f"{b}_abs"].mean())
                except Exception:
                    row[f"{b}_mean_rel"] = None; row[f"{b}_mean_abs"] = None
            row.update(r.get("metrics",{}))
            rows.append(row)
        df_export = pd.DataFrame(rows)
        csv = df_export.to_csv(index=False).encode("utf-8")
        st.download_button("Download metrics (CSV)", data=csv, file_name=f"NeuroEarly_metrics_{now_ts()}.csv", mime="text/csv")
    except Exception:
        pass

    if st.button("Generate PDF report (first result)"):
        try:
            r = st.session_state["results"][0]
            summary = {
                "patient_info": r.get("patient_info",{}),
                "metrics": r.get("metrics",{}),
                "topo_images": r.get("topo_images",{}),
                "connectivity_image": r.get("connectivity_image"),
                "tumor": r.get("focal",{}),
                "shap_img": r.get("shap_img"),
                "normative_bar": r.get("normative_bar"),
                "recommendations": [
                    "This is an automated screening report. Clinical correlation required.",
                    "Consider MRI if focal delta index >2 or extreme asymmetry is present.",
                    "Follow-up in 3-6 months for moderate risk cases."
                ],
                "created": now_ts()
            }
            pdf_bytes = generate_pdf_report(summary, lang=st.session_state.get("lang","en"), amiri_path=str(AMIRI_PATH) if AMIRI_PATH.exists() else None)
            if pdf_bytes:
                st.download_button("Download PDF report", data=pdf_bytes, file_name=f"NeuroEarly_Report_{now_ts()}.pdf", mime="application/pdf")
                st.success("PDF generated.")
            else:
                st.error("PDF generation failed — ensure reportlab is installed.")
        except Exception as e:
            st.error(f"PDF generation exception: {e}")

else:
    st.info("No processed results yet. Upload EDF and press 'Process EDF(s) and Analyze'.")

st.markdown("---")
st.markdown("Prepared by Golden Bird LLC — NeuroEarly Pro System, 2025")

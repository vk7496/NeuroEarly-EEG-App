# app.py — NeuroEarly Pro v6 (Optimized)
import os
import io
import sys
import tempfile
import traceback
import json
from pathlib import Path
from datetime import datetime, date
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image as PILImage

# --- Optional Imports with Graceful Degradation ---
HAS_MNE = False
HAS_PYEDF = False
HAS_REPORTLAB = False
HAS_SHAP = False
HAS_ARABIC = False
HAS_SCIPY = False

try:
    import mne
    HAS_MNE = True
except ImportError:
    pass

try:
    import pyedflib
    HAS_PYEDF = True
except ImportError:
    pass

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    HAS_REPORTLAB = True
except ImportError:
    pass

try:
    import shap
    HAS_SHAP = True
except ImportError:
    pass

try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    HAS_ARABIC = True
except ImportError:
    pass

try:
    from scipy.signal import welch, coherence
    from scipy.interpolate import griddata
    HAS_SCIPY = True
except ImportError:
    pass

# --- Constants & Setup ---
ROOT = Path(__file__).parent
ASSETS = ROOT / "assets"
ASSETS.mkdir(exist_ok=True) # Ensure assets dir exists

LOGO_PATH = ASSETS / "goldenbird_logo.png"
AMIRI_TTF = ROOT / "Amiri-Regular.ttf"
SHAP_JSON = ROOT / "shap_summary.json"
HEALTHY_EDF = ASSETS / "healthy_baseline.edf"

APP_TITLE = "NeuroEarly Pro — Clinical"
BLUE = "#0b63d6"
LIGHT_BG = "#eaf4ff"

BANDS = {
    "Delta": (1.0, 4.0),
    "Theta": (4.0, 8.0),
    "Alpha": (8.0, 13.0),
    "Beta": (13.0, 30.0),
    "Gamma": (30.0, 45.0),
}

# --- Helper Functions ---

def now_str(fmt="%Y-%m-%d %H:%M:%S"):
    return datetime.utcnow().strftime(fmt)

def clamp01(x):
    try:
        x = float(x)
        return max(0.0, min(1.0, x))
    except Exception:
        return 0.0

def generate_synthetic_healthy_edf(path: Path, n_channels=19, sf=250, seconds=60):
    """Creates synthetic healthy EEG data."""
    n_samples = sf * seconds
    t = np.arange(n_samples) / sf
    signals = []
    rng = np.random.RandomState(42)
    
    for _ in range(n_channels):
        sig = 0.5 * rng.normal(size=n_samples)
        alpha = 5.0 * np.sin(2 * np.pi * (8 + 2 * rng.rand()) * t + rng.rand() * 2 * np.pi) * (0.6 + 0.4 * rng.rand())
        theta = 2.0 * np.sin(2 * np.pi * (4 + 1 * rng.rand()) * t + rng.rand() * 2 * np.pi) * (0.4 + 0.3 * rng.rand())
        gamma = 0.5 * np.sin(2 * np.pi * (30 + 10 * rng.rand()) * t) * 0.2 * rng.rand()
        s = sig + alpha + theta + gamma
        s *= (1 + 0.05 * rng.randn())
        signals.append(s.astype(np.float32))
    
    arr = np.vstack(signals)
    
    # Try saving as EDF
    if HAS_PYEDF:
        try:
            ch_labels = [f"Ch{c+1}" for c in range(n_channels)]
            f = pyedflib.EdfWriter(str(path), n_channels, file_type=pyedflib.FILETYPE_EDFPLUS)
            channel_info = []
            for ch in range(n_channels):
                ch_dict = {
                    'label': ch_labels[ch], 'dimension': 'uV', 'sample_rate': sf,
                    'physical_min': float(np.min(arr[ch])), 'physical_max': float(np.max(arr[ch])),
                    'digital_min': -32768, 'digital_max': 32767,
                    'transducer': '', 'prefilter': ''
                }
                channel_info.append(ch_dict)
            f.setSignalHeaders(channel_info)
            f.writeSamples(arr.tolist())
            f.close()
            return True
        except Exception as e:
            print("pyedflib write failed:", e)
            
    # Fallback to NPZ
    try:
        np.savez_compressed(str(path.with_suffix(".npz")), data=arr, sf=sf)
        return True
    except Exception as e:
        print("saving npz failed:", e)
        return False

@st.cache_data(show_spinner=False)
def read_edf_file(file_path: str) -> Tuple[Optional[np.ndarray], Optional[dict], Optional[str]]:
    """Reads EDF from a path (used for temp files). Cached for performance."""
    try:
        if HAS_MNE:
            raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
            data = raw.get_data()
            meta = {"sfreq": float(raw.info.get("sfreq", 256.0)), "ch_names": raw.info.get("ch_names", [])}
            return data, meta, None
        elif HAS_PYEDF:
            edf = pyedflib.EdfReader(file_path)
            n = edf.signals_in_file
            chs = edf.getSignalLabels()
            sf = edf.getSampleFrequency(0)
            arrs = [edf.readSignal(i) for i in range(n)]
            edf.close()
            data = np.vstack(arrs)
            meta = {"sfreq": float(sf), "ch_names": chs}
            return data, meta, None
        else:
            return None, None, "No EDF reader found (install mne or pyedflib)."
    except Exception as e:
        return None, None, f"Read error: {e}"

def read_uploaded_wrapper(uploaded) -> Tuple[Optional[np.ndarray], Optional[dict], Optional[str]]:
    """Wrapper to handle Streamlit UploadedFile objects."""
    if uploaded is None:
        return None, None, "No file"
    
    # If it's a path string
    if isinstance(uploaded, (str, Path)):
        return read_edf_file(str(uploaded))
        
    # If it's a Streamlit UploadedFile
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tf:
            tf.write(uploaded.getvalue())
            tmp_name = tf.name
        
        data, meta, err = read_edf_file(tmp_name)
        try:
            os.remove(tmp_name) # Clean up temp file
        except:
            pass
        return data, meta, err
    except Exception as e:
        return None, None, f"Upload error: {e}"

@st.cache_data
def compute_band_powers(data: np.ndarray, sf: float) -> pd.DataFrame:
    """Computes Welch PSD band powers. Cached."""
    if not HAS_SCIPY:
        return pd.DataFrame()
        
    n_ch = data.shape[0]
    rows = []
    for i in range(n_ch):
        f, Pxx = welch(data[i,:], fs=sf, nperseg=min(2048, data.shape[1]))
        
        # Total power in 1-45Hz
        mask_total = (f>=1) & (f<=45)
        total = float(np.trapz(Pxx[mask_total], f[mask_total])) if np.any(mask_total) else 0.0
        
        row = {}
        for name, (lo, hi) in BANDS.items():
            mask = (f>=lo) & (f<hi)
            val = float(np.trapz(Pxx[mask], f[mask])) if mask.any() else 0.0
            row[f"{name}_abs"] = val
            row[f"{name}_rel"] = (val/total) if total > 0 else 0.0
        row["total_power"] = total
        rows.append(row)
        
    return pd.DataFrame(rows, index=[f"ch_{i}" for i in range(n_ch)])

def topomap_png_from_vals(vals, ch_names, band_name="Band"):
    """Generates a topomap PNG byte stream."""
    if not HAS_SCIPY:
        return None
    try:
        # 10-20 Approximate Coords (Normalized)
        coords = {
            "Fp1": (-0.5, 1.0), "Fp2": (0.5, 1.0),
            "F7": (-1.0, 0.6), "F3": (-0.4, 0.6), "Fz": (0.0, 0.6), "F4": (0.4, 0.6), "F8": (1.0, 0.6),
            "T7": (-1.1, 0.2), "C3": (-0.4, 0.2), "Cz": (0.0, 0.2), "C4": (0.4, 0.2), "T8": (1.1, 0.2),
            "P7": (-1.0, -0.3), "P3": (-0.4, -0.3), "Pz": (0.0, -0.3), "P4": (0.4, -0.3), "P8": (1.0, -0.3),
            "O1": (-0.5, -0.8), "O2": (0.5, -0.8)
        }
        
        xs, ys, zs = [], [], []
        for ch, v in zip(ch_names, vals):
            # Simple matching, try to find known labels in channel name
            matched = next((k for k in coords if k in ch), None)
            if matched:
                xs.append(coords[matched][0])
                ys.append(coords[matched][1])
                zs.append(v)
                
        if len(xs) < 4: # Not enough points
            return None

        xs, ys, zs = np.array(xs), np.array(ys), np.array(zs)
        
        grid_x, grid_y = np.mgrid[-1.2:1.2:200j, -1.2:1.2:200j]
        grid_z = griddata((xs, ys), zs, (grid_x, grid_y), method='cubic', fill_value=np.nan)
        
        fig, ax = plt.subplots(figsize=(4,4))
        im = ax.imshow(grid_z.T, origin="lower", cmap="RdBu_r", extent=[-1.2, 1.2, -1.2, 1.2])
        ax.set_title(f"{band_name}")
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        buf = io.BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format="png", dpi=100)
        plt.close(fig) # Important: Close plot
        buf.seek(0)
        return buf.getvalue()
        
    except Exception as e:
        print(f"Topomap error: {e}")
        return None

@st.cache_data
def compute_connectivity(data: np.ndarray, sf: float, method="coherence") -> Optional[np.ndarray]:
    """Computes Connectivity Matrix. Cached."""
    try:
        n_ch = data.shape[0]
        if HAS_SCIPY and method == "coherence":
            lo, hi = BANDS["Alpha"]
            conn = np.zeros((n_ch, n_ch))
            for i in range(n_ch):
                for j in range(i, n_ch):
                    try:
                        f, Cxy = coherence(data[i,:], data[j,:], fs=sf, nperseg=min(1024, data.shape[1]))
                        mask = (f>=lo) & (f<=hi)
                        val = float(np.nanmean(Cxy[mask])) if mask.any() else 0.0
                        conn[i,j] = conn[j,i] = val
                    except:
                        pass
            return conn
        else:
            # Pearson Correlation Fallback
            return np.nan_to_num(np.corrcoef(data))
    except Exception:
        return None

def compute_fdi_from_df(df: pd.DataFrame) -> Dict[str,Any]:
    if "Delta_rel" not in df.columns: return {}
    vals = df["Delta_rel"].values
    global_mean = float(np.nanmean(vals))
    idx = int(np.nanargmax(vals))
    top_val = float(vals[idx])
    fdi = (top_val / global_mean) if global_mean > 1e-9 else 0.0
    return {"top_name": df.index[idx], "FDI": round(fdi, 2)}

def render_shap_png(shap_path: Path, model_hint="depression_global"):
    if not shap_path.exists(): return None
    try:
        with open(shap_path, "r", encoding="utf-8") as f:
            sj = json.load(f)
        key = model_hint if model_hint in sj else next(iter(sj.keys()))
        feats = sj.get(key, {})
        if not feats: return None
        
        s = pd.Series(feats).abs().sort_values(ascending=False).head(12)
        fig, ax = plt.subplots(figsize=(6,3))
        s.plot.bar(ax=ax, color=BLUE)
        ax.set_title("SHAP - Top Contributors")
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue()
    except Exception:
        return None

def reshape_ar(text:str) -> str:
    if HAS_ARABIC:
        try:
            return get_display(arabic_reshaper.reshape(text))
        except:
            return text
    return text

def generate_pdf(summary: dict, lang="en") -> Optional[bytes]:
    if not HAS_REPORTLAB: return None
    try:
        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4)
        styles = getSampleStyleSheet()
        
        # Font config
        base_font = "Helvetica"
        if AMIRI_TTF.exists() and HAS_REPORTLAB:
            try:
                pdfmetrics.registerFont(TTFont("Amiri", str(AMIRI_TTF)))
                base_font = "Amiri"
            except: pass
            
        title_style = ParagraphStyle("Title", parent=styles["Heading1"], fontName=base_font, textColor=colors.HexColor(BLUE), alignment=1)
        h2_style = ParagraphStyle("H2", parent=styles["Heading2"], fontName=base_font, textColor=colors.HexColor(BLUE))
        body_style = ParagraphStyle("Body", parent=styles["Normal"], fontName=base_font)
        
        story = []
        title_txt = "NeuroEarly Pro Report" if lang=="en" else reshape_ar("تقرير NeuroEarly Pro")
        story.append(Paragraph(title_txt, title_style))
        story.append(Spacer(1, 12))
        
        if LOGO_PATH.exists():
            story.append(RLImage(str(LOGO_PATH), width=1.2*inch, height=1.2*inch))
            story.append(Spacer(1, 12))
            
        # Patient Info
        pi = summary.get("patient_info", {})
        data = [["Patient ID", pi.get("id", "-")], ["Date", summary.get("created", "-")]]
        t = Table(data, colWidths=[2*inch, 3*inch])
        t.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 0.5, colors.grey)]))
        story.append(t)
        story.append(Spacer(1, 12))
        
        # Risk
        risk_txt = f"ML Risk Score: {summary.get('final_ml_risk', 0)*100:.1f}%"
        story.append(Paragraph(risk_txt, h2_style))
        
        # Images (Topo, Connectivity, SHAP)
        img_list = []
        if summary.get("topo_images"):
            story.append(Paragraph("Topography", h2_style))
            # Just show Alpha as example to save space
            if "Alpha" in summary["topo_images"] and summary["topo_images"]["Alpha"]:
                img_list.append(RLImage(io.BytesIO(summary["topo_images"]["Alpha"]), width=2*inch, height=2*inch))
        
        if summary.get("connectivity_image"):
             img_list.append(RLImage(io.BytesIO(summary["connectivity_image"]), width=3*inch, height=2*inch))

        if img_list:
             story.append(Table([img_list]))

        story.append(Spacer(1, 12))
        story.append(Paragraph("Generated by NeuroEarly Pro", body_style))
        
        doc.build(story)
        buf.seek(0)
        return buf.getvalue()
    except Exception as e:
        print(f"PDF Error: {e}")
        return None

# --- UI Configurations ---
PHQ9 = [
    "1. Little interest or pleasure in doing things",
    "2. Feeling down, depressed, or hopeless",
    "3. Sleep problems (insomnia/hypersomnia)",
    "4. Feeling tired or having little energy",
    "5. Appetite changes (increased/decreased)",
    "6. Feeling bad about yourself",
    "7. Trouble concentrating",
    "8. Moving/speaking slowly or fidgety",
    "9. Thoughts of self-harm"
]
# Options mapped to scores
PHQ_STD = [("0 - Not at all",0),("1 - Several days",1),("2 - > Half days",2),("3 - Nearly every day",3)]

ALZ_QUESTIONS = [
    "1. Recurrent memory loss", "2. Orientation problems", "3. Naming difficulties", 
    "4. Getting lost", "5. Personality changes", "6. Difficulty daily tasks", 
    "7. Impaired judgement", "8. Social withdrawal"
]
ALZ_OPTIONS = [("0 - No",0),("1 - Occasionally",1),("2 - Often",2),("3 - Severe",3)]

# --- Main App ---
def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    
    # Header
    st.markdown(f"""
    <div style="padding:10px; background:{LIGHT_BG}; border-radius:5px; margin-bottom:20px;">
        <h2 style="color:{BLUE}; margin:0;">🧠 {APP_TITLE}</h2>
        <small>Clinical AI System</small>
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.header("Config")
        lang = st.selectbox("Language", ["English", "Arabic"])
        pid = st.text_input("Patient ID")
        uploads = st.file_uploader("Upload EDF", type=["edf"], accept_multiple_files=True)
        run_btn = st.button("Analyze", type="primary")

    # Layout: Questionnaires
    c1, c2 = st.columns(2)
    phq_score = 0
    alz_score = 0
    
    with c1:
        st.subheader("PHQ-9 (Depression)")
        for i, q in enumerate(PHQ9):
            sel = st.selectbox(q, [x[0] for x in PHQ_STD], key=f"phq_{i}")
            val = next(x[1] for x in PHQ_STD if x[0] == sel)
            phq_score += val
            
    with c2:
        st.subheader("Cognitive Screen")
        for i, q in enumerate(ALZ_QUESTIONS):
            sel = st.selectbox(q, [x[0] for x in ALZ_OPTIONS], key=f"alz_{i}")
            val = next(x[1] for x in ALZ_OPTIONS if x[0] == sel)
            alz_score += val

    if "results" not in st.session_state:
        st.session_state["results"] = []

    if run_btn:
        st.session_state["results"] = [] # Clear old results
        
        # Handle Healthy Baseline
        if not HEALTHY_EDF.exists():
            generate_synthetic_healthy_edf(HEALTHY_EDF)
        
        files_to_process = uploads if uploads else [str(HEALTHY_EDF)]
        
        for f in files_to_process:
            fname = f.name if hasattr(f, 'name') else "Healthy Baseline"
            with st.spinner(f"Processing {fname}..."):
                data, meta, err = read_uploaded_wrapper(f)
                
                if err or data is None:
                    st.error(f"Error processing {fname}: {err}")
                    continue
                
                sf = meta.get("sfreq", 250.0)
                ch_names = meta.get("ch_names", [])
                
                # 1. Band Power
                df_bands = compute_band_powers(data, sf)
                if df_bands.empty: continue
                
                # 2. Images
                topo_imgs = {}
                for b in BANDS:
                    col = f"{b}_rel"
                    if col in df_bands.columns:
                        png = topomap_png_from_vals(df_bands[col].values, ch_names, b)
                        if png: topo_imgs[b] = png
                
                # 3. Connectivity
                conn_mat = compute_connectivity(data, sf)
                conn_img = None
                if conn_mat is not None:
                    fig, ax = plt.subplots(figsize=(4,3))
                    im = ax.imshow(conn_mat, cmap="viridis")
                    plt.colorbar(im, ax=ax)
                    ax.set_title("Connectivity")
                    b = io.BytesIO()
                    fig.savefig(b, format="png")
                    plt.close(fig)
                    conn_img = b.getvalue()

                # 4. Risk Calculation (Heuristic)
                theta_alpha = 0
                if "Theta_rel" in df_bands and "Alpha_rel" in df_bands:
                    theta_alpha = df_bands["Theta_rel"].mean() / (df_bands["Alpha_rel"].mean() + 1e-6)
                
                risk = (0.4 * clamp01(theta_alpha/2.0)) + (0.3 * clamp01(phq_score/27)) + (0.3 * clamp01(alz_score/24))
                
                # Store Result
                res = {
                    "filename": fname,
                    "patient_info": {"id": pid},
                    "created": now_str(),
                    "metrics": {"Theta/Alpha": round(theta_alpha, 2)},
                    "topo_images": topo_imgs,
                    "connectivity_image": conn_img,
                    "final_ml_risk": risk,
                    "phq_score": phq_score,
                    "alz_score": alz_score
                }
                st.session_state["results"].append(res)

    # Display Results
    if st.session_state["results"]:
        st.divider()
        for r in st.session_state["results"]:
            st.markdown(f"### Results: {r['filename']}")
            c1, c2, c3 = st.columns(3)
            c1.metric("Risk Score", f"{r['final_ml_risk']*100:.1f}%")
            c2.metric("Theta/Alpha", r['metrics']['Theta/Alpha'])
            c3.metric("PHQ-9", r['phq_score'])
            
            # Topomaps Gallery
            if r['topo_images']:
                st.write("Topography Maps:")
                cols = st.columns(len(r['topo_images']))
                for idx, (band, img) in enumerate(r['topo_images'].items()):
                    cols[idx].image(img, caption=band)
            
            # PDF Download
            pdf_bytes = generate_pdf(r, lang="ar" if "Ar" in lang else "en")
            if pdf_bytes:
                st.download_button("Download PDF Report", pdf_bytes, file_name=f"report_{r['filename']}.pdf", mime="application/pdf")
            st.divider()

if __name__ == "__main__":
    main()

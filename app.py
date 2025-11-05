# app.py — NeuroEarly Pro (v3, final)
# - Bilingual (EN/AR), Amiri font support
# - Safe EDF read via temp file (no BytesIO to mne)
# - PHQ-9 (custom Q3/Q5/Q8) + AD8
# - Band powers, Theta/Alpha, Alpha asym, Connectivity (MNE/wPLI fallback->coherence), Microstates, FDI, SHAP
# - Robust PDF export (ReportLab) with unique style names and placeholders
# - Sidebar left: Language, Patient Name, Patient ID, DOB (<=2025), Meds, Labs
# - Top header with logo (assets/goldenbird_logo.png)
# - Save this file as app.py and run with `streamlit run app.py`

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
HAS_SHAP = False
HAS_REPORTLAB = False
HAS_SKLEARN = False

try:
    import mne
    HAS_MNE = True
except Exception:
    HAS_MNE = False

try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

try:
    from reportlab.lib import colors as rl_colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import inch as RL_INCH
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    HAS_REPORTLAB = True
except Exception:
    HAS_REPORTLAB = False

try:
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False

# SciPy coherence fallback
from scipy.signal import coherence

# Arabic shaping (optional)
HAS_ARABIC = False
try:
    from arabic_reshaper import reshape
    from bidi.algorithm import get_display
    HAS_ARABIC = True
except Exception:
    HAS_ARABIC = False

# -------------------------
# Paths / constants
# -------------------------
ROOT = Path(__file__).parent
ASSETS = ROOT / "assets"
AMIRI_PATH = ROOT / "Amiri-Regular.ttf"
LOGO_PATH = ASSETS / "goldenbird_logo.png"
SHAP_JSON = ROOT / "shap_summary.json"
MODEL_DIR = ROOT / "models"

PRIMARY_BLUE = "#0b63d6"
LIGHT_BG = "#eaf6ff"

BANDS = {
    "Delta": (1.0, 4.0),
    "Theta": (4.0, 8.0),
    "Alpha": (8.0, 13.0),
    "Beta": (13.0, 30.0),
    "Gamma": (30.0, 45.0)
}

# Session init
if "results" not in st.session_state:
    st.session_state["results"] = []
if "lang" not in st.session_state:
    st.session_state["lang"] = "en"

# -------------------------
# Helpers
# -------------------------
def now_ts(fmt="%Y%m%d_%H%M%S"):
    return datetime.utcnow().strftime(fmt)

def fix_arabic_text(text: str) -> str:
    if not HAS_ARABIC or not text:
        return text
    try:
        return get_display(reshape(text))
    except Exception:
        return text

def fmt(x, prec=4):
    try:
        return f"{float(x):.{prec}f}"
    except Exception:
        return str(x)

# -------------------------
# Safe EDF read (temp file)
# -------------------------
def read_edf_bytes(uploaded) -> Tuple[Optional["mne.io.Raw"], Optional[str]]:
    """
    Read uploaded EDF safely by writing to a temporary file and calling mne with the path.
    Returns (raw, error_msg)
    """
    if not uploaded:
        return None, "No file uploaded"
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmp:
            tmp.write(uploaded.getbuffer())
            tmp_path = tmp.name
        if HAS_MNE:
            raw = mne.io.read_raw_edf(tmp_path, preload=True, verbose=False)
            raw.info["__tmp_edf_path__"] = tmp_path
            return raw, None
        else:
            return None, "MNE not installed in environment"
    except Exception as e:
        return None, f"Error reading EDF: {e}"

# -------------------------
# Compute band powers (PSD Welch)
# -------------------------
def compute_band_powers(raw, bands=BANDS):
    if raw is None:
        return pd.DataFrame()
    try:
        sf = int(raw.info.get("sfreq", 256))
        picks = mne.pick_types(raw.info, eeg=True, meg=False, include=[])
        ch_names = [raw.ch_names[p] for p in picks]
        psds, freqs = mne.time_frequency.psd_welch(raw, picks=picks, fmin=1.0, fmax=45.0, verbose=False)
        rows = []
        for i, ch in enumerate(ch_names):
            pxx = psds[i]
            total = np.trapz(pxx, freqs) if freqs.size>0 else 0.0
            row = {"channel": ch}
            for bname, (lo, hi) in bands.items():
                mask = (freqs>=lo) & (freqs<hi)
                p_band = np.trapz(pxx[mask], freqs[mask]) if mask.sum()>0 else 0.0
                row[f"{bname}_abs"] = float(p_band)
                row[f"{bname}_rel"] = float(p_band/total) if total>0 else 0.0
            rows.append(row)
        return pd.DataFrame(rows)
    except Exception as e:
        print("compute_band_powers error:", e)
        return pd.DataFrame()

# -------------------------
# Figures -> PNG bytes helper
# -------------------------
def fig_to_png_bytes(fig, dpi=150):
    buf = io.BytesIO()
    try:
        fig.tight_layout()
    except Exception:
        pass
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

# Simple topomap fallback (bar per channel)
def topomap_image_for_band(vals, ch_names, band_name="Alpha"):
    try:
        fig, ax = plt.subplots(figsize=(4.0,2.4))
        x = np.arange(len(vals))
        ax.bar(x, vals, color=PRIMARY_BLUE)
        ax.set_xticks(x)
        ax.set_xticklabels(ch_names, rotation=90, fontsize=6)
        ax.set_title(f"{band_name} topography")
        return fig_to_png_bytes(fig)
    except Exception as e:
        print("topomap error:", e)
        return None

# Normative comparison bar (Theta/Alpha and Alpha Asym)
def bar_comparison_chart(theta_alpha, alpha_asym):
    try:
        fig, ax = plt.subplots(figsize=(6.0,2.4))
        metrics = ["Theta/Alpha", "Alpha Asym (F3-F4)"]
        values = [theta_alpha or 0.0, alpha_asym or 0.0]
        ax.bar(metrics, values, color=PRIMARY_BLUE)
        # normative bands (example thresholds)
        norm_low = [0.3, -0.02]; norm_high = [1.1, 0.02]
        for i in range(2):
            ax.add_patch(plt.Rectangle((i-0.25, norm_low[i]), 0.5, norm_high[i]-norm_low[i], facecolor="#ffffff", edgecolor="gray", alpha=0.3))
            if values[i] > norm_high[i]:
                ax.add_patch(plt.Rectangle((i-0.25, norm_high[i]), 0.5, values[i]-norm_high[i], facecolor="#ffcccc", alpha=0.5))
        ax.set_ylabel("Value")
        ax.set_title("Normative comparison")
        return fig_to_png_bytes(fig)
    except Exception as e:
        print("bar chart err:", e)
        return None

# -------------------------
# Connectivity (MNE preferred, fallback coherence)
# -------------------------
def compute_functional_connectivity(raw, band=(8.0,13.0)):
    nchan = 0
    try:
        data = raw.get_data(picks=mne.pick_types(raw.info, eeg=True))
        sf = int(raw.info.get("sfreq", 256))
        nchan = data.shape[0]
    except Exception:
        return None, "(no raw data)", placeholder_image("Connectivity not available"), 0.0

    conn_mat = np.zeros((nchan,nchan))
    narration = ""
    mean_conn = 0.0

    # try MNE spectral_connectivity
    try:
        from mne.connectivity import spectral_connectivity
        con, freqs, times, n_epochs, n_tapers = spectral_connectivity([raw], method="wpli", mode="multitaper",
                                                                       sfreq=sf, fmin=band[0], fmax=band[1], faverage=True, verbose=False)
        con = np.squeeze(con)
        if con.shape == (nchan,nchan) or (con.ndim==2 and con.shape[0]==nchan):
            conn_mat = con
            narration = f"wPLI {band[0]}-{band[1]} Hz (MNE)"
    except Exception:
        # fallback to coherence pairwise
        try:
            for i in range(nchan):
                for j in range(i, nchan):
                    f, Cxy = coherence(data[i], data[j], fs=sf, nperseg=min(1024, max(256, data.shape[1])))
                    mask = (f>=band[0]) & (f<=band[1])
                    val = float(np.nanmean(Cxy[mask])) if mask.sum() else 0.0
                    conn_mat[i,j] = val; conn_mat[j,i] = val
            narration = f"Coherence {band[0]}-{band[1]} Hz (scipy fallback)"
        except Exception as e:
            narration = f"(connectivity failed: {e})"

    try:
        mean_conn = float(np.nanmean(conn_mat)) if conn_mat.size else 0.0
    except Exception:
        mean_conn = 0.0

    # produce image
    try:
        fig, ax = plt.subplots(figsize=(4.5,3.2))
        im = ax.imshow(conn_mat, cmap="viridis", interpolation="nearest", aspect="auto")
        ax.set_title("Functional Connectivity")
        ax.set_xticks(range(min(40,nchan))); ax.set_xticklabels(raw.ch_names[:min(40,nchan)], fontsize=6, rotation=90)
        ax.set_yticks(range(min(40,nchan))); ax.set_yticklabels(raw.ch_names[:min(40,nchan)], fontsize=6)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        conn_img = fig_to_png_bytes(fig)
    except Exception:
        conn_img = placeholder_image("Connectivity image failed")

    return conn_mat, narration, conn_img, mean_conn

# -------------------------
# placeholder image generator
# -------------------------
def placeholder_image(text="Not available", width=600, height=200):
    try:
        fig, ax = plt.subplots(figsize=(width/100, height/100))
        ax.axis("off")
        ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=12, color="gray")
        return fig_to_png_bytes(fig)
    except Exception:
        return None

# -------------------------
# Microstate (simple)
# -------------------------
def simple_microstate_analysis(raw, n_states=4):
    out = {"maps": [], "coverage": {}, "n_states": n_states}
    try:
        data = raw.get_data()
        gfp = np.std(data, axis=0)
        thr = np.percentile(gfp, 75)
        peaks = np.where(gfp >= thr)[0]
        if peaks.size < 10:
            samples = np.linspace(0, data.shape[1]-1, min(200, data.shape[1])).astype(int)
            maps = data[:, samples].T
        else:
            maps = data[:, peaks].T
        if HAS_SKLEARN and maps.shape[0] >= n_states:
            pca = PCA(n_components=min(30, maps.shape[1]))
            maps_red = pca.fit_transform(maps)
            kmeans = KMeans(n_clusters=n_states, random_state=42, n_init=10).fit(maps_red)
            labels = kmeans.labels_
            centers = pca.inverse_transform(kmeans.cluster_centers_)
            for c in centers:
                out["maps"].append((c / (np.linalg.norm(c)+1e-12)).tolist())
            for s in range(n_states):
                out["coverage"][s] = float((labels==s).sum()) / (labels.size if labels.size else 1)
        else:
            mean_map = np.mean(maps, axis=0)
            out["maps"] = [(mean_map/(np.linalg.norm(mean_map)+1e-12)).tolist()]
            out["coverage"] = {0:1.0}
    except Exception as e:
        print("microstate err:", e)
    return out

# -------------------------
# Focal Delta Index (FDI)
# -------------------------
def compute_focal_delta_index(dfbands: pd.DataFrame, ch_names: List[str]):
    out = {"fdi": {}, "alerts": [], "max_idx": None, "max_val": None, "asymmetry": {}}
    try:
        if dfbands is None or dfbands.empty:
            return out
        # find delta absolute column
        delta_col = None
        for c in dfbands.columns:
            if "delta" in c.lower() and "abs" in c.lower():
                delta_col = c; break
        if delta_col is None:
            return out
        delta = np.array(dfbands[delta_col].values, dtype=float)
        gm = float(np.nanmean(delta)) if delta.size else 1e-9
        for i, v in enumerate(delta):
            fdi = float(v / (gm if gm>0 else 1e-9))
            out["fdi"][i] = fdi
            if fdi > 2.0:
                ch = ch_names[i] if i < len(ch_names) else f"Ch{i}"
                out["alerts"].append({"type":"FDI","channel":ch,"value":float(fdi)})
        pairs = [("T7","T8"),("F3","F4"),("P3","P4"),("O1","O2"),("C3","C4")]
        name_map = {n.upper():i for i,n in enumerate(ch_names)}
        for L,R in pairs:
            if L in name_map and R in name_map:
                li, ri = name_map[L], name_map[R]
                dl = delta[li] if li < len(delta) else 0.0
                dr = delta[ri] if ri < len(delta) else 0.0
                ratio = float(dr/(dl+1e-9)) if dl>0 else (float("inf") if dr>0 else 1.0)
                out["asymmetry"][f"{L}/{R}"] = ratio
                if (isinstance(ratio,float) and (ratio>3.0 or ratio<0.33)) or ratio==float("inf"):
                    out["alerts"].append({"type":"asymmetry","pair":f"{L}/{R}","ratio":ratio})
        max_idx = int(np.argmax(list(out["fdi"].values()))) if out["fdi"] else None
        max_val = out["fdi"].get(max_idx,None) if max_idx is not None else None
        out["max_idx"] = max_idx; out["max_val"] = max_val
    except Exception as e:
        print("compute_fdi err:", e)
    return out

# -------------------------
# ML scoring (heuristic) and model loading
# -------------------------
def load_ml_model(kind: str):
    try:
        import joblib
    except Exception:
        return None
    f = MODEL_DIR / f"{kind}.pkl"
    if f.exists():
        try:
            return joblib.load(str(f))
        except Exception:
            return None
    return None

def score_ml_models(agg: dict, phq_total: int, ad8_total: int):
    scores = {"depression": 0.0, "alzheimers": 0.0}
    try:
        dep_model = load_ml_model("depression")
        alz_model = load_ml_model("alzheimers")
        feat = [agg.get("theta_alpha_ratio",0.0), agg.get("theta_beta_ratio",0.0), agg.get("alpha_rel_mean",0.0), agg.get("gamma_rel_mean",0.0), phq_total/27.0, ad8_total/8.0]
        X = np.array(feat).reshape(1,-1)
        if dep_model is not None:
            try:
                scores["depression"] = float(dep_model.predict_proba(X)[:,1].item())
            except Exception:
                scores["depression"] = 0.0
        else:
            ta = agg.get("theta_alpha_ratio",0.0)
            phq_n = phq_total/27.0 if phq_total else 0.0
            scores["depression"] = float(min(1.0, 0.5*phq_n + 0.3*(ta/1.6)))
        if alz_model is not None:
            try:
                scores["alzheimers"] = float(alz_model.predict_proba(X)[:,1].item())
            except Exception:
                scores["alzheimers"] = 0.0
        else:
            ta = agg.get("theta_alpha_ratio",0.0)
            conn = agg.get("mean_connectivity", 0.0)
            conn_norm = 1.0 - conn if conn is not None else 1.0
            scores["alzheimers"] = float(min(1.0, 0.6*(ta/1.6) + 0.3*conn_norm + 0.1*(ad8_total/8.0 if ad8_total else 0.0)))
    except Exception as e:
        print("score_ml err:", e)
    return scores

# -------------------------
# SHAP bar generator
# -------------------------
def generate_shap_bar_from_summary(shap_summary: dict, model_key: str="alzheimers_global", top_n=10):
    if not shap_summary:
        return None
    try:
        feats = shap_summary.get(model_key) or next(iter(shap_summary.values()))
        s = pd.Series(feats).abs().sort_values(ascending=False).head(top_n)
        fig, ax = plt.subplots(figsize=(6,2))
        s.sort_values().plot.barh(ax=ax, color=PRIMARY_BLUE)
        ax.set_xlabel("SHAP (abs)")
        return fig_to_png_bytes(fig)
    except Exception as e:
        print("shap bar err:", e)
        return None

# -------------------------
# Enrichment pipeline for each result
# -------------------------
def ensure_result_fields(res: dict):
    res.setdefault("agg", {})
    res.setdefault("topo_images", {})
    res.setdefault("bar_img", None)
    res.setdefault("conn_img", None)
    res.setdefault("conn_narr", "")
    res.setdefault("focal", {})
    res.setdefault("shap_img", None)
    res.setdefault("shap_table", {})
    res.setdefault("ml_scores", {})
    return res

def enrich_result(res):
    res = ensure_result_fields(res)
    try:
        raw = res.get("raw_obj", None)
        dfbands = None
        if isinstance(res.get("dfbands"), dict):
            try:
                dfbands = pd.DataFrame(res["dfbands"])
            except Exception:
                dfbands = pd.DataFrame()
        elif isinstance(res.get("dfbands"), pd.DataFrame):
            dfbands = res["dfbands"]
        if raw is not None and (dfbands is None or dfbands.empty):
            dfbands = compute_band_powers(raw)
        ch_names = dfbands["channel"].tolist() if (dfbands is not None and not dfbands.empty and "channel" in dfbands.columns) else (raw.ch_names if raw is not None else [])
        agg = {}
        for band in ["Delta","Theta","Alpha","Beta","Gamma"]:
            col = f"{band}_rel"
            agg[f"{band.lower()}_rel_mean"] = float(dfbands[col].mean()) if (dfbands is not None and col in dfbands.columns) else 0.0
        agg["theta_alpha_ratio"] = float(agg.get("theta_rel_mean",0.0) / (agg.get("alpha_rel_mean",1e-12))) if agg.get("alpha_rel_mean",0.0)>0 else 0.0
        agg["theta_beta_ratio"] = float(agg.get("theta_rel_mean",0.0) / (agg.get("beta_rel_mean",1e-12))) if agg.get("beta_rel_mean",0.0)>0 else 0.0
        agg["alpha_rel_mean"] = agg.get("alpha_rel_mean",0.0)
        # alpha asym F3-F4
        alpha_asym = 0.0
        try:
            names = [n.upper() for n in ch_names]
            if "F3" in names and "F4" in names and (dfbands is not None and ("Alpha_rel" in dfbands.columns or "alpha_rel" in dfbands.columns)):
                i3 = names.index("F3"); i4 = names.index("F4")
                a3 = float(dfbands.iloc[i3].get("Alpha_rel", dfbands.iloc[i3].get("alpha_rel",0.0)))
                a4 = float(dfbands.iloc[i4].get("Alpha_rel", dfbands.iloc[i4].get("alpha_rel",0.0)))
                alpha_asym = float(a3 - a4)
        except Exception:
            alpha_asym = 0.0
        agg["alpha_asym_f3_f4"] = alpha_asym
        res["agg"] = agg
        # focal delta
        res["focal"] = compute_focal_delta_index(dfbands, ch_names)
        # connectivity
        if raw is not None:
            conn_mat, conn_narr, conn_img, mean_conn = compute_functional_connectivity(raw, band=BANDS["Alpha"])
            res["conn_narr"] = conn_narr
            res["conn_img"] = conn_img or placeholder_image("Connectivity image not ready")
            agg["mean_connectivity"] = mean_conn
        else:
            res["conn_narr"] = "(no raw)"
            res["conn_img"] = placeholder_image("Connectivity not available")
            agg["mean_connectivity"] = 0.0
        # microstate
        if raw is not None and HAS_MNE:
            try:
                res["microstate"] = simple_microstate_analysis(raw, n_states=4)
            except Exception:
                res["microstate"] = None
        else:
            res["microstate"] = None
        # SHAP
        shap_summary = None
        if SHAP_JSON.exists():
            try:
                with open(SHAP_JSON, "r", encoding="utf-8") as f:
                    shap_summary = json.load(f)
            except Exception:
                shap_summary = None
        if shap_summary:
            key = "alzheimers_global" if agg.get("theta_alpha_ratio",0.0) > 1.3 else "depression_global"
            res["shap_img"] = generate_shap_bar_from_summary(shap_summary, key, top_n=10)
            res["shap_table"] = shap_summary.get(key, {})
        else:
            res["shap_img"] = None
            res["shap_table"] = {}
        # topo images
        topo_imgs = {}
        for band in ["Delta","Theta","Alpha","Beta","Gamma"]:
            if dfbands is not None and f"{band}_rel" in dfbands.columns:
                topo_imgs[band] = topomap_image_for_band(dfbands[f"{band}_rel"].values, ch_names, band_name=band)
        res["topo_images"] = topo_imgs
        # bar img
        res["bar_img"] = bar_comparison_chart(agg.get("theta_alpha_ratio",0.0), agg.get("alpha_asym_f3_f4",0.0)) or placeholder_image("Normative bar not ready")
        # ML scoring
        phq_total = sum(int(st.session_state.get(f"phq_{i}",0)) for i in range(1,10))
        # map special textual PHQ answers for scoring if present
        # (the main UI computes a mapped numeric summary; still we compute again defensively)
        ad8_total = sum(int(st.session_state.get(f"ad8_{i}",0)) for i in range(1,9))
        res["ml_scores"] = score_ml_models(agg, phq_total, ad8_total)
    except Exception as e:
        print("enrich_result error:", e)
        res["error"] = str(e)
    return res

# -------------------------
# PDF generator (unique style names)
# -------------------------
def safe_bytes_img(img_bytes):
    if not img_bytes:
        return None
    try:
        return io.BytesIO(img_bytes)
    except Exception:
        return None

def attach_logo_rl(logo_path: Path, width=72, height=72):
    try:
        if logo_path.exists():
            from PIL import Image as PILImage
            pil = PILImage.open(str(logo_path))
            buf = io.BytesIO(); pil.save(buf, format="PNG"); buf.seek(0)
            return RLImage(buf, width=width, height=height)
    except Exception:
        return None
    return None

def generate_pdf_report_final(result: dict,
                              patient_info: dict,
                              phq_total: int,
                              ad8_total: int,
                              lang: str = "en",
                              amiri_path: Optional[Path] = None,
                              logo_path: Optional[Path] = None) -> bytes:
    if not HAS_REPORTLAB:
        raise RuntimeError("reportlab is required for PDF export.")
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=28, bottomMargin=28, leftMargin=36, rightMargin=36)
    styles = getSampleStyleSheet()
    base_font = "Helvetica"
    # register Amiri font if present
    if amiri_path and amiri_path.exists():
        try:
            pdfmetrics.registerFont(TTFont("Amiri", str(amiri_path)))
            base_font = "Amiri"
        except Exception:
            base_font = "Helvetica"
    # unique style names to avoid duplication errors
    styles.add(ParagraphStyle(name="CustomTitle", fontName=base_font, fontSize=16, textColor=rl_colors.HexColor(PRIMARY_BLUE), spaceAfter=6))
    styles.add(ParagraphStyle(name="CustomHeader", fontName=base_font, fontSize=11, textColor=rl_colors.HexColor(PRIMARY_BLUE), spaceAfter=4))
    styles.add(ParagraphStyle(name="CustomNormal", fontName=base_font, fontSize=10, leading=12))
    styles.add(ParagraphStyle(name="CustomSmall", fontName=base_font, fontSize=9, leading=11, textColor=rl_colors.grey))

    story = []
    # Header
    title_text = "NeuroEarly Pro — Clinical QEEG Report"
    if lang=="ar" and HAS_ARABIC:
        title_text = fix_arabic_text("تقرير QEEG السريري — NeuroEarly Pro")
    title = Paragraph(f"<b>{title_text}</b>", styles["CustomTitle"])
    logo_rl = attach_logo_rl(logo_path if logo_path else LOGO_PATH)
    try:
        if logo_rl:
            header = Table([[title, logo_rl]], colWidths=[4.8*RL_INCH, 1.4*RL_INCH])
            header.setStyle(TableStyle([("VALIGN", (0,0), (-1,-1), "MIDDLE")]))
            story.append(header)
        else:
            story.append(title)
    except Exception:
        story.append(title)
    story.append(Spacer(1,8))

    # Executive summary
    ml_score = result.get("ml_scores",{}).get("alzheimers") or result.get("ml_scores",{}).get("depression") or 0.0
    ml_disp = f"{ml_score*100:.1f}%" if isinstance(ml_score,(int,float)) else str(ml_score)
    exec_lines = f"<b>Final ML Risk Score:</b> {ml_disp}   <b>PHQ-9:</b> {phq_total}   <b>AD8:</b> {ad8_total}"
    if lang=="ar" and HAS_ARABIC:
        exec_lines = fix_arabic_text("الملخص التنفيذي") + " - " + exec_lines
    story.append(Paragraph(exec_lines, styles["CustomNormal"]))
    story.append(Spacer(1,8))

    # Patient info
    pid = patient_info.get("id","—"); dob = patient_info.get("dob","—"); sex = patient_info.get("sex","—")
    meds = patient_info.get("meds","").strip().splitlines(); labs = patient_info.get("labs","").strip().splitlines()
    meds_s = ", ".join(meds[:6]) + ("..." if len(meds)>6 else "")
    labs_s = ", ".join(labs[:6]) + ("..." if len(labs)>6 else "")
    ptab = [["Field","Value"], ["Patient ID", pid], ["DOB", dob], ["Sex", sex], ["Medications", meds_s or "—"], ["Blood tests", labs_s or "—"]]
    t = Table(ptab, colWidths=[1.6*RL_INCH, 4.2*RL_INCH])
    t.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), rl_colors.HexColor("#eaf6ff")), ("GRID", (0,0), (-1,-1), 0.25, rl_colors.lightgrey)]))
    story.append(t); story.append(Spacer(1,8))

    # Metrics table
    story.append(Paragraph("<b>QEEG Key Metrics</b>", styles["CustomHeader"]))
    agg = result.get("agg",{})
    metrics_rows = [["Metric","Value","Note"]]
    mtempl = [("theta_alpha_ratio","Theta/Alpha Ratio","Slowing indicator"), ("theta_beta_ratio","Theta/Beta Ratio","Stress/inattention"),
              ("alpha_asym_f3_f4","Alpha Asymmetry (F3-F4)","Left-right asymmetry"), ("gamma_rel_mean","Gamma Relative Mean","Cognition-related"),
              ("mean_connectivity","Mean Connectivity (alpha)","Functional coherence")]
    for key,label,note in mtempl:
        val = agg.get(key,"N/A")
        try:
            display_val = f"{float(val):.4f}"
        except Exception:
            display_val = str(val)
        metrics_rows.append([label, display_val, note])
    mt = Table(metrics_rows, colWidths=[2.8*RL_INCH, 1.2*RL_INCH, 2.0*RL_INCH])
    mt.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0), rl_colors.HexColor("#eef7ff")), ("GRID",(0,0),(-1,-1),0.25,rl_colors.grey)]))
    story.append(mt); story.append(Spacer(1,8))

    # Normative bar
    if result.get("bar_img"):
        story.append(Paragraph("<b>Normative Comparison</b>", styles["CustomHeader"]))
        bi = safe_bytes_img(result.get("bar_img"))
        if bi:
            story.append(RLImage(bi, width=5.6*RL_INCH, height=1.6*RL_INCH)); story.append(Spacer(1,6))

    # Topo maps
    topo = result.get("topo_images",{}) or {}
    if topo:
        story.append(Paragraph("<b>Topography Maps</b>", styles["CustomHeader"]))
        imgs = [safe_bytes_img(topo.get(b)) for b in ["Delta","Theta","Alpha","Beta","Gamma"] if topo.get(b)]
        rows=[]; row=[]
        for im in imgs:
            if im:
                row.append(RLImage(im, width=2.6*RL_INCH, height=1.6*RL_INCH))
            else:
                row.append("")
            if len(row)==2:
                rows.append(row); row=[]
        if row: rows.append(row)
        for r in rows:
            tbl = Table([r], colWidths=[3*RL_INCH,3*RL_INCH])
            tbl.setStyle(TableStyle([("VALIGN",(0,0),(-1,-1),"MIDDLE")]))
            story.append(tbl)
        story.append(Spacer(1,6))

    # Connectivity
    if result.get("conn_img"):
        story.append(Paragraph("<b>Functional Connectivity (Alpha)</b>", styles["CustomHeader"]))
        ci = safe_bytes_img(result.get("conn_img"))
        if ci:
            story.append(RLImage(ci, width=5.6*RL_INCH, height=2.4*RL_INCH)); story.append(Spacer(1,6))

    # SHAP
    if result.get("shap_img"):
        story.append(Paragraph("<b>Explainable AI — SHAP top contributors</b>", styles["CustomHeader"]))
        si = safe_bytes_img(result.get("shap_img"))
        if si:
            story.append(RLImage(si, width=5.6*RL_INCH, height=1.8*RL_INCH)); story.append(Spacer(1,6))
    elif result.get("shap_table"):
        stbl = [["Feature","Importance"]]
        for k,v in list(result.get("shap_table",{}).items())[:10]:
            try:
                stbl.append([k, f"{float(v):.4f}"])
            except Exception:
                stbl.append([k, str(v)])
        t3 = Table(stbl, colWidths=[3.6*RL_INCH,2.0*RL_INCH]); t3.setStyle(TableStyle([("GRID",(0,0),(-1,-1),0.25,rl_colors.grey),("BACKGROUND",(0,0),(-1,0),rl_colors.HexColor("#eef7ff"))]))
        story.append(Paragraph("<b>SHAP contributors (table)</b>", styles["CustomHeader"])); story.append(t3); story.append(Spacer(1,6))

    # Tumor / focal
    if result.get("focal"):
        story.append(Paragraph("<b>Focal Delta / Tumor indicators</b>", styles["CustomHeader"]))
        narrative = f"Max FDI: {result['focal'].get('max_val')} at idx {result['focal'].get('max_idx')}"
        story.append(Paragraph(narrative, styles["CustomNormal"]))
        if result["focal"].get("alerts"):
            for a in result["focal"]["alerts"]:
                story.append(Paragraph(f"- {a}", styles["CustomNormal"]))
        story.append(Spacer(1,6))

    # Microstate
    if result.get("microstate"):
        ms = result["microstate"]
        story.append(Paragraph("<b>Microstate summary</b>", styles["CustomHeader"]))
        story.append(Paragraph(f"Number of states: {ms.get('n_states','—')}. Coverage: {ms.get('coverage',{})}", styles["CustomNormal"]))
        story.append(Spacer(1,6))

    # Recommendations
    story.append(Paragraph("<b>Structured Clinical Recommendations</b>", styles["CustomHeader"]))
    recs = result.get("recommendations") or [
        "Correlate QEEG findings with PHQ-9 and AD8 scores.",
        "Check B12, TSH and metabolic panel to exclude reversible causes.",
        "If ML Risk Score > 25% and Theta/Alpha > 1.4 => consider MRI / FDG-PET referral.",
        "Follow-up in 3-6 months for moderate risk cases."
    ]
    for r in recs:
        story.append(Paragraph(f"- {r}", styles["CustomNormal"]))
    story.append(Spacer(1,8))

    # Footer
    story.append(Paragraph("Prepared and designed by Golden Bird LLC — Oman | 2025", styles["CustomSmall"]))
    story.append(Spacer(1,6))
    try:
        doc.build(story)
    except Exception as e:
        print("PDF build exception:", e, traceback.format_exc())
    buffer.seek(0)
    data = buffer.getvalue()
    buffer.close()
    return data

# -------------------------
# UI Layout
# -------------------------
st.set_page_config(page_title="NeuroEarly Pro — Clinical", layout="wide", initial_sidebar_state="expanded")

# Header (title + logo)
title_html = f"""
<div style="display:flex; align-items:center; gap:12px;">
  <div style="flex:1;">
    <h1 style="margin:0; color:{PRIMARY_BLUE};">🧠 NeuroEarly Pro</h1>
    <div style="color:#666;">Clinical QEEG Assistant — Golden Bird LLC</div>
  </div>
"""
if LOGO_PATH.exists():
    # Use POSIX path so Streamlit can load file directly
    title_html += f'<div><img src="{str(LOGO_PATH.as_posix())}" style="height:72px;"/></div></div>'
else:
    title_html += "</div>"

st.markdown(title_html, unsafe_allow_html=True)
st.markdown("---")

# Sidebar (left) — compact
with st.sidebar:
    st.header("Settings")
    lang_choice = st.selectbox("Language / اللغة", ["English","Arabic"], index=0 if st.session_state["lang"]=="en" else 1)
    st.session_state["lang"] = "en" if lang_choice=="English" else "ar"
    st.text_input("Patient Name (optional)", key="patient_name", placeholder="Name (not printed if empty)")
    st.text_input("Patient ID", key="patient_id", placeholder="ID (will appear in report)")
    dob_default = date(1980,1,1)
    st.date_input("Date of Birth", value=dob_default, key="dob", min_value=date(1900,1,1), max_value=date(2025,12,31))
    st.selectbox("Sex", ["Unknown","Male","Female","Other"], key="sex")
    st.markdown("---")
    st.subheader("Medications")
    st.text_area("Current meds (one per line)", key="meds", height=120)
    st.subheader("Blood tests")
    st.text_area("Labs summary (one per line)", key="labs", height=120)
    st.markdown("---")
    st.markdown("Prepared by Golden Bird LLC — NeuroEarly Pro")

# Main area: upload -> questions -> process
st.subheader("Upload EDF file(s)")
uploaded = st.file_uploader("Drag & drop .edf files (multiple allowed)", type=["edf"], accept_multiple_files=True)
if uploaded:
    st.session_state["uploaded_files"] = uploaded

st.markdown("### Questionnaires")
# PHQ-9 with requested modifications
phq_texts = [
    "Little interest or pleasure in doing things",
    "Feeling down, depressed, or hopeless",
    "Sleep changes (choose): Insomnia / Hypersomnia / Normal",
    "Feeling tired or having little energy",
    "Appetite changes (choose): Overeating / Under-eating / Normal",
    "Feeling bad about yourself — or that you are a failure",
    "Trouble concentrating on things",
    "Moving or speaking slowly OR being fidgety/restless (choose)",
    "Thoughts that you would be better off dead"
]
for i, t in enumerate(phq_texts):
    key = f"phq_{i+1}"
    if i+1 == 3:
        opts = ["Insomnia","Hypersomnia","Normal"]
        st.selectbox(f"Q{i+1}: {t}", opts, index=2 if st.session_state.get(key) is None else opts.index(st.session_state.get(key)), key=key)
    elif i+1 == 5:
        opts = ["Overeating","Under-eating","Normal"]
        st.selectbox(f"Q{i+1}: {t}", opts, index=2 if st.session_state.get(key) is None else opts.index(st.session_state.get(key)), key=key)
    elif i+1 == 8:
        opts = ["Slow speech/movement","Restlessness"]
        st.selectbox(f"Q{i+1}: {t}", opts, index=0 if st.session_state.get(key) is None else opts.index(st.session_state.get(key)), key=key)
    else:
        st.radio(f"Q{i+1}: {t}", [0,1,2,3], index=0, key=key)

# AD8
st.markdown("### AD8 (Cognitive screening)")
ad8_texts = [
    "Problems with judgment (e.g. bad decision making)",
    "Less interest in hobbies/activities",
    "Repeats questions, stories, or statements",
    "Trouble learning to use tools/ appliances",
    "Forgets correct month or year",
    "Difficulty handling complicated financial affairs",
    "Difficulty remembering appointments",
    "Daily problems with thinking and memory"
]
for i,t in enumerate(ad8_texts):
    st.radio(f"A{i+1}: {t}", [0,1], index=0, key=f"ad8_{i+1}")

st.markdown("---")
c1, c2 = st.columns([1,1])
with c1:
    if st.button("Process EDF(s) and Analyze"):
        st.session_state["results"] = []
        uploads = st.session_state.get("uploaded_files") or []
        if not uploads:
            st.error("Please upload at least one EDF file.")
        else:
            prog = st.progress(0)
            total = len(uploads)
            for idx, up in enumerate(uploads):
                try:
                    raw, err = read_edf_bytes(up)
                    if err:
                        st.error(f"{up.name}: {err}")
                        continue
                    dfbands = compute_band_powers(raw)
                    res = {"filename": up.name, "raw_obj": raw, "dfbands": dfbands.to_dict(orient="list") if not dfbands.empty else {}, "connectivity_narr": ""}
                    # enrich
                    res = enrich_result(res)
                    st.session_state["results"].append(res)
                    prog.progress(int(((idx+1)/total)*100))
                except Exception as e:
                    st.error(f"Processing error for {up.name}: {e}")
                    print(traceback.format_exc())
            st.success("Processing complete.")
with c2:
    st.write("After processing, view results in the tabs below and export a clinical PDF report.")

# Results tabs
if st.session_state.get("results"):
    r0 = st.session_state["results"][0]
    patient_info = {
        "id": st.session_state.get("patient_id",""),
        "name": st.session_state.get("patient_name",""),
        "dob": str(st.session_state.get("dob","")),
        "sex": st.session_state.get("sex",""),
        "meds": st.session_state.get("meds",""),
        "labs": st.session_state.get("labs","")
    }
    # compute numeric PHQ total
    phq_total_numeric = 0
    for i in range(1,10):
        key = f"phq_{i}"
        v = st.session_state.get(key)
        if i in (3,5):  # special mapping: Normal -> 0, changed -> 2
            if isinstance(v,str):
                phq_total_numeric += 2 if v!="Normal" else 0
        elif i==8:
            if isinstance(v,str):
                phq_total_numeric += 2 if v=="Slow speech/movement" else 1
        else:
            try:
                phq_total_numeric += int(v)
            except Exception:
                phq_total_numeric += 0
    ad8_total = sum(int(st.session_state.get(f"ad8_{i}",0)) for i in range(1,9))

    tabs = st.tabs(["Overview","Connectivity","XAI (SHAP)","Microstates","Export"])
    with tabs[0]:
        st.subheader("Overview")
        ml_val = (r0.get("ml_scores",{}).get("alzheimers") or r0.get("ml_scores",{}).get("depression") or 0.0)
        st.metric("Final ML Risk Score", f"{ml_val*100:.1f}%")
        qdf = pd.DataFrame([{
            "Theta/Alpha": r0.get("agg",{}).get("theta_alpha_ratio"),
            "Theta/Beta": r0.get("agg",{}).get("theta_beta_ratio"),
            "Alpha Asym F3-F4": r0.get("agg",{}).get("alpha_asym_f3_f4"),
            "Mean Connectivity": r0.get("agg",{}).get("mean_connectivity")
        }])
        st.table(qdf.T.rename(columns={0:"Value"}))
        colA, colB = st.columns([1,1])
        with colA:
            if r0.get("bar_img"):
                st.image(r0["bar_img"], caption="Normative comparison", width=520)
        with colB:
            if r0.get("topo_images") and r0["topo_images"].get("Alpha"):
                st.image(r0["topo_images"].get("Alpha"), caption="Alpha topography", width=320)
        st.markdown("**QEEG Interpretation (brief):**")
        narr = []
        narr.append(f"Theta/Alpha ratio: {fmt(r0.get('agg',{}).get('theta_alpha_ratio',0.0),3)}")
        narr.append(f"Mean Connectivity (alpha): {fmt(r0.get('agg',{}).get('mean_connectivity',0.0),4)}")
        if r0.get("focal",{}).get("alerts"):
            narr.append("Focal Delta Alerts detected: " + ", ".join([str(a) for a in r0.get("focal",{}).get("alerts")]))
        st.write("\n".join(narr))

    with tabs[1]:
        st.subheader("Connectivity")
        st.write(r0.get("conn_narr",""))
        if r0.get("conn_img"):
            st.image(r0["conn_img"], caption="Connectivity matrix", width=680)
        else:
            st.info("Connectivity image not available.")

    with tabs[2]:
        st.subheader("Explainable AI (SHAP)")
        if r0.get("shap_img"):
            st.image(r0["shap_img"], caption="SHAP top contributors", width=680)
        elif r0.get("shap_table"):
            st.table(pd.DataFrame(list(r0.get("shap_table",{}).items()), columns=["Feature","Importance"]).set_index("Feature"))
        else:
            st.info("SHAP not available. Place shap_summary.json in repo or upload it.")

    with tabs[3]:
        st.subheader("Microstates")
        if r0.get("microstate"):
            st.write(r0["microstate"])
        else:
            st.info("Microstate analysis not available.")

    with tabs[4]:
        st.subheader("Export / PDF")
        pdf_lang = st.selectbox("Select PDF language", options=["English","Arabic"], index=0)
        lang_code = "en" if pdf_lang=="English" else "ar"
        amiri = AMIRI_PATH if AMIRI_PATH.exists() else None
        logo = LOGO_PATH if LOGO_PATH.exists() else None
        if st.button("Generate & Download Clinical PDF"):
            try:
                pdf_bytes = generate_pdf_report_final(r0, patient_info, phq_total_numeric, ad8_total, lang=lang_code, amiri_path=amiri, logo_path=logo)
                if pdf_bytes:
                    st.success("PDF generated successfully.")
                    st.download_button("⬇️ Download Clinical Report (PDF)", data=pdf_bytes, file_name=f"NeuroEarly_Report_{now_ts()}.pdf", mime="application/pdf")
                else:
                    st.error("PDF generation returned no data.")
            except Exception as e:
                st.error(f"PDF generation failed: {e}")
                st.text(traceback.format_exc())
else:
    st.info("No processed results yet. Upload EDF and press 'Process EDF(s) and Analyze'.")

# End of app.py (v3 final)

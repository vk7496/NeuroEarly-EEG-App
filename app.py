# app.py — NeuroEarly Pro (v5 Professional)
# Full bilingual (English default / Arabic optional RTL with Amiri),
# Topomaps, Functional Connectivity, Microstates (simple), Focal Delta (tumor hint),
# SHAP (from shap_summary.json) visualization, PDF generation (reportlab),
# Sidebar left: patient info, language switch, meds, labs. Designed for Streamlit.

import os, io, sys, json, math, tempfile, traceback
from pathlib import Path
from datetime import datetime, date
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

import streamlit as st

# Optional heavy libs
HAS_MNE = False
HAS_SHAP = False
HAS_REPORTLAB = False
HAS_ARABIC = False
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
    from reportlab.lib.enums import TA_RIGHT
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    HAS_REPORTLAB = True
except Exception:
    HAS_REPORTLAB = False

try:
    from arabic_reshaper import reshape
    from bidi.algorithm import get_display
    HAS_ARABIC = True
except Exception:
    HAS_ARABIC = False

try:
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False

from scipy.signal import coherence

# constants & paths
ROOT = Path(__file__).parent
ASSETS = ROOT / "assets"
AMIRI_TTF = ASSETS / "Amiri-Regular.ttf"
LOGO_CANDIDATES = [ASSETS / "goldenbird_logo.png", ASSETS / "goldenbird_logo.svg", ASSETS / "GoldenBird_logo.png"]
SHAP_JSON = ROOT / "shap_summary.json"
MODEL_DIR = ROOT / "models"

PRIMARY_BLUE = "#0b63d6"
LIGHT_BG = "#eef7ff"
ALERT_RED = "#d62d20"

BANDS = {
    "Delta": (1.0,4.0),
    "Theta": (4.0,8.0),
    "Alpha": (8.0,13.0),
    "Beta": (13.0,30.0),
    "Gamma": (30.0,45.0),
}

HPI_CHANNELS = ["T7","T8","TP9","TP10"]

# session defaults
if "results" not in st.session_state:
    st.session_state["results"] = []
if "lang" not in st.session_state:
    st.session_state["lang"] = "en"

# helpers
def now_ts(fmt="%Y%m%d_%H%M%S"):
    return datetime.utcnow().strftime(fmt)

def fmtf(x, prec=4):
    try:
        return f"{float(x):.{prec}f}"
    except Exception:
        return str(x)

def t(en: str, ar: str) -> str:
    return ar if st.session_state.get("lang","en") == "ar" else en

def fix_arabic_text(txt: str) -> str:
    if not txt: return txt
    if HAS_ARABIC:
        try:
            return get_display(reshape(txt))
        except Exception:
            return txt
    return txt

def find_logo_path() -> Optional[Path]:
    for p in LOGO_CANDIDATES:
        if p.exists():
            return p
    return None

def load_logo_bytes(logo_path: Optional[Path]) -> Optional[bytes]:
    if not logo_path: return None
    try:
        img = Image.open(logo_path).convert("RGBA")
        buf = io.BytesIO(); img.save(buf, format="PNG"); buf.seek(0)
        return buf.getvalue()
    except Exception:
        return None

# EDF reading safe (write to temp file for libraries that expect path)
def read_edf_bytes(uploaded) -> Tuple[Optional[mne.io.Raw], Optional[str]]:
    """
    Read uploaded EDF using MNE if available, else return None.
    Returns (raw, msg) with debug info in Streamlit.
    """
    if not uploaded:
        return None, "No file uploaded"

    buf = io.BytesIO(uploaded.getvalue())

    try:
        if HAS_MNE:
            st.info("📂 Reading EDF file... please wait")

            # Read EDF file safely
            raw = mne.io.read_raw_edf(buf, preload=True, verbose=False)

            # ✅ Debug info: check shape, mean, and sample data
            data, times = raw.get_data(return_times=True)
            st.success(f"✅ EDF loaded successfully! Shape: {data.shape}")
            st.write("📡 Sampling rate (Hz):", raw.info.get("sfreq"))
            st.write("🧩 Mean amplitude:", float(np.mean(data)))
            st.write("🔸 First 10 samples of channel 0:", data[0][:10].tolist())

            # Optional: check channels list
            st.write("🧠 Channels:", raw.ch_names)

            return raw, None

        else:
            return None, "❌ MNE not available in environment"

    except Exception as e:
        st.error(f"❌ Error reading EDF: {str(e)}")
        return None, str(e)


# compute band powers
def compute_band_powers(raw, bands=BANDS):
    if not HAS_MNE or raw is None:
        return pd.DataFrame()
    try:
        picks = mne.pick_types(raw.info, eeg=True, meg=False)
        ch_names = [raw.ch_names[p] for p in picks]
        psds, freqs = mne.time_frequency.psd_welch(raw, picks=picks, fmin=1.0, fmax=45.0, verbose=False)
        rows=[]
        for i,ch in enumerate(ch_names):
            pxx = psds[i]
            total = float(np.trapz(pxx, freqs)) if freqs.size else 0.0
            row = {"channel": ch}
            for bname,(lo,hi) in bands.items():
                mask = (freqs>=lo) & (freqs<hi)
                pband = float(np.trapz(pxx[mask], freqs[mask])) if mask.sum() else 0.0
                row[f"{bname}_abs"]=pband
                row[f"{bname}_rel"]= (pband/total) if total>0 else 0.0
            rows.append(row)
        return pd.DataFrame(rows)
    except Exception as e:
        print("compute_band_powers err:", e)
        return pd.DataFrame()

# topomap plotting utilities
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

def topomap_with_mne(raw, values, ch_names, band_name="Alpha"):
    try:
        info = mne.create_info(ch_names, sfreq=int(raw.info.get("sfreq",256)), ch_types=["eeg"]*len(ch_names))
        try:
            montage = mne.channels.make_standard_montage("standard_1020")
            info.set_montage(montage, on_missing="ignore")
        except Exception:
            pass
        arr = np.array(values)
        ev = mne.EvokedArray(np.reshape(arr,(len(arr),1)), info, tmin=0.0)
        fig = ev.plot_topomap(times=0.0, ch_type="eeg", show=False, vmin=np.min(arr), vmax=np.max(arr), contours=0, titles=band_name)
        return fig_to_png_bytes(fig)
    except Exception as e:
        print("topomap_with_mne err:", e)
        return None

def simple_heat_topo(values, ch_names, band_name="Alpha"):
    try:
        fig, ax = plt.subplots(figsize=(8,2.2))
        x = np.arange(len(values))
        ax.bar(x, values, color=PRIMARY_BLUE)
        ax.set_xticks(x)
        ax.set_xticklabels(ch_names, rotation=90, fontsize=6)
        ax.set_title(f"{band_name} topography (approx)")
        return fig_to_png_bytes(fig)
    except Exception as e:
        print("simple_heat_topo err:", e)
        return None

# connectivity
def compute_functional_connectivity(raw, band=(8.0,13.0)):
    try:
        data = raw.get_data(picks=mne.pick_types(raw.info, eeg=True))
        sf = int(raw.info.get("sfreq",256))
        nchan = data.shape[0]
    except Exception as e:
        return None, "(no raw)", None, 0.0
    conn_mat = np.zeros((nchan,nchan))
    narr = ""
    mean_conn = 0.0
    # try MNE spectral_connectivity (wPLI)
    try:
        from mne.connectivity import spectral_connectivity
        con, freqs, times, n_epochs, n_tapers = spectral_connectivity([raw], method="wpli", mode="multitaper", sfreq=sf, fmin=band[0], fmax=band[1], faverage=True, verbose=False)
        con = np.squeeze(con)
        if con.shape == (nchan,nchan):
            conn_mat = con
            narr = f"wPLI {band[0]}-{band[1]} Hz (MNE)"
    except Exception:
        try:
            for i in range(nchan):
                for j in range(i, nchan):
                    f, Cxy = coherence(data[i], data[j], fs=sf, nperseg=min(1024, max(256, data.shape[1])))
                    mask = (f>=band[0]) & (f<=band[1])
                    val = float(np.nanmean(Cxy[mask])) if mask.sum() else 0.0
                    conn_mat[i,j]=val; conn_mat[j,i]=val
            narr = f"Coherence {band[0]}-{band[1]} Hz (scipy fallback)"
        except Exception as e:
            narr = f"(connectivity failed: {e})"
    try:
        mean_conn = float(np.nanmean(conn_mat)) if conn_mat.size else 0.0
    except Exception:
        mean_conn = 0.0
    # render image
    try:
        fig, ax = plt.subplots(figsize=(6,4))
        im = ax.imshow(conn_mat, cmap="viridis", interpolation="nearest", aspect="auto")
        ax.set_title("Functional Connectivity")
        chs = raw.ch_names if hasattr(raw, "ch_names") else [f"Ch{i}" for i in range(conn_mat.shape[0])]
        ax.set_xticks(range(min(len(chs),40))); ax.set_xticklabels(chs[:min(len(chs),40)], rotation=90, fontsize=6)
        ax.set_yticks(range(min(len(chs),40))); ax.set_yticklabels(chs[:min(len(chs),40)], fontsize=6)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        conn_img = fig_to_png_bytes(fig)
    except Exception:
        conn_img = None
    return conn_mat, narr, conn_img, mean_conn

# microstates (simple heuristic)
def simple_microstate_analysis(raw, n_states=4):
    out={"maps":[], "coverage":{}, "n_states":n_states}
    try:
        data = raw.get_data()
        gfp = np.std(data, axis=0)
        thr = np.percentile(gfp,75)
        peaks = np.where(gfp>=thr)[0]
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
                out["coverage"][s]=float((labels==s).sum())/ (labels.size if labels.size else 1)
        else:
            mean_map = np.mean(maps, axis=0)
            out["maps"]=[(mean_map/(np.linalg.norm(mean_map)+1e-12)).tolist()]
            out["coverage"]={0:1.0}
    except Exception as e:
        print("microstate err:", e)
    return out

# focal delta (tumor hints)
def compute_focal_delta_index(dfbands: pd.DataFrame, ch_names: List[str]):
    out={"fdi":{}, "alerts":[], "max_idx":None, "max_val":None, "asymmetry":{}}
    try:
        if dfbands is None or dfbands.empty:
            return out
        delta_col = next((c for c in dfbands.columns if "delta" in c.lower() and "abs" in c.lower()), None)
        if not delta_col:
            return out
        delta = np.array(dfbands[delta_col].values, dtype=float)
        gm = float(np.nanmean(delta)) if delta.size else 1e-9
        for i,v in enumerate(delta):
            fdi = float(v/(gm if gm>0 else 1e-9))
            out["fdi"][i]=fdi
            if fdi>2.0:
                ch = ch_names[i] if i<len(ch_names) else f"Ch{i}"
                out["alerts"].append({"type":"FDI","channel":ch,"value":float(fdi)})
        pairs=[("T7","T8"),("F3","F4"),("P3","P4"),("O1","O2"),("C3","C4")]
        name_map={n.upper():i for i,n in enumerate(ch_names)}
        for L,R in pairs:
            if L in name_map and R in name_map:
                li,ri = name_map[L], name_map[R]
                dl = delta[li] if li<len(delta) else 0.0
                dr = delta[ri] if ri<len(delta) else 0.0
                ratio = float(dr/(dl+1e-9)) if dl>0 else (float("inf") if dr>0 else 1.0)
                out["asymmetry"][f"{L}/{R}"]=ratio
                if (isinstance(ratio,float) and (ratio>3.0 or ratio<0.33)) or ratio==float("inf"):
                    out["alerts"].append({"type":"asymmetry","pair":f"{L}/{R}","ratio":ratio})
        max_idx = int(np.argmax(list(out["fdi"].values()))) if out["fdi"] else None
        max_val = out["fdi"].get(max_idx,None) if max_idx is not None else None
        out["max_idx"]=max_idx; out["max_val"]=max_val
    except Exception as e:
        print("compute_fdi err:", e)
    return out

# HPI
def compute_hpi_from_df(dfbands: pd.DataFrame, ch_names: List[str]):
    try:
        if dfbands is None or dfbands.empty:
            return None
        names=[n.upper() for n in ch_names]
        ratios=[]
        theta_col = next((c for c in dfbands.columns if c.lower().startswith("theta") and "rel" in c.lower()), None)
        alpha_col = next((c for c in dfbands.columns if c.lower().startswith("alpha") and "rel" in c.lower()), None)
        if not theta_col or not alpha_col:
            return None
        for ch in HPI_CHANNELS:
            if ch in names:
                idx = names.index(ch)
                t = float(dfbands.iloc[idx].get(theta_col,0.0))
                a = float(dfbands.iloc[idx].get(alpha_col,1e-12))
                if a<=0: continue
                ratios.append(t/a)
        if not ratios: return None
        return float(np.mean(ratios))
    except Exception as e:
        print("compute_hpi err:", e)
        return None

# ML scoring (heuristic or loaded model)
def load_ml_model(kind: str):
    try:
        import joblib
        f = MODEL_DIR / f"{kind}.pkl"
        if f.exists():
            return joblib.load(str(f))
    except Exception:
        pass
    return None

def score_ml_models(agg: dict, phq_total: int, ad8_total: int):
    scores={"depression":0.0,"alzheimers":0.0}
    try:
        dep_model = load_ml_model("depression")
        alz_model = load_ml_model("alzheimers")
        feat = [agg.get("theta_alpha_ratio",0.0), agg.get("theta_beta_ratio",0.0), agg.get("alpha_rel_mean",0.0), agg.get("gamma_rel_mean",0.0), phq_total/27.0, ad8_total/8.0, agg.get("hpi",0.0) or 0.0]
        X = np.array(feat).reshape(1,-1)
        if dep_model is not None:
            try:
                scores["depression"]=float(dep_model.predict_proba(X)[:,1].item())
            except Exception:
                scores["depression"]=0.0
        else:
            ta=agg.get("theta_alpha_ratio",0.0); phq_n=phq_total/27.0 if phq_total else 0.0
            scores["depression"]=float(min(1.0, 0.5*phq_n + 0.3*(ta/1.6)))
        if alz_model is not None:
            try:
                scores["alzheimers"]=float(alz_model.predict_proba(X)[:,1].item())
            except Exception:
                scores["alzheimers"]=0.0
        else:
            ta=agg.get("theta_alpha_ratio",0.0); conn=agg.get("mean_connectivity",0.0); conn_norm=1.0-conn if conn is not None else 1.0
            hpi=agg.get("hpi",0.0) or 0.0
            scores["alzheimers"]=float(min(1.0, 0.5*(ta/1.6) + 0.25*conn_norm + 0.15*(hpi/2.0) + 0.1*(ad8_total/8.0 if ad8_total else 0.0)))
    except Exception as e:
        print("score_ml err:", e)
    return scores

# SHAP bar from json
def generate_shap_bar_from_summary(shap_summary: dict, model_key: str="alzheimers_global", top_n:int=10):
    if not shap_summary: return None
    try:
        feats = shap_summary.get(model_key) or next(iter(shap_summary.values()))
        s = pd.Series(feats).abs().sort_values(ascending=False).head(top_n)
        fig, ax = plt.subplots(figsize=(6,2.2))
        s.sort_values().plot.barh(ax=ax, color=PRIMARY_BLUE)
        ax.set_xlabel("SHAP importance (abs)")
        return fig_to_png_bytes(fig)
    except Exception as e:
        print("generate_shap err:", e)
        return None

# enrich result (computations + images)
def ensure_result(res: dict):
    res.setdefault("agg",{})
    res.setdefault("topo_imgs",{})
    res.setdefault("bar_img", None)
    res.setdefault("conn_img", None)
    res.setdefault("conn_narr","")
    res.setdefault("focal",{})
    res.setdefault("shap_img",None)
    res.setdefault("shap_table",{})
    res.setdefault("microstate",None)
    res.setdefault("ml_scores",{})
    return res

def enrich_result(res):
    res = ensure_result(res)
    try:
        raw = res.get("raw_obj",None)
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
        agg={}
        for band in ["Delta","Theta","Alpha","Beta","Gamma"]:
            col=f"{band}_rel"
            agg[f"{band.lower()}_rel_mean"]=float(dfbands[col].mean()) if (dfbands is not None and col in dfbands.columns) else 0.0
        alpha_mean = agg.get("alpha_rel_mean",0.0)
        theta_mean = agg.get("theta_rel_mean",0.0)
        beta_mean = agg.get("beta_rel_mean",0.0)
        agg["theta_alpha_ratio"] = float(theta_mean / (alpha_mean if alpha_mean>0 else 1e-12))
        agg["theta_beta_ratio"] = float(theta_mean / (beta_mean if beta_mean>0 else 1e-12))
        agg["alpha_rel_mean"]=alpha_mean
        # alpha asymmetry
        alpha_asym = 0.0
        try:
            names=[n.upper() for n in ch_names]
            if "F3" in names and "F4" in names and (dfbands is not None and not dfbands.empty):
                i3 = names.index("F3"); i4 = names.index("F4")
                a3 = float(dfbands.iloc[i3].get("Alpha_rel", dfbands.iloc[i3].get("alpha_rel",0.0)))
                a4 = float(dfbands.iloc[i4].get("Alpha_rel", dfbands.iloc[i4].get("alpha_rel",0.0)))
                alpha_asym = float(a3 - a4)
        except Exception:
            alpha_asym = 0.0
        agg["alpha_asym_f3_f4"] = alpha_asym
        # HPI
        try:
            hpi = compute_hpi_from_df(dfbands, ch_names)
            agg["hpi"] = float(hpi) if hpi is not None else None
        except Exception:
            agg["hpi"] = None
        res["agg"]=agg
        # focal delta
        res["focal"] = compute_focal_delta_index(dfbands, ch_names)
        # connectivity
        if raw is not None:
            try:
                conn_mat, conn_narr, conn_img, mean_conn = compute_functional_connectivity(raw, band=BANDS["Alpha"])
                res["conn_narr"]=conn_narr
                res["conn_img"]=conn_img or None
                agg["mean_connectivity"]=mean_conn
            except Exception:
                res["conn_narr"]="(connectivity failed)"; res["conn_img"]=None; agg["mean_connectivity"]=0.0
        else:
            res["conn_narr"]="(no raw)"; res["conn_img"]=None; agg["mean_connectivity"]=0.0
        # microstates
        if raw is not None and HAS_MNE:
            try:
                res["microstate"]=simple_microstate_analysis(raw, n_states=4)
            except Exception:
                res["microstate"]=None
        else:
            res["microstate"]=None
        # shap
        shap_summary=None
        if SHAP_JSON.exists():
            try:
                with open(SHAP_JSON,"r",encoding="utf-8") as f:
                    shap_summary=json.load(f)
            except Exception:
                shap_summary=None
        if shap_summary:
            key = "alzheimers_global" if agg.get("theta_alpha_ratio",0.0)>1.3 else "depression_global"
            res["shap_img"]=generate_shap_bar_from_summary(shap_summary, key, top_n=10)
            res["shap_table"]=shap_summary.get(key,{})
        else:
            res["shap_img"]=None; res["shap_table"]={}
        # topo images
        topo_imgs={}
        for band in ["Delta","Theta","Alpha","Beta","Gamma"]:
            if dfbands is not None and f"{band}_rel" in dfbands.columns:
                vals = dfbands[f"{band}_rel"].values
                img = None
                if HAS_MNE and raw is not None:
                    img = topomap_with_mne(raw, vals, ch_names, band_name=band)
                if not img:
                    img = simple_heat_topo(vals, ch_names, band_name=band)
                topo_imgs[band]=img
        res["topo_imgs"]=topo_imgs
        # normative bar (theta/alpha & alpha asym)
        try:
            theta_alpha = agg.get("theta_alpha_ratio",0.0)
            alpha_asym = agg.get("alpha_asym_f3_f4",0.0)
            fig, ax = plt.subplots(figsize=(6,2.2))
            metrics = ["Theta/Alpha","Alpha Asym"]
            vals = [theta_alpha, alpha_asym]
            colors = [PRIMARY_BLUE if v<1.4 else ALERT_RED for v in vals]
            ax.bar(metrics, vals, color=colors)
            ax.axhspan(1.0,1.4,alpha=0.08,color="white")  # normative band simple visual
            ax.set_title("Theta/Alpha and Alpha Asym (F3-F4)")
            bi = fig_to_png_bytes(fig)
            res["bar_img"]=bi
        except Exception:
            res["bar_img"]=None
        # ML scoring
        phq_total = sum(int(st.session_state.get(f"phq_{i}",0) if isinstance(st.session_state.get(f"phq_{i}",0),(int,str)) else 0) for i in range(1,10))
        ad8_total = sum(int(st.session_state.get(f"ad8_{i}",0)) for i in range(1,9))
        res["ml_scores"] = score_ml_models(agg, phq_total, ad8_total)
    except Exception as e:
        print("enrich_result err:", e, traceback.format_exc())
        res["error"]=str(e)
    return res

# PDF functions
def safe_bytes_img(img_bytes):
    if not img_bytes: return None
    try:
        return io.BytesIO(img_bytes)
    except Exception:
        return None

def attach_logo_rl(logo_bytes: Optional[bytes], width=72, height=72):
    if not logo_bytes: return None
    try:
        return RLImage(io.BytesIO(logo_bytes), width=width, height=height)
    except Exception:
        return None

def generate_pdf_report(result: dict, patient_info: dict, phq_total:int, ad8_total:int, lang:str="en", amiri:Optional[Path]=None, logo_bytes:Optional[bytes]=None) -> bytes:
    if not HAS_REPORTLAB:
        raise RuntimeError("reportlab required for PDF export")
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=28, bottomMargin=28, leftMargin=36, rightMargin=36)
    styles = getSampleStyleSheet()
    base_font = "Helvetica"
    use_amiri = False
    if amiri and amiri.exists():
        try:
            pdfmetrics.registerFont(TTFont("Amiri", str(amiri)))
            base_font = "Amiri"; use_amiri=True
        except Exception:
            base_font = "Helvetica"
    # ensure unique style names to avoid "already defined" error
    styles.add(ParagraphStyle(name="TitleX", fontName=base_font, fontSize=16, textColor=rl_colors.HexColor(PRIMARY_BLUE), spaceAfter=6))
    styles.add(ParagraphStyle(name="HeaderX", fontName=base_font, fontSize=11, textColor=rl_colors.HexColor(PRIMARY_BLUE), spaceAfter=4))
    styles.add(ParagraphStyle(name="NormalX", fontName=base_font, fontSize=10, leading=12))
    styles.add(ParagraphStyle(name="SmallX", fontName=base_font, fontSize=9, leading=11, textColor=rl_colors.grey))
    if lang=="ar":
        styles.add(ParagraphStyle(name="ArabicX", fontName=base_font, fontSize=11, alignment=TA_RIGHT, leading=14))
    story=[]
    title_text = "NeuroEarly Pro — Clinical QEEG Report" if lang=="en" else fix_arabic_text("تقرير QEEG السريري — NeuroEarly Pro")
    title = Paragraph(f"<b>{title_text}</b>", styles["TitleX"])
    logo_rl = attach_logo_rl(logo_bytes)
    try:
        if logo_rl:
            header = Table([[title, logo_rl]], colWidths=[4.6*RL_INCH, 1.6*RL_INCH])
            header.setStyle(TableStyle([("VALIGN",(0,0),(-1,-1),"MIDDLE")]))
            story.append(header)
        else:
            story.append(title)
    except Exception:
        story.append(title)
    story.append(Spacer(1,8))
    # executive
    ml_score = result.get("ml_scores",{}).get("alzheimers") or result.get("ml_scores",{}).get("depression") or 0.0
    exec_line = f"<b>Final ML Risk Score:</b> {ml_score*100:.1f}%   <b>PHQ-9:</b> {phq_total}   <b>AD8:</b> {ad8_total}"
    story.append(Paragraph(exec_line, styles["NormalX"]))
    story.append(Spacer(1,8))
    # patient table
    pid = patient_info.get("id","—"); dob = patient_info.get("dob","—"); sex = patient_info.get("sex","—")
    meds = patient_info.get("meds","").strip().splitlines(); labs = patient_info.get("labs","").strip().splitlines()
    meds_s = ", ".join(meds[:6]) + ("..." if len(meds)>6 else "")
    labs_s = ", ".join(labs[:6]) + ("..." if len(labs)>6 else "")
    ptab = [["Field","Value"], ["Patient ID", pid], ["DOB", dob], ["Sex", sex], ["Medications", meds_s or "—"], ["Blood tests", labs_s or "—"]]
    t = Table(ptab, colWidths=[1.6*RL_INCH, 4.2*RL_INCH])
    t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0), rl_colors.HexColor(LIGHT_BG)), ("GRID",(0,0),(-1,-1), 0.25, rl_colors.lightgrey)]))
    story.append(t); story.append(Spacer(1,8))
    # metrics table
    story.append(Paragraph("<b>QEEG Key Metrics</b>", styles["HeaderX"]))
    agg = result.get("agg",{})
    metrics_rows=[["Metric","Value","Note"]]
    mtempl=[("theta_alpha_ratio","Theta/Alpha Ratio","Slowing indicator"),("theta_beta_ratio","Theta/Beta Ratio","Stress/inattention"),
            ("alpha_asym_f3_f4","Alpha Asymmetry (F3-F4)","Left-right asymmetry"),("gamma_rel_mean","Gamma Relative Mean","Cognition-related"),
            ("mean_connectivity","Mean Connectivity (alpha)","Functional coherence")]
    for key,label,note in mtempl:
        val = agg.get(key,"N/A")
        try:
            display_val = f"{float(val):.4f}"
        except Exception:
            display_val = str(val)
        metrics_rows.append([label, display_val, note])
    mt = Table(metrics_rows, colWidths=[2.8*RL_INCH, 1.2*RL_INCH, 2.0*RL_INCH])
    mt.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0), rl_colors.HexColor("#f7fbff")), ("GRID",(0,0),(-1,-1),0.25,rl_colors.grey)]))
    story.append(mt); story.append(Spacer(1,6))
    # HPI
    hpi = agg.get("hpi", None)
    story.append(Paragraph("<b>Hippocampal Proxy Index (HPI)</b>", styles["HeaderX"]))
    if hpi is None:
        story.append(Paragraph("HPI: — (Not enough temporal channels or data)", styles["NormalX"]))
    else:
        color = ALERT_RED if hpi>1.8 else PRIMARY_BLUE
        label = f"HPI (Theta/Alpha temporal channels): {hpi:.4f}" + (" — Elevated" if hpi>1.8 else "")
        story.append(Paragraph(f'<font color="{color}"><b>{label}</b></font>', styles["NormalX"]))
    story.append(Spacer(1,8))
    # normative bar image
    if result.get("bar_img"):
        story.append(Paragraph("<b>Normative comparison</b>", styles["HeaderX"]))
        bi = safe_bytes_img(result.get("bar_img"))
        if bi:
            story.append(RLImage(bi, width=5.6*RL_INCH, height=1.6*RL_INCH)); story.append(Spacer(1,6))
    # topomaps
    topo = result.get("topo_imgs",{}) or {}
    if topo:
        story.append(Paragraph("<b>Topography Maps</b>", styles["HeaderX"]))
        imgs=[safe_bytes_img(topo.get(b)) for b in ["Delta","Theta","Alpha","Beta","Gamma"] if topo.get(b)]
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
    # connectivity
    if result.get("conn_img"):
        story.append(Paragraph("<b>Functional Connectivity (Alpha)</b>", styles["HeaderX"]))
        ci = safe_bytes_img(result.get("conn_img"))
        if ci:
            story.append(RLImage(ci, width=5.6*RL_INCH, height=2.4*RL_INCH)); story.append(Spacer(1,6))
    # SHAP
    if result.get("shap_img"):
        story.append(Paragraph("<b>Explainable AI — SHAP</b>", styles["HeaderX"]))
        si = safe_bytes_img(result.get("shap_img"))
        if si:
            story.append(RLImage(si, width=5.6*RL_INCH, height=1.8*RL_INCH)); story.append(Spacer(1,6))
    elif result.get("shap_table"):
        stbl=[["Feature","Importance"]]
        for k,v in list(result.get("shap_table",{}).items())[:10]:
            try:
                stbl.append([k, f"{float(v):.4f}"])
            except Exception:
                stbl.append([k, str(v)])
        t3 = Table(stbl, colWidths=[3.6*RL_INCH,2.0*RL_INCH])
        t3.setStyle(TableStyle([("GRID",(0,0),(-1,-1),0.25,rl_colors.grey),("BACKGROUND",(0,0),(-1,0),rl_colors.HexColor("#eef7ff"))]))
        story.append(Paragraph("<b>SHAP contributors (table)</b>", styles["HeaderX"])); story.append(t3); story.append(Spacer(1,6))
    # focal/tumor narrative
    if result.get("focal"):
        story.append(Paragraph("<b>Focal Delta / Tumor indicators</b>", styles["HeaderX"]))
        narrative = f"Max FDI: {fmtf(result['focal'].get('max_val',0))} at idx {result['focal'].get('max_idx','—')}"
        story.append(Paragraph(narrative, styles["NormalX"]))
        if result["focal"].get("alerts"):
            for a in result["focal"]["alerts"]:
                story.append(Paragraph(f"- {str(a)}", styles["NormalX"]))
        story.append(Spacer(1,6))
    # microstate
    if result.get("microstate"):
        ms = result["microstate"]
        story.append(Paragraph("<b>Microstate summary</b>", styles["HeaderX"]))
        story.append(Paragraph(f"Number of states: {ms.get('n_states','—')}. Coverage: {ms.get('coverage',{})}", styles["NormalX"]))
        story.append(Spacer(1,6))
    # recommendations
    story.append(Paragraph("<b>Structured Clinical Recommendations</b>", styles["HeaderX"]))
    recs = result.get("recommendations") or [
        "Correlate QEEG findings with PHQ-9 and AD8 scores.",
        "Check vitamin B12, TSH and metabolic panel to exclude reversible causes.",
        "If ML Risk Score > 25% and Theta/Alpha > 1.4 => consider MRI / FDG-PET referral.",
        "Follow-up in 3-6 months for moderate risk cases."
    ]
    for r in recs:
        story.append(Paragraph(f"- {fix_arabic_text(r) if lang=='ar' and HAS_ARABIC else r}", styles["NormalX"]))
    story.append(Spacer(1,8))
    story.append(Paragraph("Prepared and designed by Golden Bird LLC — Oman | 2025", styles["SmallX"]))
    story.append(Spacer(1,6))
    try:
        doc.build(story)
    except Exception as e:
        print("PDF build exception:", e, traceback.format_exc())
    buffer.seek(0)
    data = buffer.getvalue()
    buffer.close()
    return data

# UI: Streamlit
st.set_page_config(page_title="NeuroEarly Pro — Clinical", layout="wide", initial_sidebar_state="expanded")

logo_path = find_logo_path()
logo_bytes = load_logo_bytes(logo_path) if logo_path else None

# inject CSS for Arabic font/RTL if available
if AMIRI_TTF.exists():
    st.markdown(f"<style>@font-face {{font-family: 'AmiriLocal'; src: url('/assets/{AMIRI_TTF.name}');}} body {{ font-family: 'AmiriLocal', Arial, sans-serif; }}</style>", unsafe_allow_html=True)
else:
    st.markdown("<style>@import url('https://fonts.googleapis.com/css2?family=Amiri&display=swap'); body { font-family: 'Amiri', Arial, sans-serif; }</style>", unsafe_allow_html=True)

# header
col1, col2 = st.columns([8,1])
with col1:
    st.markdown(f"<h1 style='color:{PRIMARY_BLUE}; margin:0;'>🧠 NeuroEarly Pro</h1>", unsafe_allow_html=True)
    st.markdown(f"<div style='color:#666; margin-bottom:6px;'>Clinical QEEG Assistant — Golden Bird LLC</div>", unsafe_allow_html=True)
with col2:
    if logo_bytes:
        st.image(logo_bytes, width=80)
    else:
        st.empty()

st.markdown("---")

# Sidebar left
with st.sidebar:
    st.header(t("Settings","الإعدادات"))
    lang_choice = st.selectbox(t("Language","اللغة"), ["English","Arabic"], index=0 if st.session_state["lang"]=="en" else 1)
    st.session_state["lang"] = "en" if lang_choice=="English" else "ar"
    st.text_input(t("Patient Name (optional)","اسم المريض (اختياري)"), key="patient_name", placeholder=t("Name (not printed)","الاسم (لن يطبع)"))
    st.text_input(t("Patient ID","معرف المريض"), key="patient_id", placeholder="ID")
    dob_default = date(1980,1,1)
    st.date_input(t("Date of Birth","تاريخ الميلاد"), value=dob_default, key="dob", min_value=date(1900,1,1), max_value=date(2025,12,31))
    st.selectbox(t("Sex","الجنس"), [t("Unknown","غير معروف"), t("Male","ذكر"), t("Female","أنثى"), t("Other","أخرى")], key="sex")
    st.markdown("---")
    st.subheader(t("Medications","الأدوية"))
    st.text_area(t("Current meds (one per line)","الأدوية الحالية (سطر لكل دواء)"), key="meds", height=100)
    st.subheader(t("Blood tests","تحاليل الدم"))
    st.text_area(t("Labs summary (one per line)","ملخص التحاليل (سطر لكل اختبار)"), key="labs", height=100)
    st.markdown("---")
    st.markdown(f"<small>Prepared by Golden Bird LLC — NeuroEarly Pro</small>", unsafe_allow_html=True)

# Main: uploader + questionnaires
st.subheader(t("Upload EDF file(s)","رفع ملفات EDF"))
uploaded = st.file_uploader(t("Drag & drop .edf files (multiple allowed)","اسحب وأسقط ملفات .edf (مؤثرات متعددة)"), type=["edf"], accept_multiple_files=True)
if uploaded:
    st.session_state["uploaded_files"] = uploaded

st.markdown("### " + t("Questionnaires","الاستبيانات"))
# PHQ-9 updated with special Q3,Q5,Q8 options
phq_texts_en = [
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
phq_texts_ar = [
    "قليل الاهتمام أو المتعة في القيام بالأشياء",
    "الشعور بالحزن أو الاكتئاب أو اليأس",
    "تغيرات النوم (اختر): أرق / فرط النوم / طبيعي",
    "الشعور بالتعب أو قلة الطاقة",
    "تغيرات الشهية (اختر): فرط الأكل / نقصان الأكل / طبيعي",
    "الشعور بالسوء عن نفسك أو أنك فشل",
    "صعوبة في التركيز",
    "الحركة أو الكلام ببطء أو القلق/النشاط المفرط (اختر)",
    "أفكار بأنك ستكون أفضل ميتاً"
]
for i,en_text,ar_text in zip(range(1,10), phq_texts_en, phq_texts_ar):
    key = f"phq_{i}"
    qtext = t(f"Q{i}: {en_text}", f"س{i}: {ar_text}")
    if i==3:
        opts = [t("Insomnia","أرق"), t("Hypersomnia","فرط النوم"), t("Normal","طبيعي")]
        st.selectbox(qtext, opts, index=opts.index(st.session_state.get(key)) if st.session_state.get(key) in opts else 2, key=key)
    elif i==5:
        opts = [t("Overeating","فرط الأكل"), t("Under-eating","نقص الأكل"), t("Normal","طبیعي")]
        st.selectbox(qtext, opts, index=opts.index(st.session_state.get(key)) if st.session_state.get(key) in opts else 2, key=key)
    elif i==8:
        opts = [t("Slow speech/movement","بطء الكلام/الحركة"), t("Restlessness","القلق/النشاط المفرط")]
        st.selectbox(qtext, opts, index=opts.index(st.session_state.get(key)) if st.session_state.get(key) in opts else 0, key=key)
    else:
        st.radio(qtext, [0,1,2,3], index=int(st.session_state.get(key) or 0), key=key)

# AD8
st.markdown("### " + t("AD8 (Cognitive screening)","AD8 (فحص الإدراك)"))
ad8_texts_en = [
    "Problems with judgment (e.g. bad decision making)",
    "Less interest in hobbies/activities",
    "Repeats questions, stories, or statements",
    "Trouble learning to use tools/ appliances",
    "Forgets correct month or year",
    "Difficulty handling complicated financial affairs",
    "Difficulty remembering appointments",
    "Daily problems with thinking and memory"
]
ad8_texts_ar = [
    "مشاكل في الحكم (مثلاً اتخاذ قرارات سيئة)",
    "انخفاض الاهتمام بالهوايات/الأنشطة",
    "يكرر أسئلة أو قصص أو عبارات",
    "صعوبة تعلم استخدام أدوات/أجهزة",
    "ينسى الشهر أو السنة الصحيحة",
    "صعوبة في التعامل مع شؤون مالية معقدة",
    "صعوبة تذكر المواعيد",
    "مشاكل يومية في التفكير والذاكرة"
]
for i,(en_text,ar_text) in enumerate(zip(ad8_texts_en, ad8_texts_ar), start=1):
    st.radio(t(f"A{i}: {en_text}", f"أ{i}: {ar_text}"), [0,1], index=int(st.session_state.get(f"ad8_{i}",0)), key=f"ad8_{i}")

st.markdown("---")
c1,c2 = st.columns([1,1])
with c1:
    if st.button(t("Process EDF(s) and Analyze","معالجة ملفات EDF وتحليلها")):
        st.session_state["results"] = []
        uploads = st.session_state.get("uploaded_files") or []
        if not uploads:
            st.error(t("Please upload at least one EDF file.","الرجاء تحميل ملف EDF واحد على الأقل."))
        else:
            prog = st.progress(0)
            total = len(uploads)
            for idx,up in enumerate(uploads):
                try:
                    raw, err = read_edf_bytes(up)
                    if err:
                        st.error(f"{up.name}: {err}")
                        continue
                    dfbands = compute_band_powers(raw)
                    res = {"filename": up.name, "raw_obj": raw, "dfbands": dfbands.to_dict(orient="list") if (dfbands is not None and not dfbands.empty) else {}, "connectivity_narr": ""}
                    res = enrich_result(res)
                    st.session_state["results"].append(res)
                    prog.progress(int(((idx+1)/total)*100))
                except Exception as e:
                    st.error(f"{up.name}: processing error: {e}")
                    print(traceback.format_exc())
            st.success(t("Processing complete.","اكتملت المعالجة."))
with c2:
    st.write(t("After processing, view results in tabs and export PDF.","بعد المعالجة، شاهد النتائج وصدّر PDF."))

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
    # compute PHQ numeric mapping
    phq_total_numeric = 0
    for i in range(1,10):
        key=f"phq_{i}"
        v = st.session_state.get(key)
        if i in (3,5):
            if isinstance(v,str):
                phq_total_numeric += 2 if v!=t("Normal","طبیعي") else 0
        elif i==8:
            if isinstance(v,str):
                phq_total_numeric += 2 if v==t("Slow speech/movement","بطء الكلام/الحركة") else 1
        else:
            try:
                phq_total_numeric += int(v)
            except Exception:
                phq_total_numeric += 0
    ad8_total = sum(int(st.session_state.get(f"ad8_{i}",0)) for i in range(1,9))

    tabs = st.tabs([t("Overview","نظرة عامة"), t("Connectivity","الاتصال"), t("XAI (SHAP)","الذكاء القابل للتفسير (SHAP)"), t("Microstates","الحالات الدقيقة"), t("Export","تصدير")])
    with tabs[0]:
        st.subheader(t("Overview","نظرة عامة"))
        ml_val = (r0.get("ml_scores",{}).get("alzheimers") or r0.get("ml_scores",{}).get("depression") or 0.0)
        st.metric(t("Final ML Risk Score","معدل الخطر ML النهائي"), f"{ml_val*100:.1f}%")
        qdf = pd.DataFrame([{
            "Theta/Alpha": r0.get("agg",{}).get("theta_alpha_ratio"),
            "Theta/Beta": r0.get("agg",{}).get("theta_beta_ratio"),
            "Alpha Asym F3-F4": r0.get("agg",{}).get("alpha_asym_f3_f4"),
            "Mean Connectivity": r0.get("agg",{}).get("mean_connectivity"),
            "HPI": r0.get("agg",{}).get("hpi")
        }])
        st.table(qdf.T.rename(columns={0:"Value"}))
        colA,colB = st.columns([1,1])
        with colA:
            if r0.get("bar_img"):
                st.image(r0["bar_img"], caption=t("Normative comparison","مقارنة معيارية"), width=520)
        with colB:
            if r0.get("topo_imgs") and r0["topo_imgs"].get("Alpha"):
                st.image(r0["topo_imgs"].get("Alpha"), caption="Alpha topography", width=320)
        st.markdown("**" + t("QEEG Interpretation (brief):","تفسير QEEG (مختصر):") + "**")
        narr=[]
        narr.append(t("Theta/Alpha ratio:","نسبة Theta/Alpha:") + f" {fmtf(r0.get('agg',{}).get('theta_alpha_ratio',0.0),3)}")
        narr.append(t("Mean Connectivity (alpha):","متوسط الاتصال (ألفا):") + f" {fmtf(r0.get('agg',{}).get('mean_connectivity',0.0),4)}")
        hpi_show = r0.get("agg",{}).get("hpi")
        if hpi_show is not None:
            narr.append(f"HPI: {fmtf(hpi_show,4)} {'(Elevated)' if hpi_show>1.8 else ''}")
        if r0.get("focal",{}).get("alerts"):
            narr.append(t("Focal Delta Alerts detected:","تم الكشف عن تنبيهات دلتا موضعية:") + " " + ", ".join([str(a) for a in r0.get("focal",{}).get("alerts")]))
        st.write("\n".join(narr))

    with tabs[1]:
        st.subheader(t("Connectivity","الاتصال"))
        st.write(r0.get("conn_narr",""))
        if r0.get("conn_img"):
            st.image(r0["conn_img"], caption=t("Connectivity matrix","مصفوفة الاتصال"), width=680)
        else:
            st.info(t("Connectivity image not available","صورة الاتصال غير متاحة"))

    with tabs[2]:
        st.subheader(t("Explainable AI (SHAP)","الذكاء القابل للتفسير (SHAP)"))
        if r0.get("shap_img"):
            st.image(r0["shap_img"], caption="SHAP top contributors", width=680)
        elif r0.get("shap_table"):
            st.table(pd.DataFrame(list(r0.get("shap_table",{}).items()), columns=["Feature","Importance"]).set_index("Feature"))
        else:
            st.info(t("SHAP not available. Place shap_summary.json in repo or upload it.","SHAP غير متاح. ضع shap_summary.json في المستودع أو ارفعه."))

    with tabs[3]:
        st.subheader(t("Microstates","الحالات الدقيقة"))
        if r0.get("microstate"):
            st.write(r0["microstate"])
        else:
            st.info(t("Microstate analysis not available","تحليل microstate غير متاح"))

    with tabs[4]:
        st.subheader(t("Export / PDF","تصدير / PDF"))
        pdf_lang = st.selectbox(t("Select PDF language","اختر لغة PDF"), options=[t("English","الإنجليزية"), t("Arabic","العربية")], index=0)
        lang_code = "en" if pdf_lang==t("English","الإنجليزية") else "ar"
        amiri = AMIRI_TTF if AMIRI_TTF.exists() else None
        if st.button(t("Generate & Download Clinical PDF","إنشاء وتنزيل تقرير PDF")):
            try:
                pdf_bytes = generate_pdf_report(r0, patient_info, phq_total_numeric, ad8_total, lang=lang_code, amiri=amiri, logo_bytes=logo_bytes)
                if pdf_bytes:
                    st.success(t("PDF generated successfully.","تم إنشاء PDF بنجاح."))
                    st.download_button("⬇️ " + t("Download Clinical Report (PDF)","تنزيل تقرير سريري (PDF)"), data=pdf_bytes, file_name=f"NeuroEarly_Report_{now_ts()}.pdf", mime="application/pdf")
                else:
                    st.error(t("PDF generation returned no data.","إرجاع إنشاء PDF بلا بيانات."))
            except Exception as e:
                st.error(t("PDF generation failed:","فشل إنشاء PDF:") + f" {e}")
                st.text(traceback.format_exc())
else:
    st.info(t("No processed results yet. Upload EDF and press 'Process EDF(s) and Analyze'.","لا توجد نتائج بعد. ارفع EDF واضغط 'معالجة'."))

# End of file

# app.py — NeuroEarly Pro — Clinical v4 (Final)
# Single-file Streamlit app (copy/paste into repo root)
# Requirements: use provided requirements.txt (we sent قبلاً)
# Assets:
#   - assets/GoldenBird_logo.svg.svg  (or change LOGO_PATH below)
#   - assets/Amiri-Regular.ttf       (for Arabic PDF)
#   - optional: shap_summary.json

import os, io, sys, json, math, tempfile, traceback
from pathlib import Path
from datetime import datetime, date
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# Optional heavy libs
HAS_MNE=False; HAS_PYEDF=False; HAS_REPORTLAB=False; HAS_SHAP=False; HAS_ARABIC=False
try:
    import mne; HAS_MNE=True
except Exception: HAS_MNE=False
try:
    import pyedflib; HAS_PYEDF=True
except Exception: HAS_PYEDF=False
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    HAS_REPORTLAB=True
except Exception:
    HAS_REPORTLAB=False
try:
    import shap; HAS_SHAP=True
except Exception: HAS_SHAP=False
try:
    import arabic_reshaper; from bidi.algorithm import get_display; HAS_ARABIC=True
except Exception: HAS_ARABIC=False

# Constants / paths
ROOT = Path(__file__).parent
ASSETS = ROOT / "assets"
LOGO_PATH = ASSETS / "GoldenBird_logo.svg.svg"   # <-- update if your filename/directory differs
AMIRI_PATH = ASSETS / "Amiri-Regular.ttf"
SHAP_JSON = ROOT / "shap_summary.json"

BLUE = "#3FA9F5"   # your chosen bright blue
LIGHT_BG = "#f7fbff"

BANDS = {
    "Delta": (0.5,4.0),
    "Theta": (4.0,8.0),
    "Alpha": (8.0,13.0),
    "Beta": (13.0,30.0),
    "Gamma": (30.0,45.0)
}

def now_ts(): return datetime.utcnow().strftime("%Y%m%d_%H%M%S")

def safe_trace(e):
    tb = traceback.format_exc()
    print(tb, file=sys.stderr)
    return tb

def reshape_ar(text: str)->str:
    if not HAS_ARABIC or not text: return text
    try:
        return get_display(arabic_reshaper.reshape(text))
    except Exception:
        return text

# EDF reader: prefer mne, fallback to pyedflib
def read_edf(path: str):
    try:
        if HAS_MNE:
            raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
            data = raw.get_data()   # shape (n_channels, n_samples)
            sf = float(raw.info.get("sfreq", 256.0))
            chs = raw.ch_names
            return data, sf, chs
        elif HAS_PYEDF:
            f = pyedflib.EdfReader(path)
            n = f.signals_in_file
            chs = f.getSignalLabels()
            sf = float(f.getSampleFrequency(0))
            arr = np.vstack([f.readSignal(i) for i in range(n)])
            f._close()
            return arr, sf, chs
        else:
            raise RuntimeError("No EDF backend available (install mne or pyedflib).")
    except Exception as e:
        raise

# Preprocessing: notch & bandpass (safe)
def preprocess(sig, sf, do_notch=True):
    try:
        from scipy.signal import iirnotch, filtfilt, butter
    except Exception:
        return sig
    out = sig.copy()
    for i in range(out.shape[0]):
        s = out[i,:].astype(float)
        if do_notch:
            for f0 in (50.0,60.0):
                try:
                    b,a = iirnotch(f0, 30.0, sf)
                    s = filtfilt(b,a,s)
                except Exception:
                    pass
        try:
            b,a = butter(4, [0.5/(sf/2), 45.0/(sf/2)], btype='band')
            s = filtfilt(b,a,s)
        except Exception:
            pass
        out[i,:] = s
    return out

# PSD & bandpower (welch)
def compute_bandpowers(data, sf):
    from scipy.signal import welch
    nchan = data.shape[0]
    df_list = []
    for i in range(nchan):
        try:
            freqs, pxx = welch(data[i,:], fs=sf, nperseg=min(2048, data.shape[1]))
        except Exception:
            freqs = np.array([0.]); pxx = np.array([0.])
        total = np.trapz(pxx, freqs) if freqs.size else 1.0
        row = {}
        for b,(lo,hi) in BANDS.items():
            mask = (freqs>=lo)&(freqs<=hi)
            abs_p = float(np.trapz(pxx[mask], freqs[mask])) if mask.sum() else 0.0
            rel = abs_p/(total if total>0 else 1.0)
            row[f"{b}_abs"]=abs_p; row[f"{b}_rel"]=rel
        df_list.append(row)
    df = pd.DataFrame(df_list)
    return df

# aggregate features
def aggregate(dfbands, ch_names=None):
    out = {}
    if dfbands.empty:
        for b in BANDS: out[f"{b.lower()}_rel_mean"]=0.0
        out["theta_alpha_ratio"]=0.0; out["alpha_asym_F3_F4"]=0.0
        return out
    for b in BANDS:
        col = f"{b}_rel"
        out[f"{b.lower()}_rel_mean"] = float(dfbands[col].mean()) if col in dfbands else 0.0
    alpha = out.get("alpha_rel_mean",1e-9); theta=out.get("theta_rel_mean",0.0); beta=out.get("beta_rel_mean",1e-9)
    out["theta_alpha_ratio"]=float(theta/alpha) if alpha>0 else 0.0
    out["theta_beta_ratio"]=float(theta/beta) if beta>0 else 0.0
    out["alpha_asym_F3_F4"]=0.0
    if ch_names:
        try:
            names = [n.upper() for n in ch_names]
            if "F3" in names and "F4" in names:
                i3=names.index("F3"); i4=names.index("F4")
                a3 = dfbands.iloc[i3]["Alpha_rel"]; a4=dfbands.iloc[i4]["Alpha_rel"]
                out["alpha_asym_F3_F4"]=float(a3-a4)
        except Exception:
            pass
    return out

# Focal Delta Index & asymmetry
def focal_delta(dfbands, ch_names=None):
    out={"FDI_map":{}, "alerts":[], "max_channel":None, "max_val":None, "pairs":{}}
    try:
        if "Delta_abs" not in dfbands.columns:
            return out
        delta = dfbands["Delta_abs"].values
        gm = float(np.nanmean(delta)) if delta.size else 1e-9
        fdi = delta/(gm if gm>0 else 1e-9)
        for i,val in enumerate(fdi):
            out["FDI_map"][i]=float(val)
            if val>2.0:
                ch = ch_names[i] if ch_names and i<len(ch_names) else f"Ch{i}"
                out["alerts"].append({"type":"FDI","channel":ch,"value":float(val)})
        # pairs asymmetry
        pairs=[("T7","T8"),("F3","F4"),("P3","P4"),("O1","O2"),("C3","C4")]
        nm={}
        if ch_names:
            names=[n.upper() for n in ch_names]
            for L,R in pairs:
                if L in names and R in names:
                    li=names.index(L); ri=names.index(R)
                    ratio = float(delta[ri]/(delta[li]+1e-9)) if delta[li]>0 else float('inf')
                    out["pairs"][f"{L}/{R}"]=ratio
                    if ratio>3.0 or ratio<0.33 or ratio==float('inf'):
                        out["alerts"].append({"type":"asym","pair":f"{L}/{R}","ratio":ratio})
        mx_idx = int(np.argmax(fdi)) if fdi.size else None
        out["max_channel"] = ch_names[mx_idx] if ch_names and mx_idx is not None and mx_idx < len(ch_names) else None
        out["max_val"] = float(fdi[mx_idx]) if mx_idx is not None else None
    except Exception as e:
        print("focal err", e)
    return out

# connectivity (coherence via scipy fallback; use mne if available)
def connectivity(data, sf, ch_names=None, band=(8.0,13.0)):
    try:
        n = data.shape[0]
        mat = np.zeros((n,n))
        narrative = ""
        mean_conn = 0.0
        if HAS_MNE:
            try:
                info = mne.create_info(ch_names, sf, ch_types='eeg')
                raw = mne.io.RawArray(data, info)
                from mne.connectivity import spectral_connectivity
                con, freqs, times, n_epochs, n_tapers = spectral_connectivity([raw], method='coh', sfreq=sf, fmin=band[0], fmax=band[1], faverage=True, verbose=False)
                conn = np.squeeze(con)
                if conn.shape==(n,n):
                    mat = conn
                narrative = f"Coherence {band[0]}-{band[1]} Hz (MNE)"
            except Exception as e:
                print("mne conn err", e)
                narrative="Connectivity unavailable (mne error)"
        else:
            # scipy fallback
            try:
                from scipy.signal import coherence
                for i in range(n):
                    for j in range(i,n):
                        try:
                            f,Cxy = coherence(data[i], data[j], fs=sf, nperseg=min(1024, max(256, data.shape[1])))
                            mask = (f>=band[0])&(f<=band[1])
                            val = float(np.nanmean(Cxy[mask])) if mask.sum() else 0.0
                        except Exception:
                            val = 0.0
                        mat[i,j]=mat[j,i]=val
                narrative = f"Coherence {band[0]}-{band[1]} Hz (scipy fallback)"
            except Exception as e:
                print("scipy conn err", e)
                narrative="Connectivity not computed"
        mean_conn = float(np.nanmean(mat)) if mat.size else 0.0
        # image
        conn_img = None
        try:
            fig,ax = plt.subplots(figsize=(4,3))
            im = ax.imshow(mat, cmap='viridis', aspect='auto')
            ax.set_title("Connectivity (alpha)")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format='png'); plt.close(fig); buf.seek(0)
            conn_img = buf.getvalue()
        except Exception:
            conn_img = None
        return mat, narrative, conn_img, mean_conn
    except Exception as e:
        print("connectivity overall err", e)
        return None, "(error)", None, 0.0

# topomap approximate
def topomap_image(vals, ch_names, band_name):
    try:
        pos = {'Fp1':(-0.5,1),'Fp2':(0.5,1),'F3':(-0.8,0.3),'F4':(0.8,0.3),'C3':(-0.8,-0.3),'C4':(0.8,-0.3),'P3':(-0.5,-0.8),'P4':(0.5,-0.8),'O1':(-0.2,-1),'O2':(0.2,-1),'F7':(-1,0.6),'F8':(1,0.6),'T7':(-1,-0.3),'T8':(1,-0.3)}
        xs,ys,vals_plot=[],[],[]
        for i,ch in enumerate(ch_names):
            u=ch.upper()
            if u in pos:
                x,y=pos[u]; xs.append(x); ys.append(y); vals_plot.append(float(vals[i]))
        if len(xs)>=3:
            fig,ax=plt.subplots(figsize=(3.2,2.2))
            sc=ax.scatter(xs,ys,c=vals_plot,s=260,cmap='RdBu_r')
            ax.set_xticks([]); ax.set_yticks([]); ax.set_title(f"{band_name}")
            fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
            buf=io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format='png'); plt.close(fig); buf.seek(0)
            return buf.getvalue()
        else:
            # fallback bar
            fig,ax=plt.subplots(figsize=(3.2,2.2))
            n=min(len(ch_names), len(vals))
            ax.bar(range(n), vals[:n]); ax.set_xticks(range(n)); ax.set_xticklabels(ch_names[:n], rotation=60, fontsize=7)
            ax.set_title(f"{band_name} (bar)")
            buf=io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format='png'); plt.close(fig); buf.seek(0)
            return buf.getvalue()
    except Exception as e:
        print("topomap err", e)
        return None

# SHAP utilities
def load_shap(path):
    if Path(path).exists():
        try:
            with open(path,"r",encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None

def shap_plot_bytes(feats:dict, top_n=10):
    if not feats or not plt: return None
    s = pd.Series(feats).abs().sort_values(ascending=False).head(top_n)
    fig,ax=plt.subplots(figsize=(5.6,1.8))
    s.sort_values().plot.barh(ax=ax, color=BLUE)
    ax.set_xlabel("SHAP impact (abs)")
    buf=io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format='png'); plt.close(fig); buf.seek(0)
    return buf.getvalue()

# PDF generator
def generate_pdf_report(summary:Dict[str,Any], lang="en", amiri_path:Optional[str]=None):
    if not HAS_REPORTLAB:
        raise RuntimeError("reportlab not installed in environment.")
    buffer=io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=36, bottomMargin=36, leftMargin=36, rightMargin=36)
    styles = getSampleStyleSheet()
    base_font="Helvetica"
    if amiri_path and Path(amiri_path).exists():
        try:
            pdfmetrics.registerFont(TTFont("Amiri", str(amiri_path))); base_font="Amiri"
        except Exception as e:
            print("Amiri reg fail", e)
    styles.add(ParagraphStyle(name="TitleBlue", fontName=base_font, fontSize=16, textColor=colors.HexColor(BLUE), alignment=1, spaceAfter=8))
    styles.add(ParagraphStyle(name="H2", fontName=base_font, fontSize=12, textColor=colors.HexColor(BLUE), spaceAfter=6))
    styles.add(ParagraphStyle(name="Body", fontName=base_font, fontSize=10, leading=14))
    story=[]
    # header with logo
    left = Paragraph("NeuroEarly Pro — Clinical QEEG Report", styles["TitleBlue"])
    if Path(LOGO_PATH).exists():
        try:
            img = RLImage(str(LOGO_PATH), width=1.2*inch, height=1.2*inch)
            header_tbl = Table([[left,img]], colWidths=[4.7*inch,1.4*inch])
            header_tbl.setStyle(TableStyle([("VALIGN",(0,0),(-1,-1),"MIDDLE")]))
            story.append(header_tbl)
        except Exception:
            story.append(left)
    else:
        story.append(left)
    story.append(Spacer(1,6))
    # Executive summary
    story.append(Paragraph("<b>Executive Summary</b>", styles["H2"]))
    story.append(Paragraph(f"Final ML Risk Score: {summary.get('final_ml_risk_display','N/A')}", styles["Body"]))
    story.append(Spacer(1,6))
    # Patient info
    pi = summary.get("patient_info",{})
    ptab=[["Field","Value"],["ID", pi.get("id","—")],["DOB",pi.get("dob","—")],["Sex",pi.get("sex","—")]]
    t=Table(ptab, colWidths=[1.5*inch,4.5*inch]); t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#eaf6ff")),("GRID",(0,0),(-1,-1),0.25,colors.lightgrey)]))
    story.append(t); story.append(Spacer(1,8))
    # QEEG metrics table
    story.append(Paragraph("<b>QEEG Key Metrics</b>", styles["H2"]))
    metrics = summary.get("metrics",{})
    rows=[["Metric","Value"]]
    for k,v in metrics.items():
        try: rows.append([str(k), f"{float(v):.4f}"])
        except: rows.append([str(k), str(v)])
    t2 = Table(rows, colWidths=[3.5*inch, 2.5*inch])
t2.setStyle(TableStyle([
    ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef7ff"))
]))
story.append(t2)
story.append(Spacer(1, 8))

    # bar img
try:
    if summary.get("bar_img"):
        story.append(Paragraph("Normative Comparison", styles["H2"]))
        story.append(Spacer(1, 0.15*inch))
        img = Image(io.BytesIO(summary["bar_img"]), width=5.5*inch, height=3.0*inch)
        story.append(img)
        story.append(Spacer(1, 0.3*inch))
except Exception:
    pass

    # topomaps
    if summary.get("topo_images"):
        story.append(Paragraph("<b>Topography Maps</b>", styles["H2"]))
        imgs = [RLImage(io.BytesIO(b), width=2.6*inch, height=1.6*inch) for _,b in summary["topo_images"].items() if b]
        if imgs:
            # 2 per row
            rows=[]; row=[]
            for im in imgs:
                row.append(im)
                if len(row)==2:
                    rows.append(row); row=[]
            if row: rows.append(row)
            for r in rows:
                story.append(Table([r], colWidths=[3*inch]*len(r))); story.append(Spacer(1,4))
    # connectivity
    if summary.get("conn_image"):
        story.append(Paragraph("<b>Connectivity (Alpha)</b>", styles["H2"]))
        try:
            story.append(RLImage(io.BytesIO(summary["conn_image"]), width=5.6*inch, height=2.4*inch)); story.append(Spacer(1,6))
        except Exception:
            pass
    # shap
    if summary.get("shap_img"):
        story.append(Paragraph("<b>Explainable AI — SHAP top contributors</b>", styles["H2"]))
        try:
            story.append(RLImage(io.BytesIO(summary["shap_img"]), width=5.6*inch, height=1.8*inch)); story.append(Spacer(1,6))
        except Exception:
            pass
    # tumor
    if summary.get("tumor"):
        story.append(Paragraph("<b>Focal Delta / Tumor indicators</b>", styles["H2"]))
        story.append(Paragraph(summary["tumor"].get("narrative",""), styles["Body"]))
        if summary["tumor"].get("alerts"):
            for a in summary["tumor"]["alerts"]:
                story.append(Paragraph(f"- {a}", styles["Body"]))
        story.append(Spacer(1,6))
    # recommendations
    story.append(Paragraph("<b>Structured Clinical Recommendations</b>", styles["H2"]))
    for r in summary.get("recommendations", []):
        story.append(Paragraph(r, styles["Body"]))
    story.append(Spacer(1,12))
    story.append(Paragraph("Prepared by Golden Bird LLC — NeuroEarly Pro", styles["Note"]))
    story.append(Spacer(1,18))
    story.append(Paragraph("Doctor signature: ___________________________", styles["Body"]))
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

    
# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="NeuroEarly Pro — Clinical", layout="wide")
st.markdown(f"""
<div style="display:flex;align-items:center;justify-content:space-between;padding:10px;border-radius:8px;background:{LIGHT_BG};">
 <div style="font-weight:700;color:{BLUE};font-size:18px;">🧠 NeuroEarly Pro — Clinical AI</div>
 <div style="display:flex;align-items:center;">
   <div style="font-size:12px;color:#333;margin-right:10px;">Prepared by Golden Bird LLC</div>
   {'<img src="assets/goldenbird_logo.png" style="height:40px;">' if Path(LOGO_PATH).exists() else ''}
 </div>
</div>
""", unsafe_allow_html=True)

col_main, col_side = st.columns([3,1])
with col_side:
    st.header("Settings")
    lang_choice = st.selectbox("Language", options=["English","Arabic"], index=0)
    is_ar = (lang_choice=="Arabic")
    st.markdown("---")
    st.subheader("Patient")
    pid = st.text_input("Patient ID")
    dob = st.date_input("DOB", value=date(1980,1,1), min_value=date(1900,1,1), max_value=date.today())
    sex = st.selectbox("Sex", ["Unknown","Male","Female","Other"])
    st.markdown("---")
    st.subheader("Clinical")
    labs = st.multiselect("Relevant labs", options=["B12","TSH","Vitamin D","Folate","HbA1c"])
    meds = st.text_area("Current medications")
    st.markdown("---")
    st.write(f"Backends: mne={HAS_MNE} pyedflib={HAS_PYEDF} reportlab={HAS_REPORTLAB} shap={HAS_SHAP}")
with col_main:
    st.header("1) Upload EDF files")
    uploads = st.file_uploader("Upload EDF (.edf) — multiple allowed", type=["edf"], accept_multiple_files=True)
    st.header("2) Questionnaires")
    # PHQ-9 (custom)
    PHQ_QS = [
        "Little interest or pleasure in doing things",
        "Feeling down, depressed, or hopeless",
        "Sleep: Insomnia / Short sleep / Hypersomnia",
        "Feeling tired or having little energy",
        "Appetite: Overeating / Undereating",
        "Feeling bad about yourself",
        "Trouble concentrating",
        "Moving/speaking slowly OR fidgety/restless",
        "Thoughts that you'd be better off dead"
    ]
    phq = {}
    for i,q in enumerate(PHQ_QS, start=1):
        if i==3:
            sel = st.selectbox(f"Q{i}. {q}", ["0 — Not at all","1 — Insomnia","2 — Short sleep","3 — Hypersomnia"], key=f"phq{i}")
            phq[f"Q{i}"]=int(sel.split("—")[0].strip())
        elif i==5:
            sel = st.selectbox(f"Q{i}. {q}", ["0 — Not at all","1 — Less eating","2 — More eating","3 — Variable"], key=f"phq{i}")
            phq[f"Q{i}"]=int(sel.split("—")[0].strip())
        elif i==8:
            sel = st.selectbox(f"Q{i}. {q}", ["0 — Not at all","1 — Slow speech/move","2 — Fidgety/restless","3 — Both"], key=f"phq{i}")
            phq[f"Q{i}"]=int(sel.split("—")[0].strip())
        else:
            sel = st.radio(f"Q{i}. {q}", ["0 — Not at all","1 — Several days","2 — More than half the days","3 — Nearly every day"], key=f"phq{i}", horizontal=True)
            phq[f"Q{i}"]=int(sel.split("—")[0].strip())
    phq_total = sum(phq.values())
    st.info(f"PHQ-9 total: {phq_total} /27")
    # AD8
    st.header("AD8 (Cognitive)")
    ad8 = {}
    for i in range(1,9):
        v = st.radio(f"A{i}.", options=[0,1], key=f"ad8_{i}", horizontal=True)
        ad8[f"A{i}"]=int(v)
    ad8_total=sum(ad8.values()); st.info(f"AD8 total: {ad8_total} /8")

    # Options
    st.markdown("---"); st.subheader("Processing options")
    use_notch = st.checkbox("Apply notch filter (50/60Hz)", value=True)
    do_conn = st.checkbox("Compute connectivity", value=True)
    gen_topos = st.checkbox("Generate topography maps", value=True)
    use_shap_ui = st.checkbox("Enable XAI (SHAP)", value=True)

    if st.button("Process files"):
        if not uploads:
            st.error("Upload at least one EDF")
        else:
            proc = st.empty(); proc.info("Processing...")
            results=[]
            for up in uploads:
                proc.info(f"Processing {up.name} ...")
                try:
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".edf")
                    tmp.write(up.getbuffer()); tmp.flush(); tmp.close()
                    data, sf, chs = read_edf(tmp.name)  # data: (nchan, nsamples)
                    # preprocess
                    cleaned = preprocess(data, sf, do_notch=use_notch)
                    dfbands = compute_bandpowers(cleaned, sf)
                    # attach channel names
                    try:
                        if len(chs)==dfbands.shape[0]:
                            dfbands.index = chs
                        else:
                            dfbands.index = [f"Ch{i+1}" for i in range(dfbands.shape[0])]
                    except Exception:
                        dfbands.index = [f"Ch{i+1}" for i in range(dfbands.shape[0])]
                    agg = aggregate(dfbands, ch_names=list(dfbands.index))
                    foc = focal_delta(dfbands, ch_names=list(dfbands.index))
                    topo_imgs={}
                    if gen_topos:
                        for b in ["Delta","Theta","Alpha","Beta","Gamma"]:
                            try:
                                vals = dfbands[f"{b}_rel"].values if f"{b}_rel" in dfbands.columns else np.zeros(dfbands.shape[0])
                                topo_imgs[b] = topomap_image(vals, list(dfbands.index), b)
                            except Exception:
                                topo_imgs[b]=None
                    conn_mat, conn_narr, conn_img, mean_conn = (None,"(not computed)",None, None)
                    if do_conn:
                        conn_mat, conn_narr, conn_img, mean_conn = connectivity(cleaned, sf, ch_names=list(dfbands.index))
                    # ML risk improved formula: normalized combination of theta/alpha, PHQ, AD8, mean_connectivity (lower connectivity -> higher risk)
                    ta = float(agg.get("theta_alpha_ratio",0.0))
                    phq_norm = phq_total/27.0
                    ad8_norm = ad8_total/8.0
                    mc = float(mean_conn) if mean_conn is not None else 0.0
                    mc_norm = 1.0 - mc  # lower connectivity increases risk
                    # weights tuned for clinical emphasis
                    risk_score = (min(2.0,ta)/2.0)*0.5 + phq_norm*0.25 + ad8_norm*0.15 + mc_norm*0.10
                    risk_score = max(0.0, min(1.0, risk_score))
                    results.append({
                        "file": up.name, "sf":sf, "chs":list(dfbands.index),
                        "dfbands":dfbands, "agg":agg, "focal":foc,
                        "topo_imgs":topo_imgs, "conn_img":conn_img, "conn_narr":conn_narr,
                        "mean_conn":mean_conn, "risk":risk_score
                    })
                    proc.success(f"Processed {up.name}")
                except Exception as e:
                    proc.error(f"Failed {up.name}: {e}")
                    st.error(safe_trace(e))
            if results:
                st.session_state["NE_RESULTS"]=results
                r0=results[0]
                st.metric("Final ML Risk", f"{r0['risk']*100:.1f}%")
                st.write("QEEG metrics (first file):")
                st.table(pd.DataFrame({
                    "Theta/Alpha": [r0["agg"].get("theta_alpha_ratio",0)],
                    "Theta/Beta": [r0["agg"].get("theta_beta_ratio",0)],
                    "Alpha Asym (F3-F4)": [r0["agg"].get("alpha_asym_F3_F4",0)],
                    "Mean connectivity": [r0.get("mean_conn", r0.get("mean_conn", None))]
                }).T.rename(columns={0:"Value"}))
                st.write("Topographies:")
                cols = st.columns(5)
                for i,b in enumerate(["Delta","Theta","Alpha","Beta","Gamma"]):
                    img = r0["topo_imgs"].get(b)
                    if img: cols[i].image(img, caption=b, use_column_width=True)
                if r0.get("conn_img"): st.image(r0["conn_img"], caption="Connectivity", use_column_width=True)
                st.json(r0["focal"])

    st.markdown("---")
    st.markdown("## Generate PDF Report")
    pdf_lang = st.selectbox("PDF language", options=["English","Arabic"], index=0)
    if st.button("Generate & Download PDF"):
        try:
            if "NE_RESULTS" not in st.session_state or not st.session_state["NE_RESULTS"]:
                st.error("Process EDF(s) before generating report.")
            else:
                r0 = st.session_state["NE_RESULTS"][0]
                summary={}
                summary["patient_info"]={"id": pid or "—", "dob": str(dob), "sex": sex}
                summary["final_ml_risk_display"] = f"{r0['risk']*100:.1f}%"
                summary["metrics"] = {
                    "theta_alpha_ratio": r0["agg"].get("theta_alpha_ratio",0),
                    "theta_beta_ratio": r0["agg"].get("theta_beta_ratio",0),
                    "alpha_asym_F3_F4": r0["agg"].get("alpha_asym_F3_F4",0),
                    "mean_connectivity": r0.get("mean_conn", None)
                }
                summary["topo_images"]=r0.get("topo_imgs",{})
                summary["conn_image"]=r0.get("conn_img", None)
                # bar image: Theta/Alpha vs alpha asym
                try:
                    fig,ax=plt.subplots(figsize=(5.6,1.6))
                    ta = summary["metrics"]["theta_alpha_ratio"] or 0.0
                    aa = summary["metrics"]["alpha_asym_F3_F4"] or 0.0
                    ax.bar([0,1],[ta, aa], color=BLUE)
                    ax.set_xticks([0,1]); ax.set_xticklabels(["Theta/Alpha","Alpha Asym"])
                    buf=io.BytesIO(); fig.tight_layout(); fig.savefig(buf,format='png'); plt.close(fig); buf.seek(0)
                    summary["bar_img"]=buf.getvalue()
                except Exception:
                    summary["bar_img"]=None
                # SHAP
                shapimg=None; shaptab={}
                if SHAP_JSON.exists():
                    try:
                        sdata = load_shap(SHAP_JSON)
                        key = "depression_global" if summary["metrics"]["theta_alpha_ratio"]<=1.3 else "alzheimers_global"
                        feats = sdata.get(key,{})
                        shaptab = feats
                        shapimg = shap_plot_bytes(feats, top_n=10)
                    except Exception:
                        shaptab={}
                summary["shap_img"]=shapimg; summary["shap_table"]=shaptab if 'shaptab' in locals() else shaptab
                summary["tumor"]={"narrative": f"FDI max {r0['focal'].get('max_val')} at {r0['focal'].get('max_channel')}", "alerts":[str(a) for a in r0['focal'].get("alerts",[])]}
                # recommendations
                rec=[]
                rec.append("Correlate QEEG findings with clinical exam and PHQ-9/AD8.")
                if summary["metrics"]["theta_alpha_ratio"]>1.4 or r0['focal'].get('max_val',0) and r0['focal'].get('max_val',0)>2.0:
                    rec.append("Recommend neuroimaging (MRI) and neurology referral.")
                else:
                    rec.append("Consider clinical follow-up and repeat EEG in 3-6 months.")
                rec.append("Check reversible causes: B12, TSH, metabolic panel.")
                summary["recommendations"]=rec
                amiri = str(AMIRI_PATH) if AMIRI_PATH.exists() else None
                pdf = generate_pdf_report(summary, lang=("ar" if pdf_lang=="Arabic" else "en"), amiri_path=amiri)
                if pdf:
                    st.success("PDF ready.")
                    st.download_button("⬇️ Download Clinical Report (PDF)", data=pdf, file_name=f"NeuroEarly_Report_{now_ts()}.pdf", mime="application/pdf")
                else:
                    st.error("PDF generation returned nothing.")
        except Exception as e:
            st.error(f"PDF generation failed: {e}")
            st.error(safe_trace(e))

st.markdown("---")
st.markdown("<small>Prepared by Golden Bird LLC — NeuroEarly Pro</small>", unsafe_allow_html=True)

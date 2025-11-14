# app.py — NeuroEarly Pro (v6_3 Clinical Precision)
# Full bilingual (English default / Arabic optional RTL with Amiri),
# Band topomaps (approx), SHAP bar, PDF generation (reportlab),
# Risk estimates for Depression & Alzheimer's (simple interpretable engine).
# Place Amiri-Regular.ttf at project root (optional).

import os, io, sys, json, math, tempfile, traceback
from pathlib import Path
from datetime import datetime, date
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from scipy.signal import welch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.tri import Triangulation, LinearTriInterpolator
from PIL import Image

import streamlit as st

# optional heavy libs
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

# ----------------- Config -----------------
APP_TITLE = "NeuroEarly Pro — Clinical & Research"
LOGO_PATH = Path("assets/goldenbird_logo.png")  # update if your logo path differs
AMIRI_PATH = Path("Amiri-Regular.ttf")

BANDS = {
    "Delta": (1.0, 4.0),
    "Theta": (4.0, 8.0),
    "Alpha": (8.0, 13.0),
    "Beta": (13.0, 30.0),
    "Gamma": (30.0, 45.0)
}

# simple channel positions (approx for 10-20 subset) — used to produce interpolated topomap
STANDARD_2D_POS = {
    "Fp1": (-0.5, 0.9), "Fp2": (0.5, 0.9),
    "F3": (-0.7, 0.4), "F4": (0.7, 0.4),
    "C3": (-0.8, 0.0), "C4": (0.8, 0.0),
    "P3": (-0.7, -0.4), "P4": (0.7, -0.4),
    "O1": (-0.5, -0.9), "O2": (0.5, -0.9),
    "F7": (-0.95, 0.25), "F8": (0.95, 0.25),
    "T7": (-1.0, -0.1), "T8": (1.0, -0.1),
    "Fz": (0.0, 0.6), "Cz": (0.0, 0.0), "Pz": (0.0, -0.6)
}

# Translations (extend as needed)
T = {
    "en": {
        "language": "Language",
        "process": "Process EDF(s) and Analyze",
        "download_pdf": "Download PDF report",
        "upload_edf": "Upload EDF file",
        "patient_id": "Patient ID",
        "dob": "Date of Birth",
        "sex": "Sex",
        "meds": "Current meds (one per line)",
        "labs": "Relevant labs (B12, TSH, ...)",
        "phq9": "PHQ-9 (Depression screening)",
        "alz_q": "Alzheimer / Cognitive questions",
        "no_results": "No processed results yet. Upload EDF and press 'Process EDF(s) and Analyze'.",
        "edf_loaded": "EDF loaded successfully",
        "error_read": "Error reading EDF:",
        "risk_depr": "Depression risk",
        "risk_ad": "Alzheimer risk",
    },
    "ar": {
        "language": "اللغة",
        "process": "معالجة ملفات EDF وتحليل",
        "download_pdf": "تنزيل تقرير PDF",
        "upload_edf": "رفع ملف EDF",
        "patient_id": "معرّف المريض",
        "dob": "تاريخ الميلاد",
        "sex": "الجنس",
        "meds": "الأدوية الحالية (سطر لكل دواء)",
        "labs": "فحوصات ذات صلة (B12، TSH، ...)",
        "phq9": "PHQ-9 (فحص الاكتئاب)",
        "alz_q": "أسئلة الزهايمر/المعرفية",
        "no_results": "لا توجد نتائج معالجة بعد. ارفع ملف EDF واضغط 'معالجة'.",
        "edf_loaded": "تم تحميل EDF بنجاح",
        "error_read": "خطأ في قراءة EDF:",
        "risk_depr": "خطر الاكتئاب",
        "risk_ad": "خطر الخرف/الزهايمر",
    }
}

# ------------ Helper functions ------------
def now_ts(fmt="%Y%m%d_%H%M%S"):
    return datetime.utcnow().strftime(fmt)

def tr(key):
    lang = st.session_state.get("lang", "en")
    return T.get(lang, T["en"]).get(key, key)

def read_edf_bytes(uploaded) -> Tuple[Optional[Any], Optional[str]]:
    """Write uploaded bytes to temp file and read with mne if available."""
    if uploaded is None:
        return None, "No file"
    try:
        # write to temp file
        suffix = ".edf"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.getvalue())
            tmp.flush()
            tmp_path = tmp.name
        if HAS_MNE:
            raw = mne.io.read_raw_edf(tmp_path, preload=True, verbose=False)
            raw.info['temp_path'] = tmp_path  # keep reference for debugging (non-serializable)
            return raw, None
        else:
            return None, "mne not available"
    except Exception as e:
        return None, str(e)

def compute_band_powers(raw, bands=BANDS):
    """Compute absolute and relative band power (Welch) for each channel."""
    sf = raw.info['sfreq']
    data, ch_names = raw.get_data(return_times=False), raw.ch_names
    n_channels, n_samples = data.shape
    psd_freqs, psd = [], []
    band_results = {}
    for i in range(n_channels):
        f, Pxx = welch(data[i], sf, nperseg=min(2048, n_samples))
        psd_freqs = f
        psd.append(Pxx)
    psd = np.array(psd)  # ch x freq
    total_power = np.trapz(psd, psd_freqs, axis=1)
    for bname, (f0, f1) in bands.items():
        # find indices
        idx = np.where((psd_freqs >= f0) & (psd_freqs < f1))[0]
        if len(idx) == 0:
            vals = np.zeros(n_channels)
        else:
            vals = np.trapz(psd[:, idx], psd_freqs[idx], axis=1)
        band_results[bname] = {"abs": vals, "rel": np.divide(vals, total_power, out=np.zeros_like(vals), where=total_power>0)}
    # build dataframe
    rows = []
    for i,ch in enumerate(ch_names):
        r = {"ch": ch}
        for b in bands:
            r[f"{b}_abs"] = float(band_results[b]["abs"][i])
            r[f"{b}_rel"] = float(band_results[b]["rel"][i])
        rows.append(r)
    df = pd.DataFrame(rows)
    return df, band_results, ch_names, sf

def make_topomap_array(values, ch_names):
    """Interpolate channel values onto a 40x40 grid using STANDARD_2D_POS (approx).
       Returns 2D array for imshow."""
    pts = []
    vals = []
    for i,ch in enumerate(ch_names):
        if ch in STANDARD_2D_POS:
            x,y = STANDARD_2D_POS[ch]
            pts.append((x,y))
            vals.append(values[i])
    if len(pts) < 3:
        # fallback: show 1D bar-like image
        arr = np.tile(np.array(values)[:,None], (1,10)).T
        return arr, None
    pts = np.array(pts)
    vals = np.array(vals)
    # triangulation
    tri = Triangulation(pts[:,0], pts[:,1])
    interp = LinearTriInterpolator(tri, vals)
    grid_x = np.linspace(-1.1,1.1,64)
    grid_y = np.linspace(-1.1,1.1,64)
    gx, gy = np.meshgrid(grid_x, grid_y)
    z = interp(gx, gy)
    # mask NaNs outside convex hull
    z = np.where(np.isnan(z), np.nanmean(vals), z)
    return z, (grid_x, grid_y)

def plot_topomap_image(arr2d, title):
    fig, ax = plt.subplots(figsize=(3.2,2.6))
    if isinstance(arr2d, tuple):
        arr2d = arr2d[0]
    im = ax.imshow(arr2d, origin='lower', cmap='coolwarm', aspect='auto')
    ax.set_title(title, fontsize=10)
    ax.axis('off')
    fig.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format='png', dpi=120); plt.close(fig); buf.seek(0)
    return buf.getvalue()

def plot_bar_features(series, title):
    fig, ax = plt.subplots(figsize=(4.0,2.6))
    series.plot(kind='bar', ax=ax)
    ax.set_title(title, fontsize=10)
    ax.set_xticklabels(series.index, rotation=90, fontsize=7)
    fig.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format='png', dpi=120); plt.close(fig); buf.seek(0)
    return buf.getvalue()

def compute_risks(df_metrics):
    """Simple interpretable risk engine returning % for depression and alz.
       Uses theta/alpha ratio, alpha asymmetry proxy, and beta/alpha"""
    # global averages
    theta_rel = df_metrics.get("Theta_rel", 0)
    alpha_rel = df_metrics.get("Alpha_rel", 1e-6)
    beta_rel = df_metrics.get("Beta_rel", 0)
    # theta/alpha ratio
    tar = theta_rel/alpha_rel if alpha_rel>0 else 0.0
    bar = beta_rel/alpha_rel if alpha_rel>0 else 0.0
    # heuristic scaling to 0-100
    depr_score = (min(max((tar-0.4)/1.6, 0.0), 1.0)*0.6 + min(max((bar-0.3)/1.0,0),1.0)*0.2)*100
    # alzheimer score uses theta increase and reduction alpha, put weight on theta
    alz_score = (min(max((tar-0.6)/1.8,0),1)*0.7 + min(max((1.0-alpha_rel)/1.0,0),1)*0.3)*100
    return round(depr_score,1), round(alz_score,1), {"tar":tar, "bar":bar, "alpha_rel":alpha_rel, "theta_rel":theta_rel}

# ------------- PDF generator -------------
def generate_pdf_report(summary: dict, lang="en", amiri_path:Optional[str]=None) -> Optional[bytes]:
    if not HAS_REPORTLAB:
        return None
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4,
                            topMargin=36, bottomMargin=36, leftMargin=44, rightMargin=44)
    styles = getSampleStyleSheet()
    base_font = "Helvetica"
    if amiri_path and Path(amiri_path).exists():
        try:
            pdfmetrics.registerFont(TTFont("Amiri", str(amiri_path)))
            base_font = "Amiri"
        except Exception:
            pass
    styles.add(ParagraphStyle(name="TitleBlue", fontName=base_font, fontSize=18, textColor=colors.HexColor("#0b63d6"), alignment=1, spaceAfter=8))
    styles.add(ParagraphStyle(name="H2", fontName=base_font, fontSize=12, textColor=colors.HexColor("#0b63d6"), spaceAfter=6))
    styles.add(ParagraphStyle(name="Body", fontName=base_font, fontSize=10, leading=14))
    styles.add(ParagraphStyle(name="Note", fontName=base_font, fontSize=9, textColor=colors.grey))

    story = []
    story.append(Paragraph("NeuroEarly Pro — Clinical Report", styles["TitleBlue"]))
    # patient
    pt = summary.get("patient_info", {})
    story.append(Paragraph(f"{tr('patient_id')}: {pt.get('id','-')}", styles["Body"]))
    story.append(Paragraph(f"{tr('dob')}: {pt.get('dob','-')}", styles["Body"]))
    story.append(Spacer(1,8))

    # risks
    risks = summary.get("metrics",{})
    depr = risks.get("depression_pct", None)
    alz = risks.get("alz_pct", None)
    if depr is not None and alz is not None:
        story.append(Paragraph(f"{tr('risk_depr')}: {depr}%", styles["H2"]))
        story.append(Paragraph(f"{tr('risk_ad')}: {alz}%", styles["H2"]))
        story.append(Spacer(1,6))

    # embed images if exist (bar, topo per band, shap)
    if summary.get("normative_bar"):
        try:
            img = RLImage(io.BytesIO(summary["normative_bar"]), width=5.5*inch, height=3*inch)
            story.append(img); story.append(Spacer(1,6))
        except Exception:
            pass

    if summary.get("topo_images"):
        for band, bval in summary["topo_images"].items():
            try:
                img = RLImage(io.BytesIO(bval), width=3.2*inch, height=2.4*inch)
                story.append(Paragraph(band, styles["H2"]))
                story.append(img)
                story.append(Spacer(1,6))
            except Exception:
                continue

    # SHAP
    if summary.get("shap_img"):
        try:
            story.append(Paragraph("XAI: Feature contributions", styles["H2"]))
            story.append(RLImage(io.BytesIO(summary["shap_img"]), width=5.5*inch, height=2.6*inch))
            story.append(Spacer(1,6))
        except Exception:
            pass

    # Clinical narrative and recommendations
    story.append(Paragraph("<b>Clinical Interpretation</b>", styles["H2"]))
    narrative = summary.get("narrative","Automated screening - consult specialist for final interpretation.")
    story.append(Paragraph(narrative, styles["Body"]))
    story.append(Spacer(1,12))

    story.append(Paragraph("<b>Recommendations</b>", styles["H2"]))
    for r in summary.get("recommendations",[]):
        story.append(Paragraph("- " + r, styles["Body"]))
    story.append(Spacer(1,12))

    try:
        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()
    except Exception as e:
        print("PDF gen failed:", e)
        return None

# ----------------- Streamlit UI -----------------
st.set_page_config(page_title=APP_TITLE, layout="wide")

if "lang" not in st.session_state:
    st.session_state["lang"] = "en"

# Top header with gradient
st.markdown(f"""
<div style="background:linear-gradient(90deg,#0b63d6,#2ec4ff);padding:12px;border-radius:8px;color:white;margin-bottom:12px;display:flex;align-items:center;justify-content:space-between">
  <div style="font-size:20px;font-weight:700">{APP_TITLE}</div>
  <div style="font-size:12px">Prepared by Golden Bird LLC</div>
</div>
""", unsafe_allow_html=True)

# Layout: sidebar (left) + main (right)
left, right = st.columns([1,3])

with left:
    st.header("Settings")
    lang = st.selectbox(tr("language"), ["English","العربية"], index=0 if st.session_state["lang"]=="en" else 1)
    if lang.startswith("ع"):
        st.session_state["lang"] = "ar"
    else:
        st.session_state["lang"] = "en"

    st.text_input(tr("patient_id"), key="patient_id")
    st.date_input(tr("dob"), key="dob", value=date(1980,1,1), min_value=date(1900,1,1), max_value=date(2025,12,31))
    st.selectbox(tr("sex"), ["Unknown","Male","Female"], key="sex")
    st.text_area(tr("meds"), key="meds", height=80)
    st.text_area(tr("labs"), key="labs", height=80)

    st.markdown("---")
    st.file_uploader(tr("upload_edf"), type=["edf"], key="edf_file", help="Single EDF. If you have multiple, upload one at a time.")
    st.markdown("---")
    st.write("Questionnaires")
    # PHQ-9 simple
    st.subheader(tr("phq9"))
    phq_scores = []
    phq_cols = st.columns(3)
    # minimal PHQ-9 layout (9 questions)
    q_texts = ["Little interest/pleasure", "Feeling down", "Sleep issues", "Feeling tired", "Appetite changes", "Feeling bad about self", "Trouble concentrating", "Moving slowly or restless", "Thoughts of death"]
    phq_res = {}
    for i,q in enumerate(q_texts,1):
        v = st.radio(f"Q{i}", [0,1,2,3], index=0, key=f"phq{i}", horizontal=False)
        phq_res[f"Q{i}"] = v
    # Alzheimer simple questions (example)
    st.subheader(tr("alz_q"))
    alz_qs = {
        "memory_for_recent": "Difficulty remembering recent events?",
        "orientation": "Disorientation in time/place?",
        "language": "Trouble finding words?",
        "daily_tasks": "Difficulty in daily tasks?"
    }
    alz_res = {}
    for k,q in alz_qs.items():
        alz_res[k] = st.selectbox(q, ["No","Sometimes","Often"], index=0, key=f"alz_{k}")

with right:
    st.header("Console / Visualization")
    console = st.empty()
    main_area = st.container()

    if st.button(tr("process")):
        edf_file = st.session_state.get("edf_file")
        if edf_file is None:
            console.error(tr("no_results"))
        else:
            console.info("🔎 Saving and reading EDF file... please wait")
            raw, err = read_edf_bytes(edf_file)
            if raw is None:
                console.error(f"{tr('error_read')} {err}")
            else:
                console.success(f"{tr('edf_loaded')}. Shape: {raw.get_data().shape if HAS_MNE else 'N/A'}")
                try:
                    df_bands, band_results, ch_names, sf = compute_band_powers(raw)
                    # produce band topo images
                    topo_imgs = {}
                    for bname in BANDS:
                        vals = band_results[bname]["rel"]
                        arr2d = make_topomap_array(vals, ch_names)
                        imgbytes = plot_topomap_image(arr2d, f"{bname} ({BANDS[bname][0]}-{BANDS[bname][1]} Hz)")
                        topo_imgs[bname] = imgbytes

                    # normative bar: simple global metric (theta/alpha)
                    theta_mean = np.mean(band_results["Theta"]["rel"])
                    alpha_mean = np.mean(band_results["Alpha"]["rel"])
                    tar = theta_mean/alpha_mean if alpha_mean>0 else 0.0
                    s = pd.Series({ch: band_results["Theta"]["rel"][i]-band_results["Alpha"]["rel"][i] for i,ch in enumerate(ch_names)})
                    bar_img = plot_bar_features(s.abs().sort_values(ascending=False).head(12), "Top node differences (|Theta - Alpha|)")

                    # SHAP (if available)
                    shap_img = None
                    try:
                        if Path("shap_summary.json").exists():
                            with open("shap_summary.json","r",encoding="utf-8") as f:
                                ss = json.load(f)
                            # choose model key heuristic
                            model_key = "depression_global" if tar>1.3 else "alzheimers_global"
                            feat = ss.get(model_key,{})
                            if feat:
                                ser = pd.Series(feat).abs().sort_values(ascending=False)
                                shap_img = plot_bar_features(ser.head(10), "SHAP top features")
                    except Exception as e:
                        console.warning(f"SHAP load error: {e}")

                    # compute simple summary metrics (global)
                    metrics = {
                        "Theta_rel": float(theta_mean),
                        "Alpha_rel": float(alpha_mean),
                        "Theta/Alpha_ratio": float(tar)
                    }
                    depr_pct, alz_pct, aux = compute_risks(metrics)

                    # show results
                    with main_area:
                        st.subheader("QEEG Band summary (relative power)")
                        st.dataframe(df_bands.style.format({c: "{:.4f}" for c in df_bands.columns if c!="ch"}), height=320)

                        st.subheader("Topomaps by band")
                        cols = st.columns(3)
                        i=0
                        for bname,img in topo_imgs.items():
                            cols[i%3].image(img, caption=bname, use_column_width=True)
                            i+=1

                        st.subheader("Risk & interpretation")
                        st.metric(tr("risk_depr"), f"{depr_pct}%")
                        st.metric(tr("risk_ad"), f"{alz_pct}%")

                        st.markdown("**Interpretation:**")
                        interp = f"Global Theta/Alpha ratio = {tar:.2f}. This indicates {'increased slowing' if tar>0.6 else 'within normal range'}. Consider clinical correlation and labs."
                        st.write(interp)

                        # show SHAP
                        if shap_img:
                            st.subheader("XAI: SHAP contributions")
                            st.image(shap_img, use_column_width=True)

                        # provide CSV
                        csv = df_bands.to_csv(index=False).encode("utf-8")
                        st.download_button("Download metrics (CSV)", data=csv, file_name=f"NeuroEarly_metrics_{now_ts()}.csv", mime="text/csv")

                        # prepare PDF content summary
                        summary = {
                            "patient_info": {"id": st.session_state.get("patient_id",""), "dob": str(st.session_state.get("dob",""))},
                            "metrics": {"depression_pct": depr_pct, "alz_pct": alz_pct},
                            "topo_images": topo_imgs,
                            "normative_bar": bar_img,
                            "shap_img": shap_img,
                            "narrative": interp,
                            "recommendations": [
                                "Consider blood tests: B12, TSH, electrolytes.",
                                "If cognitive symptoms progressive — neuropsychology and MRI as indicated.",
                                "Follow-up EEG in 3-6 months if clinical concern."
                            ],
                            "created": now_ts()
                        }
                        pdf_bytes = generate_pdf_report(summary, lang=st.session_state.get("lang","en"), amiri_path=str(AMIRI_PATH) if AMIRI_PATH.exists() else None)
                        if pdf_bytes:
                            st.download_button(tr("download_pdf"), data=pdf_bytes, file_name=f"NeuroEarly_Report_{now_ts()}.pdf", mime="application/pdf")
                        else:
                            st.error("PDF generation failed — reportlab missing or error.")
                except Exception as e:
                    console.exception(f"Processing exception: {e}\n{traceback.format_exc()}")

    else:
        st.info(tr("no_results"))

# footer note
st.markdown("---")
st.markdown("<small>For clinical use: this is an aid, not a standalone diagnosis. Validate with clinical exam and further tests.</small>", unsafe_allow_html=True)

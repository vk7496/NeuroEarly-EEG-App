# app.py — NeuroEarly Pro: advanced EEG preprocessing + bilingual PDF report
import io
import os
import sys
import json
import tempfile
import math
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import mne

# PDF / fonts / RTL support
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# Try import Arabic shaping libs (optional, recommended for Arabic PDF)
try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    ARABIC_SUPPORT = True
except Exception:
    ARABIC_SUPPORT = False

# ---------- Config ----------
st.set_page_config(page_title="NeuroEarly Pro", layout="wide")
st.sidebar.title("NeuroEarly Pro")
st.sidebar.info("Research demo — Not a clinical diagnosis")

LANG = st.sidebar.selectbox("Language / اللغة", options=["English", "العربية"])
IS_EN = (LANG == "English")

# ---------- Texts ----------
TEXT = {
    "English": {
        "title": "🧠 NeuroEarly Pro — EEG Screening (Demo)",
        "subtitle": "Advanced preprocessing, PHQ-9, AD8 and professional PDF report (research only).",
        "upload": "Upload EDF file(s)",
        "upload_hint": "Upload one EDF for single session or multiple EDFs for longitudinal trend.",
        "phq": "PHQ-9 (Depression)",
        "ad8": "AD8 (Cognitive screening)",
        "generate": "Generate Reports (JSON / PDF / CSV)",
        "note": "Research demo. Not a clinical diagnostic tool."
    },
    "العربية": {
        "title": "🧠 نيوروإيرلي برو — فحص EEG (نموذج تجريبي)",
        "subtitle": "معالجة متقدمة، PHQ-9، AD8 وتقرير PDF احترافي (للبحث فقط).",
        "upload": "تحميل ملفات EDF",
        "upload_hint": "ارفع ملف EDF واحد للجلسة الواحدة أو عدة ملفات لتحليل طولي.",
        "phq": "PHQ-9 (الاكتئاب)",
        "ad8": "AD8 (الفحص المعرفي)",
        "generate": "إنشاء التقارير (JSON / PDF / CSV)",
        "note": "هذا نموذج بحثي. ليس تشخيصًا طبيًا."
    }
}
T = TEXT["English"] if IS_EN else TEXT["العربية"]

st.title(T["title"])
st.caption(T["subtitle"])

# ---------- Bands ----------
BAND_DEFS = {
    "Delta (0.5–4 Hz)": (0.5, 4),
    "Theta (4–8 Hz)": (4, 8),
    "Alpha (8–12 Hz)": (8, 12),
    "Beta (12–30 Hz)": (12, 30),
    "Gamma (30–45 Hz)": (30, 45)
}

# ---------- Helpers: EEG preprocessing ----------
def preprocess_raw(raw, l_freq=0.5, h_freq=45.0):
    """Bandpass + notch + try ICA (safe). Returns cleaned raw and metadata."""
    try:
        raw.load_data()
    except Exception:
        pass
    # bandpass
    try:
        raw.filter(l_freq, h_freq, fir_design="firwin", verbose=False)
    except Exception:
        pass
    # notch: detect mains frequency by inspecting power at 50/60
    try:
        sf = raw.info.get("sfreq", 256.0)
        # apply both just in case
        for mains in (50.0, 60.0):
            try:
                raw.notch_filter(freqs=[mains], verbose=False)
            except Exception:
                pass
    except Exception:
        pass

    # Attempt ICA for artifact removal if enough channels
    ica_applied = False
    try:
        picks = mne.pick_types(raw.info, eeg=True)
        n_ch = len(picks)
        if n_ch >= 4:
            n_components = min(15, n_ch - 1)
            ica = mne.preprocessing.ICA(n_components=n_components, random_state=42, max_iter="auto")
            ica.fit(raw)
            # try to find EOG; if not available, skip exclusion
            try:
                eog_inds, scores = ica.find_bads_eog(raw)
                if eog_inds:
                    ica.exclude = eog_inds
            except Exception:
                pass
            try:
                ica.apply(raw)
                ica_applied = True
            except Exception:
                ica_applied = False
    except Exception:
        ica_applied = False

    return raw, {"ica_applied": ica_applied}

def compute_band_powers(raw, fmin=0.5, fmax=45.0):
    """Compute Welch PSD and integrate per band across channels."""
    try:
        psd = raw.compute_psd(method="welch", fmin=fmin, fmax=fmax, n_fft=2048, n_overlap=1024, verbose=False)
        psds, freqs = psd.get_data(return_freqs=True)
        mean_psd = psds.mean(axis=0) if psds.ndim==2 else psds
    except Exception:
        data = raw.get_data()
        sf = int(raw.info.get("sfreq", 256))
        N = min(4096, data.shape[1])
        freqs = np.fft.rfftfreq(N, 1.0/sf)
        mean_psd = np.abs(np.fft.rfft(data.mean(axis=0)[:N], n=N))
    band_powers = {}
    for name, (lo, hi) in BAND_DEFS.items():
        mask = (freqs >= lo) & (freqs < hi)
        band_powers[name] = float(np.trapz(mean_psd[mask], freqs[mask])) if mask.any() else 0.0
    return band_powers, freqs

# ---------- Plot helpers ----------
def plot_band_png(band_powers):
    labels = list(band_powers.keys())
    vals = [band_powers[k] for k in labels]
    fig, ax = plt.subplots(figsize=(7,2.8), dpi=140)
    ax.bar(labels, vals)
    ax.set_ylabel("Integrated power (a.u.)")
    ax.set_title("EEG Band Powers (Welch PSD)")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig); buf.seek(0)
    return buf.getvalue()

def plot_signal_png(raw, seconds=8):
    picks = mne.pick_types(raw.info, eeg=True)
    data = raw.get_data(picks=picks) if len(picks)>0 else raw.get_data()
    ch0 = data[0] if data.ndim==2 else data
    sf = int(raw.info.get("sfreq", 256))
    n = min(len(ch0), seconds*sf)
    t = np.arange(n)/sf
    fig, ax = plt.subplots(figsize=(7,2.4), dpi=140)
    ax.plot(t, ch0[:n]); ax.set_xlabel("Time (s)"); ax.set_ylabel("Amplitude (a.u.)")
    ax.set_title(f"EEG snippet (~{n/sf:.1f}s)")
    fig.tight_layout(); buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig); buf.seek(0)
    return buf.getvalue()

# ---------- Heuristics & Early Risk ----------
def eeg_heuristics(band_powers):
    alpha = band_powers.get("Alpha (8–12 Hz)", 1e-9)
    theta = band_powers.get("Theta (4–8 Hz)", 0.0)
    beta = band_powers.get("Beta (12–30 Hz)", 0.0)
    theta_alpha = theta/alpha if alpha>0 else 0.0
    beta_alpha = beta/alpha if alpha>0 else 0.0
    return {"Theta/Alpha": round(theta_alpha,3), "Beta/Alpha": round(beta_alpha,3)}

def compute_early_index(band_powers, phq_score, ad8_score, weights=(0.5,0.3,0.2)):
    heur = eeg_heuristics(band_powers)
    ta = min(heur["Theta/Alpha"], 2.0)/2.0  # normalize heuristically to [0,1]
    ba_inv = min(max(1.0 - heur["Beta/Alpha"], 0.0), 1.0)
    eeg_comp = (ta + ba_inv)/2.0
    phq_norm = min(max(phq_score/27.0, 0.0), 1.0)
    ad8_norm = min(max(ad8_score/8.0, 0.0), 1.0)
    idx = weights[0]*eeg_comp + weights[1]*phq_norm + weights[2]*ad8_norm
    return min(max(idx,0.0),1.0), {"eeg_comp": round(eeg_comp,3), "phq_norm": round(phq_norm,3), "ad8_norm": round(ad8_norm,3)}

# ---------- PDF helpers (fonts + Arabic shaping) ----------
FONT_DIR = "fonts"
AMIRI_TTF = os.path.join(FONT_DIR, "Amiri-Regular.ttf")
AMIRI_URL = "https://github.com/alif-type/amiri/raw/master/Amiri-Regular.ttf"

def ensure_amiri_font():
    """Ensure Amiri font exists locally; try to download if not present."""
    os.makedirs(FONT_DIR, exist_ok=True)
    if not os.path.exists(AMIRI_TTF):
        try:
            import urllib.request
            urllib.request.urlretrieve(AMIRI_URL, AMIRI_TTF)
        except Exception:
            return False
    try:
        pdfmetrics.registerFont(TTFont("Amiri", AMIRI_TTF))
        return True
    except Exception:
        return False

def shape_text_for_pdf(text, lang):
    """If Arabic and reshaper available, return shaped display text for ReportLab."""
    if not text:
        return ""
    if lang == "العربية":
        if ARABIC_SUPPORT:
            reshaped = arabic_reshaper.reshape(text)
            bidi_text = get_display(reshaped)
            return bidi_text
        else:
            # fallback: return text as-is (may not render correctly without font)
            return text
    return text

def build_pdf_bytes(results, band_png=None, sig_png=None, lang="English"):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4)
    styles = getSampleStyleSheet()
    flow = []

    # register Amiri if Arabic
    if lang == "العربية":
        ensure_amiri_font()
        # create paragraph style that uses Amiri if available
        try:
            styles.add(ParagraphStyle(name="Arabic", fontName="Amiri", fontSize=10))
            para_style = styles["Arabic"]
        except Exception:
            para_style = styles["Normal"]
    else:
        para_style = styles["Normal"]

    title = shape_text_for_pdf("NeuroEarly Pro — Report" if lang=="English" else "تقرير نيوروإيرلي برو", lang)
    flow.append(Paragraph(title, para_style))
    flow.append(Spacer(1,6))
    flow.append(Paragraph(shape_text_for_pdf(f"Timestamp: {results['timestamp']}", lang), para_style))
    flow.append(Spacer(1,8))

    # EEG summary table
    flow.append(Paragraph(shape_text_for_pdf("EEG Summary", lang), para_style))
    eeg = results.get("EEG", {})
    rows = [["Metric", "Value"]]
    rows.append(["File", eeg.get("filename","-")])
    rows.append(["Sampling rate (Hz)", str(eeg.get("sfreq","-"))])
    for k,v in eeg.get("bands",{}).items():
        rows.append([shape_text_for_pdf(k, lang), f"{v:.6g}"])
    table = Table(rows, colWidths=[200,300])
    table.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.lightgrey),("GRID",(0,0),(-1,-1),0.25,colors.grey)]))
    flow.append(table); flow.append(Spacer(1,8))

    # images
    if band_png:
        flow.append(RLImage(io.BytesIO(band_png), width=420, height=160)); flow.append(Spacer(1,6))
    if sig_png:
        flow.append(RLImage(io.BytesIO(sig_png), width=420, height=140)); flow.append(Spacer(1,8))

    # PHQ9 & AD8 summary + detailed answers
    flow.append(Paragraph(shape_text_for_pdf("PHQ-9 (Depression)", lang), para_style))
    phq = results.get("PHQ9", {})
    flow.append(Paragraph(shape_text_for_pdf(f"Score: {phq.get('score','-')} — {phq.get('label','')}", lang), para_style))
    # answers table
    phq_qs = phq.get("questions", [])
    phq_ans = phq.get("answers", [])
    if phq_qs:
        ptab = [["Question", "Answer"]]
        for q,a in zip(phq_qs, phq_ans):
            ptab.append([shape_text_for_pdf(q, lang), str(a)])
        t = Table(ptab, colWidths=[320,180])
        t.setStyle(TableStyle([("GRID",(0,0),(-1,-1),0.25,colors.grey),("BACKGROUND",(0,0),(-1,0),colors.lightgrey),("FONTSIZE",(0,0),(-1,-1),8)]))
        flow.append(t)
    flow.append(Spacer(1,8))

    flow.append(Paragraph(shape_text_for_pdf("AD8 (Cognition)", lang), para_style))
    ad8 = results.get("AD8", {})
    flow.append(Paragraph(shape_text_for_pdf(f"Score: {ad8.get('score','-')} — {ad8.get('label','')}", lang), para_style))
    if ad8.get("questions"):
        atab = [["Question","Answer"]]
        for q,a in zip(ad8.get("questions"), ad8.get("answers")):
            atab.append([shape_text_for_pdf(q, lang), "Yes" if a==1 and lang=="English" else ("نعم" if a==1 and lang=="العربية" else ("No" if lang=="English" else "لا"))])
        t2 = Table(atab, colWidths=[320,180])
        t2.setStyle(TableStyle([("GRID",(0,0),(-1,-1),0.25,colors.grey),("BACKGROUND",(0,0),(-1,0),colors.lightgrey),("FONTSIZE",(0,0),(-1,-1),8)]))
        flow.append(t2)
    flow.append(Spacer(1,8))

    # Early index
    early = results.get("EarlyRisk", {})
    flow.append(Paragraph(shape_text_for_pdf("Early Risk Index", lang), para_style))
    flow.append(Paragraph(shape_text_for_pdf(f"Index: {early.get('index','-')} — components: {early.get('components','')}", lang), para_style))
    flow.append(Spacer(1,8))

    # Interpretation
    interp_en = ("This report is a research demo. Elevated theta/alpha may relate to depressive patterns; "
                 "low beta/alpha may suggest cognitive concerns. Results are preliminary and require clinical follow-up.")
    interp_ar = ("هذا التقرير بحثي. قد يرتبط ارتفاع نسبة theta/alpha بأنماط اكتئابية؛ قد يشير انخفاض beta/alpha إلى مخاوف إدراكية. "
                 "النتائج أولية ويجب متابعتها سريريًا.")
    flow.append(Paragraph(shape_text_for_pdf(interp_en if lang=="English" else interp_ar, lang), para_style))
    flow.append(Spacer(1,10))
    flow.append(Paragraph(shape_text_for_pdf(T.get("note",""), lang), para_style))

    doc.build(flow)
    buf.seek(0)
    return buf.getvalue()

# ---------- UI: Upload ----------
st.header(T["upload"])
st.write(T["upload_hint"])
uploaded_files = st.file_uploader("Select EDF file(s)", type=["edf"], accept_multiple_files=True)

session_list = []
if uploaded_files:
    st.info(f"{len(uploaded_files)} file(s) selected.")
    for f in uploaded_files:
        with st.spinner(f"Processing {f.name} ..."):
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmp:
                    tmp.write(f.read()); tmp_path = tmp.name
                raw = mne.io.read_raw_edf(tmp_path, preload=True, verbose=False)
                # preprocessing
                raw_clean, meta = preprocess_raw(raw)
                bands, freqs = compute_band_powers(raw_clean)
                heur = eeg_heuristics(bands)
                sig_png = plot_signal_png(raw_clean, seconds=8)
                band_png = plot_band_png(bands)
                session_list.append({
                    "filename": f.name,
                    "sfreq": float(raw.info.get("sfreq", math.nan)),
                    "bands": bands,
                    "heuristics": heur,
                    "meta": meta,
                    "sig_png": sig_png,
                    "band_png": band_png
                })
                st.success(f"{f.name} processed — channels: {len(mne.pick_types(raw.info,eeg=True))}, ICA: {'applied' if meta.get('ica_applied') else 'skipped'}")
                st.image(band_png, use_column_width=True)
            except Exception as e:
                st.error(f"Failed to process {f.name}: {e}")

# ---------- UI: PHQ-9 ----------
st.header(T["phq"])
PHQ_QUESTIONS_EN = [
    "Little interest or pleasure in doing things",
    "Feeling down, depressed, or hopeless",
    "Trouble falling or staying asleep, or sleeping too much",
    "Feeling tired or having little energy",
    "Poor appetite or overeating (specify type below)",
    "Feeling bad about yourself — or feeling like a failure",
    "Trouble concentrating on things",
    "Moving or speaking slowly or being very fidgety",
    "Thoughts that you would be better off dead or of hurting yourself"
]
PHQ_QUESTIONS_AR = [
    "قلة الاهتمام أو المتعة في القيام بالأنشطة",
    "الشعور بالحزن أو الاكتئاب أو اليأس",
    "مشاكل في النوم أو النوم المفرط",
    "الشعور بالتعب أو قلة الطاقة",
    "فقدان الشهية أو الإفراط في الأكل (حدد النوع أدناه)",
    "الشعور بأنك شخص سيء أو فاشل",
    "صعوبة في التركيز",
    "الحركة أو الكلام ببطء أو قلق/فرط الحركة",
    "أفكار بأنك أفضل حالاً ميتاً أو أفكار إيذاء النفس"
]
PHQ_OPTS_EN = ["0 = Not at all", "1 = Several days", "2 = More than half the days", "3 = Nearly every day"]
PHQ_OPTS_AR = ["0 = أبداً", "1 = عدة أيام", "2 = أكثر من نصف الأيام", "3 = كل يوم تقريباً"]

phq_qs = PHQ_QUESTIONS_EN if IS_EN else PHQ_QUESTIONS_AR
phq_opts = PHQ_OPTS_EN if IS_EN else PHQ_OPTS_AR

phq_answers = []
for i, q in enumerate(phq_qs, start=1):
    ans = st.selectbox(f"{i}. {q}", phq_opts, key=f"phq_{i}_{LANG}")
    # parse numeric prefix
    try:
        val = int(ans.split("=")[0].strip())
    except Exception:
        val = 0
    phq_answers.append(val)

# Q5 type (appetite)
if IS_EN:
    appetite_type = st.radio("If appetite change: which?", ["Poor appetite", "Overeating"], key="q5_type")
else:
    appetite_type = st.radio("إذا كان هناك تغيّر في الشهية: ما نوعه؟", ["فقدان الشهية", "الإفراط في الأكل"], key="q5_type")

phq_score = sum(phq_answers)
if phq_score < 5:
    phq_label = "Minimal" if IS_EN else "طفيف"
elif phq_score < 10:
    phq_label = "Mild" if IS_EN else "خفيف"
elif phq_score < 15:
    phq_label = "Moderate" if IS_EN else "متوسط"
elif phq_score < 20:
    phq_label = "Moderately severe" if IS_EN else "شديد إلى حد ما"
else:
    phq_label = "Severe" if IS_EN else "شديد"

st.write((f"PHQ-9: {phq_score} / 27 — {phq_label}") if IS_EN else (f"درجة PHQ-9: {phq_score} / ٢٧ — {phq_label}"))

# ---------- UI: AD8 ----------
st.header(T["ad8"])
AD8_QS_EN = [
    "Problems with judgment (e.g., bad financial decisions)",
    "Reduced interest in hobbies/activities",
    "Repeats the same questions or stories",
    "Trouble learning to use a tool or gadget",
    "Forgets the correct month or year",
    "Difficulty handling finances (e.g., paying bills)",
    "Trouble remembering appointments",
    "Everyday thinking is getting worse"
]
AD8_QS_AR = [
    "مشاكل في الحكم أو اتخاذ القرار",
    "انخفاض الاهتمام بالهوايات/الأنشطة",
    "تكرار نفس الأسئلة أو القصص",
    "صعوبة في تعلم استخدام أداة أو جهاز",
    "نسيان الشهر أو السنة الصحيحة",
    "صعوبة في إدارة الشؤون المالية (مثل دفع الفواتير)",
    "صعوبة في تذكر المواعيد",
    "تدهور التفكير اليومي"
]
ad8_qs = AD8_QS_EN if IS_EN else AD8_QS_AR
ad8_opts = ["No", "Yes"] if IS_EN else ["لا", "نعم"]
ad8_answers = []
for i,q in enumerate(ad8_qs, start=1):
    ans = st.selectbox(f"{i}. {q}", ad8_opts, key=f"ad8_{i}_{LANG}")
    ad8_answers.append(1 if ans == ( "Yes" if IS_EN else "نعم") else 0)
ad8_score = sum(ad8_answers)
ad8_label = ("Possible concern (≥2)" if ad8_score>=2 else "Low") if IS_EN else ("احتمال قلق (≥٢)" if ad8_score>=2 else "منخفض")
st.write((f"AD8: {ad8_score} / 8 — {ad8_label}") if IS_EN else (f"درجة AD8: {ad8_score} / ٨ — {ad8_label}"))

# ---------- Generate Reports ----------
st.header(T["generate"])
if st.button("Generate Reports (JSON / PDF / CSV)"):
    # sessions -> use last session if exists
    sessions_meta = []
    for s in session_list:
        sessions_meta.append({
            "filename": s["filename"],
            "sfreq": s["sfreq"],
            "bands": s["bands"],
            "heuristics": s["heuristics"],
            "ica_applied": s["meta"].get("ica_applied", False)
        })
    last_bands = session_list[-1]["bands"] if session_list else {k:0.0 for k in BAND_DEFS.keys()}
    early_idx, early_comp = compute_early_index(last_bands, phq_score, ad8_score)

    results = {
        "timestamp": datetime.now().isoformat(),
        "language": LANG,
        "PHQ9": {"score": phq_score, "label": phq_label, "answers": phq_answers, "appetite_type": appetite_type, "questions": phq_qs},
        "AD8": {"score": ad8_score, "label": ad8_label, "answers": ad8_answers, "questions": ad8_qs},
        "EarlyRisk": {"index": round(early_idx,3), "components": early_comp},
        "sessions": sessions_meta,
        "note": T.get("note","")
    }

    # JSON download
    json_bytes = io.BytesIO(json.dumps(results, indent=2, ensure_ascii=False).encode("utf-8"))
    st.download_button(label="Download JSON", data=json_bytes, file_name="neuroearly_report.json", mime="application/json")

    # CSV features (sessions)
    if sessions_meta:
        df_rows = []
        for s in sessions_meta:
            r = {"filename": s["filename"], "sfreq": s["sfreq"]}
            r.update(s["bands"])
            r.update({k: v for k,v in s["heuristics"].items()})
            df_rows.append(r)
        df = pd.DataFrame(df_rows)
        csv_buf = io.StringIO(); df.to_csv(csv_buf, index=False)
        st.download_button(label="Download CSV (features)", data=csv_buf.getvalue(), file_name="neuroearly_features.csv", mime="text/csv")

    # PDF
    band_png = session_list[-1]["band_png"] if session_list else None
    sig_png = session_list[-1]["sig_png"] if session_list else None
    pdf_bytes = build_pdf_bytes(results, band_png=band_png, sig_png=sig_png, lang=("English" if IS_EN else "العربية"))
    st.download_button(label="Download PDF (advanced)", data=pdf_bytes, file_name="neuroearly_advanced_report.pdf", mime="application/pdf")

st.caption(T.get("note",""))

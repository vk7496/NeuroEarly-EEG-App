# app.py — NeuroEarly Pro (fixed font + PHQ tweaks + optional ICA)
import os
import io
import json
import tempfile
import requests
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import mne

from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
)
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors

# ---------------------------
# Texts (EN / AR)
# ---------------------------
TEXTS = {
    "en": {
        "title": "🧠 NeuroEarly Pro — EEG + PHQ-9 + AD8",
        "subtitle": "Prototype for early Alzheimer’s & Depression risk screening using EEG, questionnaires and cognitive micro-tasks.",
        "upload": "1) Upload EEG file (.edf)",
        "phq9": "2) Depression Screening — PHQ-9",
        "ad8": "3) Cognitive Screening — AD8",
        "report": "4) Generate Report",
        "download_json": "⬇️ Download JSON",
        "download_pdf": "⬇️ Download PDF",
        "note": "⚠️ Research demo only — Not a clinical diagnostic tool.",
        "phq9_questions": [
            "Little interest or pleasure in doing things",
            "Feeling down, depressed, or hopeless",
            "Trouble falling or staying asleep, or sleeping too much",
            "Feeling tired or having little energy",
            "Poor appetite or overeating (direction will be asked if present)",
            "Feeling bad about yourself or feeling like a failure",
            "Trouble concentrating (e.g., reading, watching TV)",
            "Moving or speaking slowly OR being very restless (direction will be asked if present)",
            "Thoughts of being better off dead or self-harm"
        ],
        "phq9_options": [
            "0 = Not at all", "1 = Several days",
            "2 = More than half the days", "3 = Nearly every day"
        ],
        "appetite_follow": ["No preference", "Decreased appetite", "Increased appetite"],
        "movement_follow": ["No preference", "Moving/speaking slowly", "Being restless"],
        "ad8_questions": [
            "Problems with judgment (e.g., poor financial decisions)",
            "Reduced interest in hobbies/activities",
            "Repeats questions or stories",
            "Trouble using a tool or gadget",
            "Forgets the correct month or year",
            "Difficulty managing finances (e.g., paying bills)",
            "Trouble remembering appointments",
            "Everyday thinking is getting worse"
        ],
        "ad8_options": ["No", "Yes"]
    },
    "ar": {
        "title": "🧠 نيوروإيرلي برو — تخطيط الدماغ + PHQ-9 + AD8",
        "subtitle": "نموذج أولي للفحص المبكر لمخاطر الزهايمر والاكتئاب باستخدام EEG والاستبيانات والاختبارات المعرفية الصغيرة.",
        "upload": "١) تحميل ملف تخطيط الدماغ (.edf)",
        "phq9": "٢) فحص الاكتئاب — PHQ-9",
        "ad8": "٣) الفحص المعرفي — AD8",
        "report": "٤) إنشاء التقرير",
        "download_json": "⬇️ تنزيل JSON",
        "download_pdf": "⬇️ تنزيل PDF",
        "note": "⚠️ هذا نموذج بحثي فقط — ليس أداة تشخيص سريري.",
        "phq9_questions": [
            "قلة الاهتمام أو المتعة في القيام بالأنشطة",
            "الشعور بالحزن أو الاكتئاب أو اليأس",
            "مشاكل في النوم أو النوم المفرط",
            "الشعور بالتعب أو قلة الطاقة",
            "فقدان الشهية أو الإفراط في الأكل (سيُسأل عن الاتجاه إذا وُجد)",
            "الشعور بأنك شخص سيء أو فاشل",
            "صعوبة في التركيز (مثل القراءة أو مشاهدة التلفاز)",
            "الحركة أو الكلام ببطء شديد أو فرط الحركة (سيُسأل عن الاتجاه إذا وُجد)",
            "أفكار بأنك أفضل حالاً ميتاً أو أفكار إيذاء النفس"
        ],
        "phq9_options": [
            "0 = أبداً", "1 = عدة أيام",
            "2 = أكثر من نصف الأيام", "3 = كل يوم تقريباً"
        ],
        "appetite_follow": ["لا تفضيل", "انخفاض الشهية", "زيادة الشهية"],
        "movement_follow": ["لا تفضيل", "الحركة/الحديث ببطء", "القلق/فرط الحركة"],
        "ad8_questions": [
            "مشاكل في الحكم أو اتخاذ القرارات",
            "انخفاض الاهتمام بالهوايات أو الأنشطة",
            "تكرار الأسئلة أو القصص",
            "صعوبة في استخدام أداة أو جهاز",
            "نسيان الشهر أو السنة الصحيحة",
            "صعوبة في إدارة الشؤون المالية (مثل دفع الفواتير)",
            "صعوبة في تذكر المواعيد",
            "تدهور التفكير اليومي"
        ],
        "ad8_options": ["لا", "نعم"]
    }
}

# ---------------------------
# Font (Amiri) — save to temp file then register
# ---------------------------
def ensure_amiri(st_logger=None):
    # If already registered, skip
    if "Amiri" in pdfmetrics.getRegisteredFontNames():
        return True
    # 1) Prefer a local bundled font (repo/fonts/Amiri-Regular.ttf)
    local_path = os.path.join(os.path.dirname(__file__), "fonts", "Amiri-Regular.ttf") if "__file__" in globals() else None
    try:
        if local_path and os.path.exists(local_path):
            pdfmetrics.registerFont(TTFont("Amiri", local_path))
            return True
    except Exception as e:
        if st_logger:
            st_logger.warning(f"Local Amiri font load failed: {e}")

    # 2) Try to download to a temp file and register
    url = "https://github.com/alif-type/amiri-font/raw/master/ttf/Amiri-Regular.ttf"
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        tmpf = tempfile.NamedTemporaryFile(delete=False, suffix=".ttf")
        tmpf.write(r.content)
        tmpf.flush()
        tmpf.close()
        pdfmetrics.registerFont(TTFont("Amiri", tmpf.name))
        return True
    except Exception as e:
        if st_logger:
            st_logger.warning(f"Could not load Amiri font automatically: {e}")
        return False

# ---------------------------
# EEG helpers (filter + PSD)
# ---------------------------
def preprocess_eeg(raw: mne.io.BaseRaw, apply_ica=False, st_logger=None):
    # band-pass and notch first
    try:
        raw.filter(0.5, 45, fir_design="firwin", verbose=False)
        # try both 50 and 60 to handle different countries
        raw.notch_filter(freqs=[50.0, 60.0], verbose=False)
    except Exception as e:
        if st_logger:
            st_logger.warning(f"Basic filtering failed: {e}")

    # Optional ICA cleaning
    if apply_ica:
        try:
            nchan = raw.info.get("nchan", 0)
            if nchan >= 4:
                n_comp = min(15, nchan - 1)
                ica = mne.preprocessing.ICA(n_components=n_comp, random_state=97, max_iter="auto")
                ica.fit(raw)
                # try find EOG by correlation (best-effort)
                try:
                    eog_inds, scores = ica.find_bads_eog(raw, threshold=3.0)
                    ica.exclude = eog_inds
                except Exception:
                    # attempt frontal channels if present
                    picks = mne.pick_channels_regexp(raw.ch_names, "Fp|AF|Fp")
                    try:
                        eog_inds, scores = ica.find_bads_eog(raw, picks=picks)
                        ica.exclude = eog_inds
                    except Exception:
                        pass
                ica.apply(raw)
                if st_logger:
                    st_logger.success("ICA applied (best-effort).")
            else:
                if st_logger:
                    st_logger.info("ICA skipped: not enough channels.")
        except Exception as e:
            if st_logger:
                st_logger.warning(f"ICA failed or was skipped: {e}")
    return raw

def compute_band_powers(raw: mne.io.BaseRaw):
    psd = raw.compute_psd(fmin=0.5, fmax=45.0, method="welch", n_fft=2048, n_overlap=1024, verbose=False)
    psds, freqs = psd.get_data(return_freqs=True)
    # average across channels if multi-dim
    mean_psd = psds.mean(axis=0) if psds.ndim == 2 else psds
    bands = {"Delta": (0.5, 4), "Theta": (4, 8), "Alpha": (8, 12), "Beta": (12, 30), "Gamma": (30, 45)}
    powers = {}
    for name, (lo, hi) in bands.items():
        mask = (freqs >= lo) & (freqs < hi)
        powers[name] = float(np.trapz(mean_psd[mask], freqs[mask])) if mask.any() else 0.0
    return powers

def plot_bands(powers: dict):
    fig, ax = plt.subplots(figsize=(7,3))
    ax.bar(list(powers.keys()), list(powers.values()))
    ax.set_ylabel("Integrated power (a.u.)")
    ax.set_title("EEG Band Powers")
    plt.xticks(rotation=20)
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

# ---------------------------
# PDF builder (uses Amiri if available)
# ---------------------------
def build_pdf(results: dict, lang="en", band_png=None):
    # Try to ensure Amiri font; if fails, proceed with default fonts and warn
    font_ok = ensure_amiri(st_logger=st)
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4)
    styles = getSampleStyleSheet()
    if font_ok and lang == "ar":
        for style_name in ("Normal", "Heading2", "Title", "Italic"):
            try:
                styles[style_name].fontName = "Amiri"
            except Exception:
                pass

    flow = []
    flow.append(Paragraph(TEXTS[lang]["title"], styles["Title"]))
    flow.append(Paragraph(TEXTS[lang]["subtitle"], styles["Normal"]))
    flow.append(Spacer(1, 12))

    # EEG table
    flow.append(Paragraph("EEG Results:", styles["Heading2"]))
    rows = [["Band", "Power"]]
    for k, v in results.get("bands", {}).items():
        rows.append([k, f"{v:.6g}"])
    table = Table(rows, colWidths=[200, 200])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
        ("BOX", (0,0), (-1,-1), 0.5, colors.black),
        ("INNERGRID", (0,0), (-1,-1), 0.25, colors.grey),
    ]))
    flow.append(table)
    flow.append(Spacer(1, 12))

    flow.append(Paragraph(f"PHQ-9 Score: {results.get('phq_score', '—')} / 27", styles["Normal"]))
    flow.append(Paragraph(f"AD8 Score: {results.get('ad8_score', '—')} / 8", styles["Normal"]))
    flow.append(Spacer(1, 12))

    if band_png:
        flow.append(RLImage(io.BytesIO(band_png), width=420, height=200))
        flow.append(Spacer(1, 12))

    flow.append(Paragraph(TEXTS[lang]["note"], styles["Italic"]))
    doc.build(flow)
    buf.seek(0)
    return buf.getvalue()

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="NeuroEarly Pro", layout="centered")
st.sidebar.title("🌐 Language / اللغة")
lang = st.sidebar.radio("Choose / اختر", ("en", "ar"))

t = TEXTS[lang]
st.title(t["title"])
st.write(t["subtitle"])

# 1) Upload
st.header(t["upload"])
uploaded = st.file_uploader("", type=["edf"])
bands = {}
band_png = None
raw = None
apply_ica_checkbox = st.checkbox("Apply ICA cleaning (experimental, can be slow)", value=False)

if uploaded:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmp:
        tmp.write(uploaded.read())
        tmp.flush()
        tmp_path = tmp.name
    try:
        raw = mne.io.read_raw_edf(tmp_path, preload=True, verbose=False)
        raw = preprocess_eeg(raw, apply_ica=apply_ica_checkbox, st_logger=st)
        bands = compute_band_powers(raw)
        band_png = plot_bands(bands)
        st.image(band_png, use_column_width=True)
        st.success(f"Sampling rate: {raw.info.get('sfreq', '—')} Hz | Channels: {raw.info.get('nchan', '—')}")
    except Exception as e:
        st.error(f"Error reading / processing EDF: {e}")

# 2) PHQ-9
st.header(t["phq9"])
phq_answers = []
for i, q in enumerate(t["phq9_questions"], start=1):
    ans = st.selectbox(q, t["phq9_options"], key=f"phq{i}")
    # parse leading number before '='
    try:
        val = int(ans.split("=")[0].strip())
    except Exception:
        val = 0
    phq_answers.append(val)

# follow-ups for appetite (question index 5 -> zero-based 4) and movement (index 8 -> 7)
appetite_direction = None
movement_direction = None
try:
    if len(phq_answers) >= 5 and phq_answers[4] > 0:
        appetite_direction = st.selectbox(
            "If present, specify appetite change:",
            t["appetite_follow"], key="app_dir"
        )
    if len(phq_answers) >= 8 and phq_answers[7] > 0:
        movement_direction = st.selectbox(
            "If present, specify the motor change:",
            t["movement_follow"], key="move_dir"
        )
except Exception as e:
    st.warning(f"Follow-up input issue: {e}")

phq_score = sum(phq_answers)
# risk label (standard PHQ-9 buckets)
if phq_score < 5:
    phq_risk = "Minimal"
elif phq_score < 10:
    phq_risk = "Mild"
elif phq_score < 15:
    phq_risk = "Moderate"
elif phq_score < 20:
    phq_risk = "Moderately severe"
else:
    phq_risk = "Severe"
st.write(f"PHQ-9 Score: **{phq_score} / 27** → **{phq_risk}**")

# 3) AD8
st.header(t["ad8"])
ad8_answers = []
for i, q in enumerate(t["ad8_questions"], start=1):
    ans = st.selectbox(q, t["ad8_options"], index=0, key=f"ad8{i}")
    ad8_answers.append(1 if ans == t["ad8_options"][1] else 0)
ad8_score = sum(ad8_answers)
ad8_risk = "Possible concern (≥2)" if ad8_score >= 2 else "Low"
st.write(f"AD8 Score: **{ad8_score} / 8** → **{ad8_risk}**")

# 4) Generate + Download
st.header(t["report"])
if st.button("Generate"):
    results = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "bands": bands,
        "phq_score": phq_score,
        "phq_answers": phq_answers,
        "appetite_direction": appetite_direction,
        "movement_direction": movement_direction,
        "ad8_score": ad8_score,
        "ad8_answers": ad8_answers
    }

    # JSON
    json_bytes = io.BytesIO(json.dumps(results, indent=2, ensure_ascii=False).encode("utf-8"))
    st.download_button(t["download_json"], data=json_bytes, file_name="neuroearly_report.json", mime="application/json")

    # PDF
    try:
        pdf_bytes = build_pdf(results, lang=lang, band_png=band_png)
        st.download_button(t["download_pdf"], data=pdf_bytes, file_name="neuroearly_report.pdf", mime="application/pdf")
    except Exception as e:
        st.error(f"PDF generation failed: {e}")
        # fallback: offer JSON only

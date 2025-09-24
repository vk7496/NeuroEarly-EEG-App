import io
import os
import json
import tempfile
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import mne

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors

# Arabic support
import arabic_reshaper
from bidi.algorithm import get_display

AMIRI_PATH = "Amiri-Regular.ttf"
BANDS = {"Delta": (0.5, 4), "Theta": (4, 8), "Alpha": (8, 12), "Beta": (12, 30), "Gamma": (30, 45)}

if os.path.exists(AMIRI_PATH):
    if "Amiri" not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont("Amiri", AMIRI_PATH))

# ---------------------------
# Arabic reshape
# ---------------------------

def reshape_arabic(text: str) -> str:
    reshaped = arabic_reshaper.reshape(text)
    return get_display(reshaped)

# ---------------------------
# Texts
# ---------------------------
TEXTS = {
    "en": {
        "title": "🧠 NeuroEarly Pro",
        "subtitle": "Prototype for early Alzheimer’s & Depression risk screening using EEG and QEEG features.",
        "upload": "1) Upload EEG file (.edf)",
        "clean": "Apply ICA artifact removal (experimental, slow)",
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
            "Poor appetite or overeating",
            "Feeling bad about yourself or feeling like a failure",
            "Trouble concentrating (e.g., reading, watching TV)",
            "Moving or speaking slowly, OR being fidgety/restless",
            "Thoughts of being better off dead or self-harm"
        ],
        "phq9_options": ["0 = Not at all", "1 = Several days", "2 = More than half the days", "3 = Nearly every day"],
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
        "title": "🧠 نيوروإيرلي برو",
        "subtitle": "نموذج أولي للفحص المبكر لمخاطر الزهايمر والاكتئاب باستخدام EEG و QEEG.",
        "upload": "١) تحميل ملف تخطيط الدماغ (.edf)",
        "clean": "إزالة التشويش باستخدام ICA (تجريبي، بطيء)",
        "phq9": "٢) فحص الاكتئاب — PHQ-9",
        "ad8": "٣) الفحص المعرفي — AD8",
        "report": "٤) إنشاء التقرير",
        "download_json": "⬇️ تنزيل JSON",
        "download_pdf": "⬇️ تنزيل PDF",
        "note": "⚠️ هذا نموذج بحثي فقط — ليس أداة تشخيص سريري.",
        "phq9_questions": [
            "قلة الاهتمام أو المتعة في الأنشطة",
            "الشعور بالحزن أو الاكتئاب أو اليأس",
            "مشاكل في النوم أو النوم المفرط",
            "الشعور بالتعب أو قلة الطاقة",
            "فقدان الشهية أو الإفراط في الأكل",
            "الشعور بأنك شخص سيء أو فاشل",
            "صعوبة في التركيز (مثل القراءة أو مشاهدة التلفاز)",
            "الحركة أو الكلام ببطء شديد، أو فرط الحركة",
            "أفكار بأنك أفضل حالاً ميتاً أو أفكار لإيذاء النفس"
        ],
        "phq9_options": ["0 = أبداً", "1 = عدة أيام", "2 = أكثر من نصف الأيام", "3 = كل يوم تقريباً"],
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
# EEG / QEEG helpers
# ---------------------------

def preprocess_raw(raw, l_freq=1.0, h_freq=45.0, notch_freqs=(50, 60)):
    raw = raw.copy()
    raw.load_data()
    try:
        raw.filter(l_freq=l_freq, h_freq=h_freq, fir_design='firwin', verbose=False)
    except Exception:
        pass
    try:
        raw.notch_filter(freqs=notch_freqs, verbose=False)
    except Exception:
        pass
    try:
        raw.set_eeg_reference('average', verbose=False)
    except Exception:
        pass
    return raw


def compute_band_powers_per_channel(raw, bands=BANDS, n_fft=2048):
    psds, freqs = mne.time_frequency.psd_welch(raw, fmin=0.5, fmax=45, n_fft=n_fft, verbose=False)
    band_abs = {}
    band_per_channel = {}
    total_power_per_channel = np.trapz(psds, freqs, axis=1)
    for name, (fmin, fmax) in bands.items():
        mask = (freqs >= fmin) & (freqs < fmax)
        band_power_ch = np.trapz(psds[:, mask], freqs[mask], axis=1)
        band_per_channel[name] = band_power_ch
        band_abs[name] = float(np.mean(band_power_ch))
    total_mean = sum(band_abs.values()) + 1e-12
    band_rel = {k: float(v / total_mean) for k, v in band_abs.items()}
    return {
        'abs_mean': band_abs,
        'rel_mean': band_rel,
        'per_channel': band_per_channel,
        'total_power_per_channel': total_power_per_channel,
        'freqs': freqs
    }


def compute_qeeg_features(raw):
    raw = preprocess_raw(raw)
    bp = compute_band_powers_per_channel(raw)
    feats = {}
    for b, v in bp['abs_mean'].items():
        feats[f"{b}_abs_mean"] = v
    for b, v in bp['rel_mean'].items():
        feats[f"{b}_rel_mean"] = v
    if 'Theta' in bp['abs_mean'] and 'Beta' in bp['abs_mean']:
        feats['Theta_Beta_ratio'] = bp['abs_mean']['Theta'] / (bp['abs_mean']['Beta'] + 1e-12)
    if 'Theta' in bp['abs_mean'] and 'Alpha' in bp['abs_mean']:
        feats['Theta_Alpha_ratio'] = bp['abs_mean']['Theta'] / (bp['abs_mean']['Alpha'] + 1e-12)
    def idx(ch_name):
        try:
            return raw.ch_names.index(ch_name)
        except ValueError:
            return None
    pairs = [('F3', 'F4'), ('Fp1', 'Fp2'), ('F7', 'F8')]
    for left, right in pairs:
        i = idx(left)
        j = idx(right)
        if i is not None and j is not None:
            alpha_power = bp['per_channel'].get('Alpha')
            if alpha_power is not None:
                asym = float(np.log(alpha_power[i] + 1e-12) - np.log(alpha_power[j] + 1e-12))
                feats[f'alpha_asym_{left}_{right}'] = asym
    return feats, bp


def plot_band_bar(band_dict):
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(list(band_dict.keys()), list(band_dict.values()))
    ax.set_title('EEG Band Powers (mean across channels)')
    ax.set_ylabel('Power (a.u.)')
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    return buf.getvalue()

# ---------------------------
# Streamlit App UI
# ---------------------------

st.sidebar.title("🌐 Language / اللغة")
lang = st.sidebar.radio("Choose / اختر", ["en", "ar"])
t = TEXTS[lang]

st.title(t["title"])
st.write(t["subtitle"])

# 1) EEG Upload
st.header(t["upload"])
uploaded = st.file_uploader("EDF", type=["edf"])
apply_ica = st.checkbox(t["clean"])
bands = {}
qeeg_features = {}
if uploaded:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmp:
        tmp.write(uploaded.read())
        raw = mne.io.read_raw_edf(tmp.name, preload=True, verbose=False)
        if apply_ica:
            ica = mne.preprocessing.ICA(n_components=10, random_state=42)
            ica.fit(raw)
            raw = ica.apply(raw)
        qeeg_features, bp = compute_qeeg_features(raw)
        bands = bp['abs_mean']
        st.image(plot_band_bar(bands))

# 2) PHQ-9
st.header(t["phq9"])
phq_answers = []
for i, q in enumerate(t["phq9_questions"], 1):
    ans = st.selectbox(q, t["phq9_options"], key=f"phq{i}")
    phq_answers.append(int(ans.split("=")[0].strip()) if "=" in ans else t["phq9_options"].index(ans))
phq_score = sum(phq_answers)
phq_risk = ("Minimal" if phq_score<5 else "Mild" if phq_score<10 else "Moderate" if phq_score<15 else "Moderately severe" if phq_score<20 else "Severe")

# 3) AD8
st.header(t["ad8"])
ad8_answers = []
for i, q in enumerate(t["ad8_questions"], 1):
    ans = st.selectbox(q, t["ad8_options"], key=f"ad8{i}")
    ad8_answers.append(1 if ans==t["ad8_options"][1] else 0)
ad8_score = sum(ad8_answers)
ad8_risk = "Low" if ad8_score<2 else "Possible concern"

# 4) Report
st.header(t["report"])
if st.button("Generate"):
    results = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "EEG": {"bands": bands, "relative": bp['rel_mean'] if bands else {}},
        "QEEG": qeeg_features,
        "Depression": {"score": phq_score, "risk": phq_risk},
        "Alzheimer": {"score": ad8_score, "risk": ad8_risk}
    }
    json_bytes = io.BytesIO(json.dumps(results, indent=2, ensure_ascii=False).encode())
    st.download_button(t["download_json"], json_bytes, file_name="report.json")

    # PDF generation
    def build_pdf():
        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4)
        styles = getSampleStyleSheet()
        if lang=='ar' and os.path.exists(AMIRI_PATH):
            for s in ['Normal','Title','Heading2','Italic']:
                styles[s].fontName='Amiri'
        flow = []
        L = lambda txt: reshape_arabic(txt) if lang=='ar' else txt
        flow.append(Paragraph(L(t['title']), styles['Title']))
        flow.append(Paragraph(L(t['subtitle']), styles['Normal']))
        flow.append(Spacer(1,12))
        flow.append(Paragraph(L(f"Generated: {results['timestamp']}"), styles['Normal']))
        flow.append(Spacer(1,12))

        flow.append(Paragraph(L("EEG Band Powers" if lang=='en' else "قوى موجات الدماغ"), styles['Heading2']))
        rows=[["Band","Absolute","Relative"]]
        for k,v in results['EEG']['bands'].items():
            rel = results['EEG']['relative'].get(k,0)
            rows.append([k,f"{v:.4f}",f"{rel:.4f}"])
        table = Table(rows, colWidths=[150,150,150])
        table.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.lightgrey),("GRID",(0,0),(-1,-1),0.25,colors.black)]))
        flow.append(table)
        flow.append(Spacer(1,12))

        flow.append(Paragraph(L("Depression PHQ-9" if lang=='en' else "فحص الاكتئاب PHQ-9"), styles['Heading2']))
        flow.append(Paragraph(L(f"Score: {phq_score} → {phq_risk}"), styles['Normal']))
        flow.append(Spacer(1,12))

        flow.append(Paragraph(L("Cognitive AD8" if lang=='en' else "الفحص المعرفي AD8"), styles['Heading2']))
        flow.append(Paragraph(L(f"Score: {ad8_score} → {ad8_risk}"), styles['Normal']))
        flow.append(Spacer(1,12))

        flow.append(Paragraph(L("Recommendation:" if lang=='en' else "التوصية:"), styles['Heading2']))
        rec = "Follow up with neurologist and psychiatrist." if lang=='en' else "ينصح بمتابعة مع طبيب الأعصاب وطبيب نفسي."
        flow.append(Paragraph(L(rec), styles['Normal']))
        flow.append(Spacer(1,12))

        flow.append(Paragraph(L(t['note']), styles['Italic']))
        if bands:
            buf_img = plot_band_bar(results['EEG']['bands'])
            flow.append(RLImage(io.BytesIO(buf_img), width=400, height=200))
        doc.build(flow)
        buf.seek(0)
        return buf.getvalue()

    pdf_bytes = build_pdf()
    st.download_button(t["download_pdf"], pdf_bytes, file_name="report.pdf")

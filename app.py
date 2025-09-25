import io
import os
import json
import tempfile
from datetime import datetime

import numpy as np
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
            "Thoughts of being better off dead\nor self-harm"
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
            "أفكار بأنك أفضل حالاً ميتاً\nأو أفكار لإيذاء النفس"
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


def compute_band_powers_per_channel(raw, bands=BANDS):
    psd = raw.compute_psd(fmin=0.5, fmax=45, method="welch", verbose=False)
    psds, freqs = psd.get_data(return_freqs=True)
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
            try:
                import sklearn
                ica = mne.preprocessing.ICA(n_components=10, random_state=42)
                ica.fit(raw)
                raw = ica.apply(raw)
            except ImportError:
                st.warning("ICA requires scikit-learn. Skipping ICA step.")
        qeeg_features, bp = compute_qeeg_features(raw)
        bands = bp['abs_mean']
        st.image(plot_band_bar(bands))

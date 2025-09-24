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

# ---------------------------
# Config / constants
# ---------------------------
AMIRI_PATH = "Amiri-Regular.ttf"
BANDS = {"Delta": (0.5, 4), "Theta": (4, 8), "Alpha": (8, 12), "Beta": (12, 30), "Gamma": (30, 45)}

# register font if available
if os.path.exists(AMIRI_PATH):
    if "Amiri" not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont("Amiri", AMIRI_PATH))

# ---------------------------
# Utility (Arabic reshaping)
# ---------------------------
def reshape_arabic(text: str) -> str:
    reshaped = arabic_reshaper.reshape(text)
    return get_display(reshaped)

# ---------------------------
# Text dictionary
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
    }
}

# ---------------------------
# EEG / QEEG helpers
# ---------------------------

def preprocess_raw(raw, l_freq=1.0, h_freq=45.0, notch_freqs=(50, 60), ref_type='average'):
    raw = raw.copy()
    raw.load_data()
    try:
        raw.filter(l_freq=l_freq, h_freq=h_freq, fir_design='firwin', verbose=False)
    except Exception:
        # some raw objects need different filter handling; ignore for robustness
        pass
    try:
        raw.notch_filter(freqs=notch_freqs, verbose=False)
    except Exception:
        pass
    if ref_type == 'average':
        try:
            raw.set_eeg_reference('average', verbose=False)
        except Exception:
            pass
    return raw


def compute_band_powers_per_channel(raw, bands=BANDS, n_fft=2048):
    """Return dict of absolute power (mean across channels) and per-channel arrays."""
    psds, freqs = mne.time_frequency.psd_welch(raw, fmin=0.5, fmax=45, n_fft=n_fft, verbose=False)
    # psds: shape (n_channels, n_freqs)
    band_abs = {}
    band_per_channel = {}
    total_power_per_channel = np.trapz(psds, freqs, axis=1)
    for name, (fmin, fmax) in bands.items():
        mask = (freqs >= fmin) & (freqs < fmax)
        band_power_ch = np.trapz(psds[:, mask], freqs[mask], axis=1)  # per channel
        band_per_channel[name] = band_power_ch
        band_abs[name] = float(np.mean(band_power_ch))
    # compute relative powers (mean across channels)
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
    # absolute + relative
    for b, v in bp['abs_mean'].items():
        feats[f"{b}_abs_mean"] = v
    for b, v in bp['rel_mean'].items():
        feats[f"{b}_rel_mean"] = v
    # ratios
    if 'Theta' in bp['abs_mean'] and 'Beta' in bp['abs_mean']:
        feats['Theta_Beta_ratio'] = bp['abs_mean']['Theta'] / (bp['abs_mean']['Beta'] + 1e-12)
    if 'Theta' in bp['abs_mean'] and 'Alpha' in bp['abs_mean']:
        feats['Theta_Alpha_ratio'] = bp['abs_mean']['Theta'] / (bp['abs_mean']['Alpha'] + 1e-12)
    # frontal asymmetry - try common labels
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
# PDF Generator (enhanced)
# ---------------------------

def build_pdf_bytes(results: dict, lang='en', band_png=None, logo_path=None):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=36,leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()

    if lang == 'ar' and os.path.exists(AMIRI_PATH):
        for s in ['Normal', 'Title', 'Heading2', 'Italic']:
            styles[s].fontName = 'Amiri'

    flow = []
    t = TEXTS[lang]

    def L(txt):
        return reshape_arabic(txt) if lang == 'ar' else txt

    # Header
    flow.append(Paragraph(L(t['title']), styles['Title']))
    flow.append(Paragraph(L(t['subtitle']), styles['Normal']))
    flow.append(Spacer(1, 12))

    # Meta
    flow.append(Paragraph(L(f"Generated: {results['timestamp']}"), styles['Normal']))
    flow.append(Spacer(1, 12))

    # EEG Band table
    flow.append(Paragraph(L("EEG Band Powers (mean across channels):" if lang=='en' else "قوى موجات الدماغ (متوسط عبر کانال‌ها):"), styles['Heading2']))
    rows = [["Band", "Absolute (a.u.)", "Relative"]]
    for k, v in results['EEG']['bands'].items():
        rel = results['EEG']['relative'].get(k, 0)
        rows.append([k, f"{v:.4f}", f"{rel:.4f}"])
    table = Table(rows, colWidths=[120, 120, 120])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
        ("GRID", (0,0), (-1,-1), 0.25, colors.black),
        ("ALIGN", (1,1), (-1,-1), 'RIGHT')
    ]))
    flow.append(table)
    flow.append(Spacer(1, 12))

    # QEEG features table (ratios, asymmetry)
    flow.append(Paragraph(L("QEEG Features:" if lang=='en' else "ویژگی‌های QEEG:"), styles['Heading2']))
    qrows = [["Feature", "Value"]]
    for k, v in results['QEEG'].items():
        qrows.append([k, f"{v:.4f}" if isinstance(v, (int, float)) else str(v)])
    qtable = Table(qrows, colWidths=[240, 120])
    qtable.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
        ("GRID", (0,0), (-1,-1), 0.25, colors.black),
        ("ALIGN", (1,1), (-1,-1), 'RIGHT')
    ]))
    flow.append(qtable)
    flow.append(Spacer(1, 12))

    # PHQ & AD8
    flow.append(Paragraph(L(f"PHQ-9 Score: {results['Depression']['score']} → {results['Depression']['risk']}"), styles['Normal']))
    flow.append(Paragraph(L(f"AD8 Score: {results['Alzheimer']['score']} → {results['Alzheimer']['risk']}"), styles['Normal']))
    flow.append(Spacer(1, 12))

    # Band image
    if band_png:
        flow.append(RLImage(io.BytesIO(band_png), width=450, height=180))
        flow.append(Spacer(1, 12))

    # Recommendation / interpretation
    rec = "Follow up with a neurologist and psychiatrist for combined evaluation."
    if lang == 'ar':
        rec = "ينصح بمتابعة مع طبيب الأعصاب وطبيب نفسي لتقييم مشترك."
    flow.append(Paragraph(L("Recommendation:" if lang=='en' else "التوصية:"), styles['Heading2']))
    flow.append(Paragraph(L(rec), styles['Normal']))
    flow.append(Spacer(1, 12))

    flow.append(Paragraph(L(t['note']), styles['Italic']))
    doc.build(flow)
    buf.seek(0)
    return buf.getvalue()

# ---------------------------
# Streamlit App
# ---------------------------
st.set_page_config(page_title="EEG + QEEG Depression Detection", layout="wide")
st.sidebar.title("🌐 Language / اللغة")
lang = st.sidebar.radio("Choose / اختر", ["en", "ar"]) 

t = TEXTS[lang]
st.title(t['title'])
st.write(t['subtitle'])

# Layout: tabs
tabs = st.tabs([t['upload'], t['phq9'], t['ad8'], t['report']])

# Shared state
raw = None
bands = {}
qeeg_feats = {}

# 1) Upload & EEG processing
with tabs[0]:
    st.header(t['upload'])
    uploaded = st.file_uploader("EDF", type=["edf"] )
    apply_ica = st.checkbox(t['clean'])
    if uploaded:
        with st.spinner("Reading EDF and preprocessing..."):
            try:
                # write to temp file for mne
                with tempfile.NamedTemporaryFile(delete=False, suffix='.edf') as tmp:
                    tmp.write(uploaded.read())
                    tmp.flush()
                    raw = mne.io.read_raw_edf(tmp.name, preload=True, verbose=False)
                if apply_ica:
                    with st.spinner("Running ICA (may be slow)..."):
                        ica = mne.preprocessing.ICA(n_components=15, random_state=42, verbose=False)
                        ica.fit(raw)
                        raw = ica.apply(raw)
                # compute band powers
                qeeg_feats, bp = compute_qeeg_features(raw)
                bands = bp['abs_mean']
                band_png = plot_band_bar(bands)
                st.success('EEG processed ✅')
                # show a few plots
                st.subheader('EEG Band Powers')
                st.image(band_png)
                st.subheader('Raw EEG (first channel sample)')
                chan0 = raw.get_data(picks=[0]).ravel()
                fig, ax = plt.subplots(figsize=(10, 2))
                ax.plot(chan0[: int(min(len(chan0), raw.info['sfreq'] * 10))])
                ax.set_title(f"Channel: {raw.ch_names[0]}")
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Error reading EDF: {e}")

# 2) PHQ-9
with tabs[1]:
    st.header(t['phq9'])
    phq_answers = []
    phq_qs = [
        "Little interest or pleasure in doing things",
        "Feeling down, depressed, or hopeless",
        "Trouble falling or staying asleep, or sleeping too much",
        "Feeling tired or having little energy",
        "Poor appetite or overeating",
        "Feeling bad about yourself or feeling like a failure",
        "Trouble concentrating (e.g., reading, watching TV)",
        "Moving or speaking slowly, OR being fidgety/restless",
        "Thoughts of being better off dead or self-harm"
    ]
    phq_opts = ["0 = Not at all", "1 = Several days", "2 = More than half the days", "3 = Nearly every day"]
    for i, q in enumerate(phq_qs, 1):
        ans = st.selectbox(q, phq_opts, key=f"phq{i}")
        phq_answers.append(int(ans.split('=')[0].strip()))
    phq_score = sum(phq_answers)
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
    st.write(f"PHQ-9 Score: **{phq_score}** → {phq_risk}")

# 3) AD8
with tabs[2]:
    st.header(t['ad8'])
    ad8_qs = [
        "Problems with judgment (e.g., poor financial decisions)",
        "Reduced interest in hobbies/activities",
        "Repeats questions or stories",
        "Trouble using a tool or gadget",
        "Forgets the correct month or year",
        "Difficulty managing finances (e.g., paying bills)",
        "Trouble remembering appointments",
        "Everyday thinking is getting worse"
    ]
    ad8_opts = ["No", "Yes"]
    ad8_answers = []
    for i, q in enumerate(ad8_qs, 1):
        ans = st.selectbox(q, ad8_opts, key=f"ad8{i}")
        ad8_answers.append(1 if ans == "Yes" else 0)
    ad8_score = sum(ad8_answers)
    ad8_risk = "Low" if ad8_score < 2 else "Possible concern"
    st.write(f"AD8 Score: **{ad8_score}** → {ad8_risk}")

# 4) Report
with tabs[3]:
    st.header(t['report'])
    if st.button("Generate"):
        if raw is None:
            st.error("Please upload an EDF file first.")
        else:
            results = {
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'EEG': {'bands': bands, 'relative': {k: v for k, v in compute_band_powers_per_channel(raw)['rel_mean'].items()}},
                'QEEG': qeeg_feats,
                'Depression': {'score': phq_score, 'risk': phq_risk},
                'Alzheimer': {'score': ad8_score, 'risk': ad8_risk}
            }
            # JSON download
            json_bytes = io.BytesIO(json.dumps(results, indent=2, ensure_ascii=False).encode())
            st.download_button(t['download_json'], json_bytes, file_name='report.json')

            # PDF
            band_png = plot_band_bar(bands) if bands else None
            pdf_bytes = build_pdf_bytes(results, lang=lang, band_png=band_png)
            st.download_button(t['download_pdf'], pdf_bytes, file_name='report.pdf')

    st.markdown("---")
    st.info(t['note'])

# Footer: dependencies hints
with st.expander("🛠️ Installation & Notes"):
    st.write("Make sure dependencies are installed in your environment:")
    st.code("pip install mne numpy pandas matplotlib streamlit reportlab arabic-reshaper python-bidi")
    st.write("If App is slow: avoid enabling ICA or lower n_components; PSD on long recordings may be slow.")

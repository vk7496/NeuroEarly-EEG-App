# app.py — Complete EEG + QEEG clinical helper
import io
import os
import json
import tempfile
from datetime import datetime
from typing import List, Tuple, Dict

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

# Arabic support (optional)
try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    HAS_ARABIC_TOOLS = True
except Exception:
    HAS_ARABIC_TOOLS = False

# ---------------------------
# Config
# ---------------------------
AMIRI_PATH = "Amiri-Regular.ttf"  # optional: put this font file next to app.py to improve Arabic PDF rendering
BANDS = {"Delta": (0.5, 4), "Theta": (4, 8), "Alpha": (8, 12), "Beta": (12, 30), "Gamma": (30, 45)}
DEFAULT_NOTCH = [50, 100]  # customize to 60/120 if needed for your country

# register Amiri font if present
if os.path.exists(AMIRI_PATH):
    try:
        if "Amiri" not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont("Amiri", AMIRI_PATH))
    except Exception:
        pass

# ---------------------------
# Helpers
# ---------------------------
def reshape_arabic(text: str) -> str:
    if HAS_ARABIC_TOOLS:
        reshaped = arabic_reshaper.reshape(text)
        return get_display(reshaped)
    return text

# Small util to format floats safely
def fmt(x):
    try:
        return f"{float(x):.4f}"
    except Exception:
        return str(x)

# ---------------------------
# Texts (EN / AR short)
# ---------------------------
TEXTS = {
    "en": {
        "title": "🧠 NeuroEarly Pro — Clinical Helper",
        "subtitle": "EEG + QEEG assistant for supporting clinical screening (Depression / Cognitive).",
        "upload": "1) Upload EEG file(s) (.edf) — you can upload multiple",
        "clean": "Apply ICA artifact removal (requires scikit-learn)",
        "compute_connectivity": "Compute Connectivity (slow)",
        "phq9": "2) Depression Screening — PHQ-9",
        "ad8": "3) Cognitive Screening — AD8",
        "report": "4) Generate Report (JSON / PDF)",
        "download_json": "⬇️ Download JSON",
        "download_pdf": "⬇️ Download PDF",
        "download_csv": "⬇️ Download CSV",
        "note": "⚠️ Research / decision-support tool only — not a definitive clinical diagnosis.",
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
        "title": "🧠 نيوروإيرلي برو — مساعد سريري",
        "subtitle": "کمک‌ابزار EEG و QEEG برای پشتیبانی از غربالگری بالینی (اكتئاب / شناختی).",
        "upload": "١) بارگذاری یک یا چند فایل EEG (.edf)",
        "clean": "إزالة التشويش باستخدام ICA (يحتاج scikit-learn)",
        "compute_connectivity": "حساب الاتصالات (بطيء)",
        "phq9": "٢) فحص الاكتئاب — PHQ-9",
        "ad8": "٣) الفحص المعرفي — AD8",
        "report": "٤) إنشاء التقرير (JSON / PDF)",
        "download_json": "⬇️ تنزيل JSON",
        "download_pdf": "⬇️ تنزيل PDF",
        "download_csv": "⬇️ تنزيل CSV",
        "note": "⚠️ أداة بحثية / مساعدة فقط — ليست تشخيصًا قاطعًا.",
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
# EEG preprocessing & QEEG
# ---------------------------
def preprocess_raw(raw: mne.io.BaseRaw, l_freq=1.0, h_freq=45.0, notch_freqs=DEFAULT_NOTCH) -> mne.io.BaseRaw:
    """Bandpass + notch + average reference (robust to errors)."""
    raw = raw.copy()
    try:
        raw.load_data()
    except Exception:
        pass
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

def compute_band_powers_per_channel(raw: mne.io.BaseRaw, bands=BANDS) -> dict:
    """Compute PSD using Raw.compute_psd (compatible with MNE modern versions)."""
    psd = raw.compute_psd(fmin=0.5, fmax=45, method="welch", verbose=False)
    psds, freqs = psd.get_data(return_freqs=True)  # shape: (n_channels, n_freqs)
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

def compute_qeeg_features(raw: mne.io.BaseRaw) -> Tuple[dict, dict]:
    """Return features dict and band power dict (bp)."""
    raw = preprocess_raw(raw)
    bp = compute_band_powers_per_channel(raw)
    feats = {}
    # abs & rel
    for b, v in bp['abs_mean'].items():
        feats[f"{b}_abs_mean"] = v
    for b, v in bp['rel_mean'].items():
        feats[f"{b}_rel_mean"] = v
    # ratios
    if 'Theta' in bp['abs_mean'] and 'Beta' in bp['abs_mean']:
        feats['Theta_Beta_ratio'] = bp['abs_mean']['Theta'] / (bp['abs_mean']['Beta'] + 1e-12)
    if 'Theta' in bp['abs_mean'] and 'Alpha' in bp['abs_mean']:
        feats['Theta_Alpha_ratio'] = bp['abs_mean']['Theta'] / (bp['abs_mean']['Alpha'] + 1e-12)
    # frontal asymmetry
    def idx(ch_name):
        try:
            return raw.ch_names.index(ch_name)
        except Exception:
            return None
    for left, right in [('F3','F4'), ('Fp1','Fp2'), ('F7','F8')]:
        i = idx(left); j = idx(right)
        if i is not None and j is not None:
            alpha_power = bp['per_channel'].get('Alpha')
            if alpha_power is not None:
                feats[f'alpha_asym_{left}_{right}'] = float(np.log(alpha_power[i] + 1e-12) - np.log(alpha_power[j] + 1e-12))
    return feats, bp

def compute_connectivity_optional(raw: mne.io.BaseRaw, fmin=4, fmax=30, method='coh') -> dict:
    """Compute a simple global connectivity metric (mean coherence) if user asks.
       This is optional and may be slow for long recordings."""
    try:
        from mne.connectivity import spectral_connectivity
    except Exception:
        return {'connectivity_error': 'mne.connectivity not available'}
    try:
        # spectral_connectivity expects epochs or Raw with mode='multitaper'/'fourier'
        # We'll compute on raw (may be slow); aggregate mean connectivity
        con, freqs, times, n_epochs, n_tapers = spectral_connectivity(
            raw, method=method, mode='fourier', sfreq=raw.info['sfreq'],
            fmin=fmin, fmax=fmax, faverage=True, tmin=0.0, tmax=None,
            mt_adaptive=False, n_jobs=1, verbose=False
        )
        # con shape: (n_connections, n_freqs) — take mean
        mean_con = float(np.nanmean(con))
        return {'connectivity_mean': mean_con}
    except Exception as e:
        return {'connectivity_error': str(e)}

# ---------------------------
# Plot helpers
# ---------------------------
def plot_band_bar(band_dict: dict) -> bytes:
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
# Smart textual interpretation rules (heuristics)
# ---------------------------
def interpret_qeeg(bp: dict, qeeg_feats: dict, connectivity: dict=None) -> List[str]:
    """Produce short, cautious interpretation lines for physician to review."""
    notes = []
    # example heuristics:
    # 1) Alpha asymmetry left > right indicates left hypoactivity -> associated with depression
    for k, v in qeeg_feats.items():
        if k.startswith('alpha_asym_'):
            if v > 0.2:  # threshold heuristic
                notes.append(f"Alpha asymmetry {k.split('_',2)[2]}: left>right ({fmt(v)}). This pattern can be associated with reduced left frontal activity — reported in some depression studies.")
            elif v < -0.2:
                notes.append(f"Alpha asymmetry {k.split('_',2)[2]}: right>left ({fmt(v)}).")
    # 2) Theta increase frontal -> cognitive concern
    theta = qeeg_feats.get('Theta_abs_mean')
    alpha = qeeg_feats.get('Alpha_abs_mean')
    if theta is not None and alpha is not None:
        ratio = theta / (alpha + 1e-12)
        if ratio > 1.2:
            notes.append(f"Elevated Theta/Alpha ratio ({fmt(ratio)}). This may reflect increased slow-wave activity which can appear in cognitive impairment contexts (consider correlation with AD8 and clinical exam).")
    # 3) Theta/Beta ratio high -> attention / frontal dysfunction (heuristic)
    tb = qeeg_feats.get('Theta_Beta_ratio')
    if tb is not None:
        if tb > 2.5:
            notes.append(f"High Theta/Beta ratio ({fmt(tb)}). Observed in attention/frontal regulation abnormalities; interpret with clinical context.")
    # 4) connectivity
    if connectivity:
        if 'connectivity_mean' in connectivity:
            val = connectivity['connectivity_mean']
            if val < 0.15:
                notes.append(f"Global connectivity (mean coherence) is low ({fmt(val)}). Reduced coherence between regions has been described in mood and cognitive disorders.")
            elif val > 0.4:
                notes.append(f"Global connectivity (mean coherence) is high ({fmt(val)}).")
        elif 'connectivity_error' in connectivity:
            notes.append(f"Connectivity not computed: {connectivity['connectivity_error']}")
    if not notes:
        notes.append("No striking QEEG-specific abnormalities observed by heuristics.")
    return notes

# ---------------------------
# PDF builder
# ---------------------------
def build_pdf_bytes(results: dict, patient_info: dict, lab_results: dict, meds_list: List[str], lang='en',
                    band_pngs: Dict[str, bytes]=None, interpretations: List[str]=None) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=36,leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    if lang == 'ar' and os.path.exists(AMIRI_PATH):
        for s in ['Normal','Title','Heading2','Italic']:
            styles[s].fontName = 'Amiri'
    flow = []
    t = TEXTS[lang]
    def L(txt): return reshape_arabic(txt) if lang=='ar' else txt

    # Header + patient
    flow.append(Paragraph(L(t['title']), styles['Title']))
    flow.append(Paragraph(L(t['subtitle']), styles['Normal']))
    flow.append(Spacer(1, 8))
    flow.append(Paragraph(L(f"Generated: {results['timestamp']}"), styles['Normal']))
    flow.append(Spacer(1, 8))

    # Patient info (optional)
    flow.append(Paragraph(L("Patient information:" if lang=='en' else "معلومات المريض:"), styles['Heading2']))
    p_rows = []
    for k in ['name','id','gender','dob','age','phone','email','history']:
        if patient_info.get(k):
            p_rows.append([k.capitalize() if lang=='en' else k, str(patient_info.get(k))])
    if p_rows:
        pt_table = Table(p_rows, colWidths=[150, 300])
        pt_table.setStyle(TableStyle([("GRID",(0,0),(-1,-1),0.25,colors.black)]))
        flow.append(pt_table)
    else:
        flow.append(Paragraph(L("No patient info provided." if lang=='en' else "لا توجد معلومات للمريض."), styles['Normal']))
    flow.append(Spacer(1, 8))

    # Lab tests
    flow.append(Paragraph(L("Recent lab tests (if provided):" if lang=='en' else "نتائج الفحوصات (إن وُجدت):"), styles['Heading2']))
    if lab_results:
        lab_rows = [["Test","Value"]]
        for k,v in lab_results.items():
            lab_rows.append([k, str(v)])
        lab_table = Table(lab_rows, colWidths=[200,200])
        lab_table.setStyle(TableStyle([("GRID",(0,0),(-1,-1),0.25,colors.black),("BACKGROUND",(0,0),(-1,0),colors.lightgrey)]))
        flow.append(lab_table)
    else:
        flow.append(Paragraph(L("No lab data provided." if lang=='en' else "لا توجد بيانات مختبرية."), styles['Normal']))
    flow.append(Spacer(1,8))

    # Meds
    flow.append(Paragraph(L("Current medications (if provided):" if lang=='en' else "الأدوية الحالية (إن وُجدت):"), styles['Heading2']))
    if meds_list:
        for m in meds_list:
            flow.append(Paragraph(L(m), styles['Normal']))
    else:
        flow.append(Paragraph(L("No medication data provided." if lang=='en' else "لا توجد أدوية مسجلة."), styles['Normal']))
    flow.append(Spacer(1,12))

    # For each EEG file: bands + qeeg features + optional image
    for fname, eegblock in results['EEG_files'].items():
        flow.append(Paragraph(L(f"EEG file: {fname}"), styles['Heading2']))
        # bands table
        b_rows = [["Band","Absolute (a.u.)","Relative"]]
        for k, v in eegblock.get('bands', {}).items():
            rel = eegblock.get('relative', {}).get(k, 0)
            b_rows.append([k, f"{v:.4f}", f"{rel:.4f}"])
        btable = Table(b_rows, colWidths=[120,120,120])
        btable.setStyle(TableStyle([("GRID",(0,0),(-1,-1),0.25,colors.black),("BACKGROUND",(0,0),(-1,0),colors.lightgrey)]))
        flow.append(btable)
        flow.append(Spacer(1,6))

        # QEEG features
        flow.append(Paragraph(L("QEEG features:" if lang=='en' else "ویژگی‌های QEEG:"), styles['Normal']))
        q_rows = [["Feature","Value"]]
        for k,v in eegblock.get('QEEG', {}).items():
            q_rows.append([k, fmt(v) if isinstance(v,(int,float)) else str(v)])
        qtab = Table(q_rows, colWidths=[240,120])
        qtab.setStyle(TableStyle([("GRID",(0,0),(-1,-1),0.25,colors.black),("BACKGROUND",(0,0),(-1,0),colors.lightgrey)]))
        flow.append(qtab)
        flow.append(Spacer(1,6))

        # optional band image
        img = band_pngs.get(fname) if band_pngs else None
        if img:
            flow.append(RLImage(io.BytesIO(img), width=450, height=160))
            flow.append(Spacer(1,6))

        # connectivity
        conn = eegblock.get('connectivity', {})
        if conn:
            flow.append(Paragraph(L("Connectivity summary:" if lang=='en' else "ملخص الاتصالات:"), styles['Normal']))
            for kk, vv in conn.items():
                flow.append(Paragraph(L(f"{kk}: {fmt(vv)}"), styles['Normal']))
        flow.append(Spacer(1,12))

    # Interpretations
    flow.append(Paragraph(L("Interpretation (heuristic):" if lang=='en' else "تفسير (استنتاجي):"), styles['Heading2']))
    if interpretations:
        for line in interpretations:
            flow.append(Paragraph(L(line), styles['Normal']))
    else:
        flow.append(Paragraph(L("No automated interpretation." if lang=='en' else "لا توجد استنتاجات تلقائية."), styles['Normal']))
    flow.append(Spacer(1,12))

    # Recommendation block (structured, not prescriptive)
    flow.append(Paragraph(L("Structured recommendation for clinician:" if lang=='en' else "توصية منظمة للطبيب:"), styles['Heading2']))
    rec_lines = []
    # generic but actionable
    rec_lines.append("1) Correlate QEEG findings with PHQ-9/AD8 results and clinical interview.")
    rec_lines.append("2) If PHQ-9 suggests moderate/severe depression or QEEG shows left frontal alpha asymmetry, consider psychiatric referral for full evaluation and treatment planning (psychotherapy ± pharmacotherapy).")
    rec_lines.append("3) If AD8 score elevated or Theta increase seen, consider neurocognitive assessment and neuroimaging (MRI) as indicated.")
    rec_lines.append("4) Consider medication review (see medication list) for agents that may affect EEG.")
    rec_lines.append("5) If suicidal ideation present (PHQ item), arrange urgent psychiatric evaluation.")
    for r in rec_lines:
        flow.append(Paragraph(L(r if lang=='en' else r), styles['Normal']))
    flow.append(Spacer(1,12))

    flow.append(Paragraph(L(t['note']), styles['Italic']))
    doc.build(flow)
    buf.seek(0)
    return buf.getvalue()

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="EEG + QEEG Clinical Helper", layout="wide")
st.sidebar.title("🌐 Language / اللغة")
lang = st.sidebar.radio("Choose / اختر", ["en", "ar"])
t = TEXTS[lang]

st.title(t["title"])
st.write(t["subtitle"])

# Layout columns for patient form on the side
with st.expander("🔎 Optional: Patient information (fill if available)"):
    st.write("Provide patient data to include in the report (optional).")
    name = st.text_input("Full name")
    patient_id = st.text_input("Patient ID / Record #")
    gender = st.selectbox("Gender", ["", "Male", "Female", "Other"])
    dob = st.date_input("Date of birth", value=None)
    phone = st.text_input("Phone (optional)")
    email = st.text_input("Email (optional)")
    history = st.text_area("Relevant medical history (diabetes, hypertension, family history, etc.)", height=80)

patient_info = {
    "name": name,
    "id": patient_id,
    "gender": gender,
    "dob": dob.strftime("%Y-%m-%d") if dob else "",
    "age": int((datetime.now().date() - dob).days / 365) if dob else "",
    "phone": phone,
    "email": email,
    "history": history
}

# Lab tests form
with st.expander("🧪 Optional: Recent lab tests (enter values)"):
    st.write("Enter any relevant lab values — include units in value text.")
    lab_glucose = st.text_input("Glucose (e.g. 5.6 mmol/L)")
    lab_b12 = st.text_input("Vitamin B12 (e.g. 350 pg/mL)")
    lab_vitd = st.text_input("Vitamin D (e.g. 25 ng/mL)")
    lab_tsh = st.text_input("TSH (e.g. 2.1 uIU/mL)")
    lab_crp = st.text_input("CRP (e.g. 1.2 mg/L)")
lab_results = {}
if lab_glucose: lab_results['Glucose'] = lab_glucose
if lab_b12: lab_results['Vitamin B12'] = lab_b12
if lab_vitd: lab_results['Vitamin D'] = lab_vitd
if lab_tsh: lab_results['TSH'] = lab_tsh
if lab_crp: lab_results['CRP'] = lab_crp

# Medications
with st.expander("💊 Current medications (one per line)"):
    meds_text = st.text_area("List medications (e.g. Sertraline 50 mg once daily)", height=120)
meds_list = [m.strip() for m in meds_text.splitlines() if m.strip()]

# Tabs: Upload / PHQ / AD8 / Report
tab_upload, tab_phq, tab_ad8, tab_report = st.tabs([t["upload"], t["phq9"], t["ad8"], t["report"]])

# Shared variables
EEG_results = {"EEG_files": {}}  # will hold per-file results
band_pngs = {}

# Upload tab (multiple files)
with tab_upload:
    st.header(t["upload"])
    uploaded_files = st.file_uploader("EDF files", type=["edf"], accept_multiple_files=True)
    apply_ica = st.checkbox(t["clean"])
    compute_conn = st.checkbox(t["compute_connectivity"])
    selected_notch = st.multiselect("Notch frequencies (Hz) — choose depending on grid (usually 50 or 60)", [50,60,100,120], default=[50,100])
    if uploaded_files:
        for uf in uploaded_files:
            st.info(f"Processing: {uf.name}")
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmp:
                    tmp.write(uf.read())
                    tmp.flush()
                    tmp_name = tmp.name
                raw = mne.io.read_raw_edf(tmp_name, preload=True, verbose=False)
                # preprocess with chosen notch
                raw = preprocess_raw(raw, notch_freqs=selected_notch)
                # optional ICA (safeguard)
                if apply_ica:
                    try:
                        import sklearn  # ensures scikit-learn present
                        ica = mne.preprocessing.ICA(n_components=15, random_state=42, verbose=False)
                        ica.fit(raw)
                        raw = ica.apply(raw)
                    except ImportError:
                        st.warning("ICA requires scikit-learn. Skipping ICA step.")
                    except Exception as e:
                        st.warning(f"ICA failed or skipped: {e}")
                # compute qeeg
                qeeg_feats, bp = compute_qeeg_features(raw)
                # optional connectivity
                conn_res = {}
                if compute_conn:
                    with st.spinner("Computing connectivity (this can be slow)..."):
                        conn_res = compute_connectivity_optional(raw)
                # store results
                EEG_results['EEG_files'][uf.name] = {
                    'bands': bp['abs_mean'],
                    'relative': bp['rel_mean'],
                    'QEEG': qeeg_feats,
                    'connectivity': conn_res
                }
                # band png
                band_png = plot_band_bar(bp['abs_mean'])
                band_pngs[uf.name] = band_png
                st.image(band_png, caption=f"{uf.name} — band powers")
                st.success(f"{uf.name} processed.")
            except Exception as e:
                st.error(f"Error processing {uf.name}: {e}")

# PHQ-9 tab
with tab_phq:
    st.header(t["phq9"])
    phq_answers = []
    for i, q in enumerate(t["phq9_questions"], 1):
        ans = st.selectbox(q, t["phq9_options"], key=f"phq{i}")
        try:
            phq_answers.append(int(ans.split("=")[0].strip()))
        except Exception:
            phq_answers.append(t["phq9_options"].index(ans))
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

# AD8 tab
with tab_ad8:
    st.header(t["ad8"])
    ad8_answers = []
    for i, q in enumerate(t["ad8_questions"], 1):
        ans = st.selectbox(q, t["ad8_options"], key=f"ad8{i}")
        ad8_answers.append(1 if ans == t["ad8_options"][1] else 0)
    ad8_score = sum(ad8_answers)
    ad8_risk = "Low" if ad8_score < 2 else "Possible concern"
    st.write(f"AD8 Score: **{ad8_score}** → {ad8_risk}")

# Report tab
with tab_report:
    st.header(t["report"])
    if st.button("Generate"):
        # assemble results
        EEG_results['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        EEG_results['Depression'] = {"score": phq_score, "risk": phq_risk}
        EEG_results['Alzheimer'] = {"score": ad8_score, "risk": ad8_risk}
        # interpretations aggregated across files
        all_interpretations = []
        for fname, block in EEG_results['EEG_files'].items():
            interp = interpret_qeeg(block.get('bands', {}), block.get('QEEG', {}), block.get('connectivity', {}))
            all_interpretations.extend([f"{fname}: {s}" for s in interp])
        # build json
        json_bytes = io.BytesIO(json.dumps(EEG_results, indent=2, ensure_ascii=False).encode())
        st.download_button(t["download_json"], json_bytes, file_name="report.json")
        # CSV for QEEG features (each file a row)
        if EEG_results['EEG_files']:
            rows = []
            for fname, b in EEG_results['EEG_files'].items():
                row = {'file': fname}
                # add QEEG features
                for k, v in b.get('QEEG', {}).items():
                    row[k] = v
                rows.append(row)
            df_q = pd.DataFrame(rows)
            csv_bytes = df_q.to_csv(index=False).encode('utf-8')
            st.download_button(t["download_csv"], csv_bytes, file_name="qeeg_features.csv", mime="text/csv")
        # PDF
        try:
            pdf_bytes = build_pdf_bytes(EEG_results, patient_info, lab_results, meds_list, lang=lang, band_pngs=band_pngs, interpretations=all_interpretations)
            st.download_button(t["download_pdf"], pdf_bytes, file_name="report.pdf")
            st.success("Report (PDF) generated — ready to download.")
        except Exception as e:
            st.error(f"PDF generation failed: {e}")
    st.markdown("---")
    st.info(t["note"])

# Footer / notes
with st.expander("🛠️ Installation & Notes"):
    st.write("Make sure dependencies are installed or add them to requirements.txt (see below).")
    st.code("pip install mne numpy pandas matplotlib streamlit reportlab arabic-reshaper python-bidi scikit-learn")
    st.write("If App is slow: avoid enabling ICA or connectivity computation or reduce ICA n_components.")

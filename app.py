# app.py — NeuroEarly (multilingual, PHQ two-step, ICA, trend)
import io
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
from reportlab.lib import colors

# -------------------------
# Config & UI basics
# -------------------------
st.set_page_config(page_title="NeuroEarly — EEG Screening", layout="centered")
st.sidebar.title("NeuroEarly")
st.sidebar.markdown("Prototype — research demo. Not a medical diagnosis.")

# Language selection
lang = st.sidebar.radio("Language / اللغة", options=["English", "العربية"])

# Simple translation dict (extendable)
TEXT = {
    "English": {
        "title": "🧠 NeuroEarly — EEG + PHQ-9 + AD8 (Demo)",
        "subtitle": "Prototype for early screening: EEG bands + PHQ-9 (depression) + AD8 (cognition).",
        "upload": "Upload EDF files (single or multiple)",
        "upload_hint": "You can upload one file (single session) or multiple files (longitudinal trend).",
        "phq": "PHQ-9 (Depression) — answer frequency for past 2 weeks",
        "ad8": "AD8 (Informant cognition screening)",
        "generate": "Create Reports & Downloads",
        "download_json": "Download JSON",
        "download_pdf": "Download PDF",
        "download_csv": "Download CSV (features / trend)",
        "note": "Research demo only — not a clinical diagnostic tool.",
        "ica_warning": "ICA attempted — if low channel count, ICA is skipped automatically."
    },
    "العربية": {
        "title": "🧠 نيوروإرلي — EEG + PHQ-9 + AD8 (نموذج تجريبي)",
        "subtitle": "نموذج أولي للفحص المبكر: نطاقات EEG + PHQ-9 (الاكتئاب) + AD8 (الإدراك).",
        "upload": "ارفع ملف(ملفات) EDF (مفردة أو متعددة)",
        "upload_hint": "يمكنك رفع یک فایل یا چند فایل (برای تحلیل روند طولی).",
        "phq": "PHQ-9 (الاكتئاب) — اختر التكرار خلال الأسبوعين الماضيين",
        "ad8": "AD8 (فحص إدراكي - من شخص آخر)",
        "generate": "إنشاء التقارير وملفات التحميل",
        "download_json": "تحميل JSON",
        "download_pdf": "تحميل PDF",
        "download_csv": "تحميل CSV (الخصائص / الاتجاه)",
        "note": "نموذج بحثي فقط — ليس تشخيصًا طبيًا.",
        "ica_warning": "تمت محاولة ICA — إذا كان عدد القنوات منخفضًا، يتم تخطي ICA تلقائيًا."
    }
}

T = TEXT[lang]

st.title(T["title"])
st.caption(T["subtitle"])

# -------------------------
# EEG band definitions
# -------------------------
BANDS = {
    "Delta (0.5-4 Hz)": (0.5, 4),
    "Theta (4-8 Hz)": (4, 8),
    "Alpha (8-12 Hz)": (8, 12),
    "Beta (12-30 Hz)": (12, 30),
    "Gamma (30-45 Hz)": (30, 45),
}

# -------------------------
# Helper functions
# -------------------------
def safe_ica_clean(raw, n_components=15):
    """Try ICA artifact removal; if fails or too few channels, return raw unchanged."""
    try:
        picks = mne.pick_types(raw.info, eeg=True)
        if len(picks) < 4:
            # too few channels for reliable ICA
            return raw, False
        ica = mne.preprocessing.ICA(n_components=min(n_components, len(picks)-1),
                                    random_state=42, max_iter="auto")
        ica.fit(raw)
        # automatic detection of EOG components (if EOG channels present)
        eog_chs = mne.pick_types(raw.info, eog=True)
        if len(eog_chs) > 0:
            eog_inds, scores = ica.find_bads_eog(raw, threshold=3.0)
            if eog_inds:
                ica.exclude = eog_inds
        ica.apply(raw)
        return raw, True
    except Exception:
        return raw, False

def preprocess_raw_for_analysis(raw):
    """Apply filters and ICA (with safe fallback)."""
    # basic bandpass
    try:
        raw.filter(1., 50., fir_design="firwin", verbose=False)
    except Exception:
        # sometimes filtering fails if data has issues; skip in that case
        pass
    # notch line noise
    try:
        raw.notch_filter(np.arange(50, 251, 50), verbose=False)
    except Exception:
        pass
    raw_copy = raw.copy()
    raw_clean, ica_ok = safe_ica_clean(raw_copy)
    return raw_clean, ica_ok

def compute_band_powers_from_raw(raw, fmin=0.5, fmax=45.0):
    """Compute Welch PSD and integrate powers in BANDS. Return dict."""
    try:
        psd = raw.compute_psd(method="welch", fmin=fmin, fmax=fmax, n_fft=2048, n_overlap=1024, verbose=False)
        psds, freqs = psd.get_data(return_freqs=True)
        psd_mean = psds.mean(axis=0) if psds.ndim == 2 else psds
    except Exception:
        # fallback: compute simple spectrum from data
        data = raw.get_data()
        sf = int(raw.info.get("sfreq", 256))
        freqs = np.fft.rfftfreq(min(4096, data.shape[1]), 1.0/sf)
        S = np.abs(np.fft.rfft(data.mean(axis=0), n=4096))
        psd_mean = S
    band_powers = {}
    for name, (lo, hi) in BANDS.items():
        mask = (freqs >= lo) & (freqs < hi)
        if mask.any():
            band_powers[name] = float(np.trapz(psd_mean[mask], freqs[mask]))
        else:
            band_powers[name] = 0.0
    return band_powers

def make_bar_png(band_powers):
    labels = list(band_powers.keys())
    values = [band_powers[k] for k in labels]
    fig, ax = plt.subplots(figsize=(7,3), dpi=120)
    ax.bar(labels, values)
    ax.set_ylabel("Integrated power (a.u.)")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

def make_signal_snippet_png(raw, seconds=10):
    picks = mne.pick_types(raw.info, eeg=True)
    data = raw.get_data(picks=picks) if len(picks)>0 else raw.get_data()
    ch0 = data[0] if data.ndim==2 else data
    sf = int(raw.info.get("sfreq", 256))
    n = min(len(ch0), seconds * sf)
    t = np.arange(n) / sf
    fig, ax = plt.subplots(figsize=(7,2), dpi=120)
    ax.plot(t, ch0[:n])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (a.u.)")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

def eeg_heuristics(band_powers):
    a = band_powers.get("Alpha (8-12 Hz)", 1e-9)
    th = band_powers.get("Theta (4-8 Hz)", 0.0)
    b = band_powers.get("Beta (12-30 Hz)", 0.0)
    theta_alpha = float(th / a) if a>0 else 0.0
    beta_alpha = float(b / a) if a>0 else 0.0
    return {"Theta/Alpha": round(theta_alpha,3), "Beta/Alpha": round(beta_alpha,3),
            "EEG_Depression_hint": "Higher theta/alpha suggests increased depression-related pattern" if theta_alpha>0.8 else "Lower",
            "EEG_Cognitive_hint": "Low beta/alpha may indicate cognitive concern" if beta_alpha<0.5 else "Normal"}

def normalize_dict(d):
    vals = np.array(list(d.values()), dtype=float)
    if vals.max()==vals.min():
        return {k:0.0 for k in d}
    mn, mx = vals.min(), vals.max()
    return {k:(float(v)-mn)/(mx-mn) for k,v in d.items()}

def compute_early_risk(band_powers, phq_score, ad8_score, weights=(0.5,0.3,0.2)):
    # EEG proxy: high theta/alpha (risk) and low beta/alpha (risk)
    heur = eeg_heuristics(band_powers)
    theta_alpha = heur["Theta/Alpha"]
    beta_alpha = heur["Beta/Alpha"]
    eeg_feat = {"theta_alpha": theta_alpha, "inv_beta_alpha": 1.0-beta_alpha}
    eeg_norm = normalize_dict(eeg_feat)
    eeg_score = (eeg_norm.get("theta_alpha",0)+eeg_norm.get("inv_beta_alpha",0))/2.0
    phq_norm = min(max(phq_score/27.0,0.0),1.0)
    ad8_norm = min(max(ad8_score/8.0,0.0),1.0)
    early = weights[0]*eeg_score + weights[1]*phq_norm + weights[2]*ad8_norm
    return min(max(early,0.0),1.0), {"eeg_score":round(eeg_score,3), "phq_norm":round(phq_norm,3), "ad8_norm":round(ad8_norm,3)}

# -------------------------
# UI: Upload (single or multiple)
# -------------------------
st.header(T["upload"])
st.write(T["upload_hint"])
uploaded_files = st.file_uploader("Select .edf file(s)", type=["edf"], accept_multiple_files=True)

# placeholders for results
session_results = []  # list of dicts per file/session

if uploaded_files:
    st.info(f"{len(uploaded_files)} file(s) received.")
    for f in uploaded_files:
        with st.spinner(f"Processing {f.name} ..."):
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmp:
                    tmp.write(f.read())
                    tmp_path = tmp.name
                raw = mne.io.read_raw_edf(tmp_path, preload=True, verbose=False)
                raw_clean, ica_ok = preprocess_raw_for_analysis(raw)
                band_p = compute_band_powers_from_raw(raw_clean)
                heur = eeg_heuristics(band_p)
                sig_png = make_signal_snippet_png(raw_clean, seconds=8)
                bar_png = make_bar_png(band_p)
                session_results.append({
                    "filename": f.name,
                    "sfreq": raw.info.get("sfreq"),
                    "band_powers": band_p,
                    "heuristics": heur,
                    "ica_ok": ica_ok,
                    "sig_png": sig_png,
                    "bar_png": bar_png
                })
                st.success(f"{f.name} processed. Channels: {len(mne.pick_types(raw.info,eeg=True))}, ICA: {'OK' if ica_ok else 'skipped'}")
            except Exception as e:
                st.error(f"Error processing {f.name}: {e}")

# -------------------------
# PHQ-9 (two-step for Q5 & Q8)
# -------------------------
st.header(T["phq"])
phq_answers = []
phq_details = {}
PHQ_OPTIONS_EN = ["0 = Not at all","1 = Several days","2 = More than half the days","3 = Nearly every day"]
PHQ_OPTIONS_AR = ["٠ = أبدا","١ = عدة أيام","٢ = أكثر من نصف الأيام","٣ = تقريباً كل يوم"]
PHQ_OPTIONS = PHQ_OPTIONS_EN if lang=="English" else PHQ_OPTIONS_AR

# Q1-Q4
q_texts_en = [
    "1. Little interest or pleasure in doing things",
    "2. Feeling down, depressed, or hopeless",
    "3. Trouble falling or staying asleep, or sleeping too much",
    "4. Feeling tired or having little energy"
]
q_texts_ar = [
    "١. قلة الاهتمام أو المتعة في القيام بالأشياء",
    "٢. الشعور بالحزن أو الاكتئاب أو اليأس",
    "٣. مشاكل في النوم أو النوم المفرط",
    "٤. الشعور بالتعب أو قلة الطاقة"
]
q_texts = q_texts_en if lang=="English" else q_texts_ar

for i, q in enumerate(q_texts, start=1):
    ans = st.selectbox(q, PHQ_OPTIONS, index=0, key=f"phq_{i}_{lang}")
    phq_answers.append(PHQ_OPTIONS.index(ans))

# Q5 appetite (two-step)
q5_title = "5. Appetite changes / Poor appetite or overeating" if lang=="English" else "٥. تغيّر الشهية (فقدان الشهية أو الإفراط في الأكل)"
st.subheader(q5_title)
q5_types_en = ["Poor appetite","Overeating"]
q5_types_ar = ["فقدان الشهية","الإفراط في الأكل"]
q5_types = q5_types_en if lang=="English" else q5_types_ar
q5_type = st.radio("" , q5_types, index=0, key=f"q5_type_{lang}")
q5_ans = st.selectbox("How often?" if lang=="English" else "كم مرة؟", PHQ_OPTIONS, index=0, key=f"phq5_{lang}")
phq_answers.append(PHQ_OPTIONS.index(q5_ans))
phq_details["Q5_type"] = q5_type

# Q6-Q7
q6_text = "6. Feeling bad about yourself — or that you are a failure" if lang=="English" else "٦. الشعور بأنك عديم القيمة أو فاشل"
q7_text = "7. Trouble concentrating on things (reading/TV)" if lang=="English" else "٧. صعوبة في التركيز على القراءة أو مشاهدة التلفاز"
ans6 = st.selectbox(q6_text, PHQ_OPTIONS, index=0, key=f"phq6_{lang}")
phq_answers.append(PHQ_OPTIONS.index(ans6))
ans7 = st.selectbox(q7_text, PHQ_OPTIONS, index=0, key=f"phq7_{lang}")
phq_answers.append(PHQ_OPTIONS.index(ans7))

# Q8 movement (two-step)
q8_title = "8. Movement / speaking slowly OR being fidgety/restless" if lang=="English" else "٨. بطء في الحركة/الكلام أو القلق/الحركة المفرطة"
st.subheader(q8_title)
q8_types_en = ["Moving/speaking slowly","Fidgety or restless"]
q8_types_ar = ["حركة/تحدث ببطء","قلق أو توتر (حركة مفرطة)"]
q8_types = q8_types_en if lang=="English" else q8_types_ar
q8_type = st.radio("", q8_types, index=0, key=f"q8_type_{lang}")
q8_ans = st.selectbox("How often?" if lang=="English" else "كم مرة؟", PHQ_OPTIONS, index=0, key=f"phq8_{lang}")
phq_answers.append(PHQ_OPTIONS.index(q8_ans))
phq_details["Q8_type"] = q8_type

# Q9
q9_text = "9. Thoughts that you would be better off dead or hurting yourself" if lang=="English" else "٩. أفكار عن الموت أو إيذاء النفس"
ans9 = st.selectbox(q9_text, PHQ_OPTIONS, index=0, key=f"phq9_{lang}")
phq_answers.append(PHQ_OPTIONS.index(ans9))

phq_score = sum(phq_answers)
# risk label
if phq_score < 5:
    phq_label = "Minimal" if lang=="English" else "طفيف"
elif phq_score < 10:
    phq_label = "Mild" if lang=="English" else "خفيف"
elif phq_score < 15:
    phq_label = "Moderate" if lang=="English" else "متوسط"
elif phq_score < 20:
    phq_label = "Moderately severe" if lang=="English" else "شديد إلى حد ما"
else:
    phq_label = "Severe" if lang=="English" else "شديد"

st.write((f"PHQ-9 Score: {phq_score} / 27 — {phq_label}") if lang=="English" else (f"درجة PHQ-9: {phq_score} / ٢٧ — {phq_label}"))

# -------------------------
# AD8
# -------------------------
st.header(T["ad8"])
AD8_QS_EN = [
    "1. Problems with judgment (e.g., bad financial decisions)",
    "2. Reduced interest in hobbies/activities",
    "3. Repeats the same questions/stories",
    "4. Trouble learning how to use a tool/appliance",
    "5. Forgets the correct month or year",
    "6. Difficulty handling finances (paying bills)",
    "7. Trouble remembering appointments",
    "8. Everyday thinking is getting worse",
]
AD8_QS_AR = [
    "١. مشاكل في الحكم (مثل قرارات مالية سيئة)",
    "٢. قلة الاهتمام بالهوايات/الأنشطة",
    "٣. يكرر نفس الأسئلة أو القصص",
    "٤. صعوبة في تعلم استخدام أدوات أو أجهزة",
    "٥. ينسى الشهر أو السنة الصحيحة",
    "٦. صعوبة في التعامل مع الشؤون المالية",
    "٧. صعوبة في تذكر المواعيد",
    "٨. تدهور التفكير اليومي",
]
ad8_qs = AD8_QS_EN if lang=="English" else AD8_QS_AR
ad8_answers = []
for i,q in enumerate(ad8_qs, start=1):
    ans = st.selectbox(q, ["No","Yes"] if lang=="English" else ["لا","نعم"], index=0, key=f"ad8_{i}_{lang}")
    yes_label = "Yes" if lang=="English" else "نعم"
    ad8_answers.append(1 if ans==yes_label else 0)
ad8_score = sum(ad8_answers)
ad8_label = ( "Possible concern (≥2)" if ad8_score>=2 else "Low" ) if lang=="English" else ( "احتمال قلق (≥٢)" if ad8_score>=2 else "منخفض" )
st.write((f"AD8 Score: {ad8_score} / 8 — {ad8_label}") if lang=="English" else (f"درجة AD8: {ad8_score} / ٨ — {ad8_label}"))

# -------------------------
# Generate reports / trend analysis
# -------------------------
st.header(T["generate"])
if st.button("Generate (JSON / PDF / CSV)"):
    # if no sessions processed, allow generating with only questionnaires
    if not session_results:
        st.warning("No EEG files processed — generating questionnaire-only report." if lang=="English" else "لم تتم معالجة أي ملفات EEG — سيتم إنشاء تقرير بالاستبيانات فقط.")
    # collect per-session features
    features_rows = []
    for s in session_results:
        row = {
            "filename": s["filename"],
            "sfreq": s.get("sfreq"),
        }
        row.update(s["band_powers"])
        row.update({f"{k}_heur": v for k,v in s["heuristics"].items()})
        features_rows.append(row)
    # combine into DataFrame (if any)
    df_features = pd.DataFrame(features_rows) if features_rows else pd.DataFrame()

    # Early risk per latest session (or NaN)
    if session_results:
        last = session_results[-1]
        early_index, early_components = compute_early_risk(last["band_powers"], phq_score, ad8_score)
    else:
        early_index, early_components = compute_early_risk({k:0.0 for k in BANDS.keys()}, phq_score, ad8_score)

    results = {
        "timestamp": datetime.now().isoformat(),
        "language": lang,
        "PHQ9": {"score": phq_score, "label": phq_label, "answers": phq_answers, "details": phq_details},
        "AD8": {"score": ad8_score, "label": ad8_label, "answers": ad8_answers},
        "EarlyRisk": {"index": round(early_index,3), "components": early_components},
        "sessions": [
            {"filename": s["filename"], "band_powers": s["band_powers"], "heuristics": s["heuristics"], "ica_ok": s["ica_ok"]}
            for s in session_results
        ],
        "note": T["note"]
    }

    # JSON download
    json_bytes = io.BytesIO(json.dumps(results, indent=2, ensure_ascii=False).encode("utf-8"))
    st.download_button(T["download_json"], data=json_bytes, file_name="neuroearly_report.json", mime="application/json")

    # CSV features (if any)
    if not df_features.empty:
        csv_buf = io.StringIO()
        df_features.to_csv(csv_buf, index=False)
        st.download_button(T["download_csv"], data=csv_buf.getvalue(), file_name="neuroearly_features.csv", mime="text/csv")

    # PDF: include summary + last session plots + trend if multiple
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4)
    styles = getSampleStyleSheet()
    flow = []
    flow.append(Paragraph(T["title"], styles["Title"]))
    flow.append(Spacer(1,8))
    flow.append(Paragraph(f"Timestamp: {results['timestamp']}", styles["Normal"]))
    flow.append(Spacer(1,8))
    # Early Risk
    txt_er = f"Early Risk Index: {results['EarlyRisk']['index']}"
    flow.append(Paragraph(txt_er, styles["Heading3"]))
    flow.append(Paragraph(f"Components: {results['EarlyRisk']['components']}", styles["Normal"]))
    flow.append(Spacer(1,8))
    # PHQ and AD8
    flow.append(Paragraph("<b>PHQ-9</b>", styles["Heading3"]))
    flow.append(Paragraph(f"Score: {phq_score} / 27 — {phq_label}", styles["Normal"]))
    flow.append(Spacer(1,6))
    flow.append(Paragraph("<b>AD8</b>", styles["Heading3"]))
    flow.append(Paragraph(f"Score: {ad8_score} / 8 — {ad8_label}", styles["Normal"]))
    flow.append(Spacer(1,8))

    # If sessions exist: add table and images
    if session_results:
        # table of band powers for each session
        cols = ["filename"] + list(BANDS.keys())
        table_data = [cols]
        for s in session_results:
            row = [s["filename"]] + [f"{s['band_powers'].get(b,0):.6g}" for b in BANDS.keys()]
            table_data.append(row)
        tbl = Table(table_data, colWidths=[120] + [70]*len(BANDS))
        tbl.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,0),colors.lightgrey),
            ("GRID",(0,0),(-1,-1),0.3,colors.grey),
            ("FONTSIZE",(0,0),(-1,-1),8)
        ]))
        flow.append(tbl)
        flow.append(Spacer(1,8))

        # add last session images (bar + signal)
        last = session_results[-1]
        if last.get("bar_png"):
            flow.append(RLImage(io.BytesIO(last["bar_png"]), width=420, height=180))
            flow.append(Spacer(1,6))
        if last.get("sig_png"):
            flow.append(RLImage(io.BytesIO(last["sig_png"]), width=420, height=140))
            flow.append(Spacer(1,8))

        # if multiple sessions, create trend plot and add it
        if len(session_results) > 1:
            # trend of EarlyRisk (compute per session)
            trend_idx = []
            names = []
            for s in session_results:
                idx, _ = compute_early_risk(s["band_powers"], phq_score, ad8_score)
                trend_idx.append(idx)
                names.append(s["filename"])
            # plot trend
            fig, ax = plt.subplots(figsize=(6,2.5))
            ax.plot(names, trend_idx, marker='o')
            ax.set_ylabel("Early Risk Index")
            ax.set_xlabel("Session")
            ax.set_ylim(0,1)
            fig.tight_layout()
            tb = io.BytesIO()
            fig.savefig(tb, format="png")
            plt.close(fig)
            tb.seek(0)
            flow.append(RLImage(tb, width=420, height=140))
            flow.append(Spacer(1,8))

    flow.append(Paragraph(T["note"], styles["Italic"]))
    doc.build(flow)
    buf.seek(0)
    st.download_button(T["download_pdf"], data=buf.getvalue(), file_name="neuroearly_report.pdf", mime="application/pdf")

st.caption(T["note"])

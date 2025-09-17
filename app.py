# app.py — NeuroEarly (advanced PDF with detailed QA + radar, multilingual, ICA, trend)
import io
import json
import tempfile
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import mne

from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
)
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

# -------------------------
# Config
# -------------------------
st.set_page_config(page_title="NeuroEarly — Advanced Report", layout="wide")
st.sidebar.title("NeuroEarly")
st.sidebar.info("Prototype — research demo. Not a clinical diagnostic tool.")

lang = st.sidebar.radio("Language / اللغة", options=["English", "العربية"])
is_en = (lang == "English")

# -------------------------
# Texts
# -------------------------
TEXT = {
    "English": {
        "title": "🧠 NeuroEarly — EEG + PHQ-9 + AD8 (Advanced Report)",
        "subtitle": "Advanced PDF with question-level answers and radar (EEG vs PHQ vs AD8).",
        "upload": "Upload EDF files (single or multiple)",
        "upload_hint": "Upload one file for single session or several files for longitudinal trend.",
        "phq": "PHQ-9 (Depression) — frequency over past 2 weeks",
        "ad8": "AD8 (Informant cognition screening)",
        "generate": "Create Reports & Downloads",
        "download_json": "Download JSON",
        "download_pdf": "Download PDF (advanced)",
        "download_csv": "Download CSV (features/trend)",
        "note": "Research demo only — not a clinical diagnostic tool."
    },
    "العربية": {
        "title": "🧠 نيوروإرلي — EEG + PHQ-9 + AD8 (تقرير متقدّم)",
        "subtitle": "تقرير متقدم مع إجابات مفصلة لكل سؤال ومخطط رادار (EEG مقابل PHQ مقابل AD8).",
        "upload": "ارفع ملف(ملفات) EDF (مفردة أو متعددة)",
        "upload_hint": "ارفع ملف واحد للجلسة الواحدة أو عدة ملفات لتحليل الاتجاه الطولي.",
        "phq": "PHQ-9 (الاكتئاب) — التكرار خلال الأسبوعين الماضيين",
        "ad8": "AD8 (فحص إدراكي بواسطة شخص آخر)",
        "generate": "إنشاء التقارير وملفات التحميل",
        "download_json": "تحميل JSON",
        "download_pdf": "تحميل PDF (متقدم)",
        "download_csv": "تحميل CSV (الخصائص/الاتجاه)",
        "note": "نموذج بحثي فقط — ليس تشخيصًا طبيًا."
    }
}
T = TEXT[lang]

st.title(T["title"])
st.caption(T["subtitle"])

# -------------------------
# Bands & helpers
# -------------------------
BANDS = {
    "Delta (0.5-4 Hz)": (0.5, 4),
    "Theta (4-8 Hz)": (4, 8),
    "Alpha (8-12 Hz)": (8, 12),
    "Beta (12-30 Hz)": (12, 30),
    "Gamma (30-45 Hz)": (30, 45),
}

def safe_ica_clean(raw, n_components=15):
    try:
        picks = mne.pick_types(raw.info, eeg=True)
        if len(picks) < 4:
            return raw, False
        n_comp = min(n_components, max(1, len(picks)-1))
        ica = mne.preprocessing.ICA(n_components=n_comp, random_state=42, max_iter="auto")
        ica.fit(raw)
        try:
            eog_inds, _ = ica.find_bads_eog(raw)
            if eog_inds:
                ica.exclude = eog_inds
        except Exception:
            pass
        ica.apply(raw)
        return raw, True
    except Exception:
        return raw, False

def preprocess_raw(raw):
    try:
        raw.filter(1., 50., fir_design="firwin", verbose=False)
    except Exception:
        pass
    try:
        raw.notch_filter(np.arange(50, 251, 50), verbose=False)
    except Exception:
        pass
    rc = raw.copy()
    rc_clean, ica_ok = safe_ica_clean(rc)
    return rc_clean, ica_ok

def compute_band_powers(raw, fmin=0.5, fmax=45.0):
    try:
        psd = raw.compute_psd(method="welch", fmin=fmin, fmax=fmax, n_fft=2048, n_overlap=1024, verbose=False)
        psds, freqs = psd.get_data(return_freqs=True)
        psd_mean = psds.mean(axis=0) if psds.ndim==2 else psds
    except Exception:
        data = raw.get_data()
        sf = int(raw.info.get("sfreq", 256))
        N = min(4096, data.shape[1])
        freqs = np.fft.rfftfreq(N, 1.0/sf)
        S = np.abs(np.fft.rfft(data.mean(axis=0)[:N], n=N))
        psd_mean = S
    band_powers = {}
    for name, (lo, hi) in BANDS.items():
        mask = (freqs >= lo) & (freqs < hi)
        band_powers[name] = float(np.trapz(psd_mean[mask], freqs[mask])) if mask.any() else 0.0
    return band_powers

def eeg_heuristics(band_powers):
    alpha = band_powers.get("Alpha (8-12 Hz)", 1e-9)
    theta = band_powers.get("Theta (4-8 Hz)", 0.0)
    beta = band_powers.get("Beta (12-30 Hz)", 0.0)
    theta_alpha = theta/alpha if alpha>0 else 0.0
    beta_alpha = beta/alpha if alpha>0 else 0.0
    return {"Theta/Alpha": round(theta_alpha,3), "Beta/Alpha": round(beta_alpha,3)}

def normalize(val, lo=0.0, hi=1.0):
    return float(min(max(val, lo), hi))

def compute_early_risk(band_powers, phq_score, ad8_score, weights=(0.5,0.3,0.2)):
    heur = eeg_heuristics(band_powers)
    ta = heur["Theta/Alpha"]
    ba = heur["Beta/Alpha"]
    eeg_feats = {"ta": ta, "inv_ba": 1.0-ba}
    # simple normalization by max expected (tunable)
    eeg_norm = {}
    # set reasonable caps to avoid blow-ups
    eeg_norm["ta"] = normalize(min(ta/1.5, 1.0))   # assume 1.5 as high
    eeg_norm["inv_ba"] = normalize(min((1.0-ba)/1.0, 1.0))
    eeg_score = (eeg_norm["ta"] + eeg_norm["inv_ba"]) / 2.0
    phq_norm = normalize(phq_score / 27.0)
    ad8_norm = normalize(ad8_score / 8.0)
    early = weights[0]*eeg_score + weights[1]*phq_norm + weights[2]*ad8_norm
    return min(max(early,0.0),1.0), {"eeg_score":round(eeg_score,3), "phq_norm":round(phq_norm,3), "ad8_norm":round(ad8_norm,3)}

# plotting helpers (band bar, signal snippet, PHQ & AD8 bars, radar)
def plot_band_bar(band_powers):
    labels = list(band_powers.keys())
    vals = [band_powers[k] for k in labels]
    fig, ax = plt.subplots(figsize=(6,2.6), dpi=120)
    ax.bar(labels, vals); ax.set_ylabel("Integrated power (a.u.)"); ax.tick_params(axis='x', rotation=25)
    fig.tight_layout(); b=io.BytesIO(); fig.savefig(b,format='png'); plt.close(fig); b.seek(0); return b.getvalue()

def plot_signal_snippet(raw, seconds=8):
    picks = mne.pick_types(raw.info, eeg=True)
    data = raw.get_data(picks=picks) if len(picks)>0 else raw.get_data()
    ch0 = data[0] if data.ndim==2 else data
    sf = int(raw.info.get("sfreq",256))
    n = min(len(ch0), seconds*sf)
    t = np.arange(n)/sf
    fig,ax=plt.subplots(figsize=(6,2.2), dpi=120)
    ax.plot(t,ch0[:n]); ax.set_xlabel("Time (s)"); ax.set_ylabel("Amplitude (a.u.)"); fig.tight_layout()
    b=io.BytesIO(); fig.savefig(b,format='png'); plt.close(fig); b.seek(0); return b.getvalue()

def plot_phq_bar(answers):
    labels = [f"Q{i+1}" for i in range(len(answers))]
    vals = answers
    fig, ax = plt.subplots(figsize=(6,1.8), dpi=120)
    ax.bar(labels, vals); ax.set_ylim(0,3); ax.set_ylabel("0-3"); ax.set_title("PHQ-9 responses")
    fig.tight_layout(); b=io.BytesIO(); fig.savefig(b,format='png'); plt.close(fig); b.seek(0); return b.getvalue()

def plot_ad8_bar(answers):
    labels = [f"Q{i+1}" for i in range(len(answers))]
    vals = answers
    fig, ax = plt.subplots(figsize=(6,1.2), dpi=120)
    ax.bar(labels, vals); ax.set_ylim(0,1); ax.set_ylabel("0/1"); ax.set_title("AD8 responses")
    fig.tight_layout(); b=io.BytesIO(); fig.savefig(b,format='png'); plt.close(fig); b.seek(0); return b.getvalue()

def plot_radar(eeg_score, phq_norm, ad8_norm):
    # radar for three axes
    labels = ['EEG','PHQ9','AD8']
    stats = [eeg_score, phq_norm, ad8_norm]
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    stats = stats + stats[:1]
    angles = angles + angles[:1]
    fig=plt.figure(figsize=(4.5,4.5), dpi=120)
    ax=fig.add_subplot(111, polar=True)
    ax.plot(angles, stats, 'o-', linewidth=2)
    ax.fill(angles, stats, alpha=0.25)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_ylim(0,1)
    fig.tight_layout()
    b=io.BytesIO(); fig.savefig(b,format='png'); plt.close(fig); b.seek(0); return b.getvalue()

# -------------------------
# UI: upload files
# -------------------------
st.header(T["upload"])
st.write(T["upload_hint"])
uploaded_files = st.file_uploader("Select .edf file(s)", type=["edf"], accept_multiple_files=True)

session_results = []
if uploaded_files:
    st.info(f"{len(uploaded_files)} file(s) uploaded.")
    for f in uploaded_files:
        with st.spinner(f"Processing {f.name} ..."):
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmp:
                    tmp.write(f.read()); tmp_path = tmp.name
                raw = mne.io.read_raw_edf(tmp_path, preload=True, verbose=False)
                raw_clean, ica_ok = preprocess_raw(raw)
                band_p = compute_band_powers(raw_clean)
                heur = eeg_heuristics(band_p)
                sig_png = plot_signal_snippet(raw_clean, seconds=8)
                bar_png = plot_band_bar(band_p)
                session_results.append({
                    "filename": f.name,
                    "sfreq": raw.info.get("sfreq"),
                    "band_powers": band_p,
                    "heuristics": heur,
                    "ica_ok": ica_ok,
                    "sig_png": sig_png,
                    "bar_png": bar_png
                })
                st.success(f"{f.name} processed — channels: {len(mne.pick_types(raw.info,eeg=True))}, ICA: {'OK' if ica_ok else 'skipped'}")
                st.image(bar_png, caption=f"{f.name} - band powers", use_column_width=False)
            except Exception as e:
                st.error(f"Error processing {f.name}: {e}")

# -------------------------
# Questionnaires: PHQ-9 (two-step Q5 & Q8) + AD8
# -------------------------
st.header(T["phq"])
PHQ_OPTIONS_EN = ["0 = Not at all","1 = Several days","2 = More than half the days","3 = Nearly every day"]
PHQ_OPTIONS_AR = ["٠ = أبداً","١ = عدة أيام","٢ = أكثر من نصف الأيام","٣ = تقريباً كل يوم"]
PHQ_OPTIONS = PHQ_OPTIONS_EN if is_en else PHQ_OPTIONS_AR

phq_questions_en = [
    "1. Little interest or pleasure in doing things",
    "2. Feeling down, depressed, or hopeless",
    "3. Trouble falling or staying asleep, or sleeping too much",
    "4. Feeling tired or having little energy",
]
phq_questions_ar = [
    "١. قلة الاهتمام أو المتعة في القيام بالأشياء",
    "٢. الشعور بالحزن أو الاكتئاب أو اليأس",
    "٣. مشاكل في النوم أو النوم المفرط",
    "٤. الشعور بالتعب أو قلة الطاقة",
]
q_texts = phq_questions_en if is_en else phq_questions_ar
phq_answers = []
phq_q_details = {}

for i,q in enumerate(q_texts, start=1):
    ans = st.selectbox(q, PHQ_OPTIONS, index=0, key=f"phq_{i}_{lang}")
    phq_answers.append(PHQ_OPTIONS.index(ans))

# Q5 appetite two-step
q5_title = "5. Appetite changes (Poor appetite OR Overeating)" if is_en else "٥. تغيّر الشهية (فقدان الشهية أو الإفراط في الأكل)"
st.subheader(q5_title)
q5_types_en = ["Poor appetite","Overeating"]
q5_types_ar = ["فقدان الشهية","الإفراط في الأكل"]
q5_types = q5_types_en if is_en else q5_types_ar
q5_type = st.radio("", q5_types, index=0, key=f"q5_type_{lang}")
q5_opt = st.selectbox("How often?" if is_en else "كم مرة؟", PHQ_OPTIONS, index=0, key=f"phq5_{lang}")
phq_answers.append(PHQ_OPTIONS.index(q5_opt))
phq_q_details["Q5_type"] = q5_type

# Q6-Q7
q6 = "6. Feeling bad about yourself — or that you are a failure" if is_en else "٦. الشعور بأنك عديم القيمة أو فاشل"
q7 = "7. Trouble concentrating on things (reading/TV)" if is_en else "٧. صعوبة في التركيز على القراءة أو مشاهدة التلفاز"
opt6 = st.selectbox(q6, PHQ_OPTIONS, index=0, key=f"phq6_{lang}'); ")
phq_answers.append(PHQ_OPTIONS.index(opt6))
opt7 = st.selectbox(q7, PHQ_OPTIONS, index=0, key=f"phq7_{lang}")
phq_answers.append(PHQ_OPTIONS.index(opt7))

# Q8 movement two-step
q8_title = "8. Movement / restlessness (Moving/speaking slowly OR Fidgety/restless)" if is_en else "٨. الحركة/القلق (بطء في الحركة أو قلق/حركة مفرطة)"
st.subheader(q8_title)
q8_types_en = ["Moving/speaking slowly","Fidgety or restless"]
q8_types_ar = ["حركة/تحدث ببطء","قلق أو توتر (حركة مفرطة)"]
q8_types = q8_types_en if is_en else q8_types_ar
q8_type = st.radio("", q8_types, index=0, key=f"q8_type_{lang}")
q8_opt = st.selectbox("How often?" if is_en else "كم مرة؟", PHQ_OPTIONS, index=0, key=f"phq8_{lang}")
phq_answers.append(PHQ_OPTIONS.index(q8_opt))
phq_q_details["Q8_type"] = q8_type

# Q9
q9 = "9. Thoughts that you would be better off dead or hurting yourself" if is_en else "٩. أفكار عن الموت أو إيذاء النفس"
opt9 = st.selectbox(q9, PHQ_OPTIONS, index=0, key=f"phq9_{lang}")
phq_answers.append(PHQ_OPTIONS.index(opt9))

phq_score = sum(phq_answers)
if phq_score < 5: phq_label = "Minimal" if is_en else "طفيف"
elif phq_score < 10: phq_label = "Mild" if is_en else "خفيف"
elif phq_score < 15: phq_label = "Moderate" if is_en else "متوسط"
elif phq_score < 20: phq_label = "Moderately severe" if is_en else "شديد إلى حد ما"
else: phq_label = "Severe" if is_en else "شديد"

st.write((f"PHQ-9 Score: {phq_score} / 27 — {phq_label}") if is_en else (f"درجة PHQ-9: {phq_score} / ٢٧ — {phq_label}"))

# AD8
st.header(T["ad8"])
AD8_EN = [
    "1. Problems with judgment (e.g., bad financial decisions)",
    "2. Reduced interest in hobbies/activities",
    "3. Repeats the same questions or stories",
    "4. Trouble learning to use tools or appliances",
    "5. Forgets the correct month or year",
    "6. Difficulty handling finances (paying bills)",
    "7. Trouble remembering appointments",
    "8. Everyday thinking is getting worse",
]
AD8_AR = [
    "١. مشاكل في الحكم (مثل قرارات مالية سيئة)",
    "٢. قلة الاهتمام بالهوايات/الأنشطة",
    "٣. يكرر نفس الأسئلة أو القصص",
    "٤. صعوبة في تعلم استخدام أدوات أو أجهزة",
    "٥. ينسى الشهر أو السنة الصحيحة",
    "٦. صعوبة في التعامل مع الشؤون المالية (دفع الفواتير)",
    "٧. صعوبة في تذكر المواعيد",
    "٨. تدهور التفكير اليومي",
]
qs_ad8 = AD8_EN if is_en else AD8_AR
ad8_options = ["No","Yes"] if is_en else ["لا","نعم"]
yes_label = "Yes" if is_en else "نعم"
ad8_answers = []
for i,q in enumerate(qs_ad8, start=1):
    ans = st.selectbox(q, ad8_options, index=0, key=f"ad8_{i}_{lang}")
    ad8_answers.append(1 if ans==yes_label else 0)
ad8_score = sum(ad8_answers)
ad8_label = ("Possible concern (≥2)" if ad8_score>=2 else "Low") if is_en else ("احتمال قلق (≥٢)" if ad8_score>=2 else "منخفض")
st.write((f"AD8 Score: {ad8_score} / 8 — {ad8_label}") if is_en else (f"درجة AD8: {ad8_score} / ٨ — {ad8_label}"))

# -------------------------
# Generate
# -------------------------
st.header(T["generate"])
if st.button("Generate (JSON / PDF / CSV)"):
    sessions = []
    for s in session_results:
        sessions.append({"filename": s["filename"], "sfreq": s.get("sfreq"), "band_powers": s["band_powers"], "heuristics": s["heuristics"], "ica_ok": s["ica_ok"]})
    last_bands = session_results[-1]["band_powers"] if session_results else {b:0.0 for b in BANDS.keys()}
    early_idx, early_comp = compute_early_risk(last_bands, phq_score, ad8_score)

    results = {
        "timestamp": datetime.now().isoformat(),
        "language": lang,
        "PHQ9": {"score": phq_score, "label": phq_label, "answers": phq_answers, "details": phq_q_details},
        "AD8": {"score": ad8_score, "label": ad8_label, "answers": ad8_answers},
        "EarlyRisk": {"index": round(early_idx,3), "components": early_comp},
        "sessions": sessions,
        "note": T.get("note","")
    }

    # JSON download
    json_bytes = io.BytesIO(json.dumps(results, indent=2, ensure_ascii=False).encode("utf-8"))
    st.download_button(T["download_json"], data=json_bytes, file_name="neuroearly_report.json", mime="application/json")

    # CSV features
    if sessions:
        df_rows = []
        for s in sessions:
            r = {"filename": s["filename"], "sfreq": s.get("sfreq")}
            r.update(s["band_powers"])
            r.update({k: v for k, v in s["heuristics"].items()})
            df_rows.append(r)
        df = pd.DataFrame(df_rows)
        csv_buf = io.StringIO(); df.to_csv(csv_buf, index=False)
        st.download_button(T["download_csv"], data=csv_buf.getvalue(), file_name="neuroearly_features.csv", mime="text/csv")

    # Build advanced PDF
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4)
    styles = getSampleStyleSheet()
    flow = []
    flow.append(Paragraph(T["title"], styles["Title"])); flow.append(Spacer(1,8))
    flow.append(Paragraph(f"Timestamp: {results['timestamp']}", styles["Normal"])); flow.append(Spacer(1,8))
    flow.append(Paragraph(f"<b>Early Risk Index:</b> {results['EarlyRisk']['index']}", styles["Heading3"]))
    flow.append(Paragraph(f"Components: {results['EarlyRisk']['components']}", styles["Normal"])); flow.append(Spacer(1,8))

    # Add PHQ table with question-level answers (localized)
    flow.append(Paragraph("<b>PHQ-9 (detailed answers)</b>" if is_en else "<b>PHQ-9 (الإجابات التفصيلية)</b>", styles["Heading3"]))
    # Build table rows: Question | Answer (text)
    phq_table = [["Question", "Answer"]] 
    # localized question text list (complete)
    PHQ_FULL_EN = [
        "1. Little interest or pleasure in doing things",
        "2. Feeling down, depressed, or hopeless",
        "3. Trouble falling or staying asleep, or sleeping too much",
        "4. Feeling tired or having little energy",
        "5. Appetite change (type recorded separately)",
        "6. Feeling bad about yourself — or that you are a failure",
        "7. Trouble concentrating on things (reading/TV)",
        "8. Movement or restlessness (type recorded separately)",
        "9. Thoughts that you would be better off dead or hurting yourself"
    ]
    PHQ_FULL_AR = [
        "١. قلة الاهتمام أو المتعة في القيام بالأشياء",
        "٢. الشعور بالحزن أو الاكتئاب أو اليأس",
        "٣. مشاكل في النوم أو النوم المفرط",
        "٤. الشعور بالتعب أو قلة الطاقة",
        "٥. تغيّر الشهية (النوع محفوظ منفصلاً)",
        "٦. الشعور بأنك عديم القيمة أو فاشل",
        "٧. صعوبة في التركيز على القراءة أو مشاهدة التلفاز",
        "٨. الحركة أو القلق (النوع محفوظ منفصلاً)",
        "٩. أفكار عن الموت أو إيذاء النفس"
    ]
    PHQ_FULL = PHQ_FULL_EN if is_en else PHQ_FULL_AR
    for i, qtxt in enumerate(PHQ_FULL):
        ans_text = PHQ_OPTIONS_EN[phq_answers[i]] if is_en else PHQ_OPTIONS_AR[phq_answers[i]]
        phq_table.append([qtxt, ans_text])
    # add Q5 and Q8 type details
    phq_table.append(["Q5_type", results["PHQ9"]["details"].get("Q5_type","-")])
    phq_table.append(["Q8_type", results["PHQ9"]["details"].get("Q8_type","-")])
    t = Table(phq_table, colWidths=[300,200])
    t.setStyle(TableStyle([("GRID",(0,0),(-1,-1),0.3,colors.grey),("BACKGROUND",(0,0),(-1,0),colors.lightgrey),("FONTSIZE",(0,0),(-1,-1),8)]))
    flow.append(t); flow.append(Spacer(1,8))

    # AD8 detailed
    flow.append(Paragraph("<b>AD8 (detailed answers)</b>" if is_en else "<b>AD8 (الإجابات التفصيلية)</b>", styles["Heading3"]))
    ad8_table = [["Question","Answer"]]
    AD8_FULL = AD8_EN if is_en else AD8_AR
    for i, qtxt in enumerate(AD8_FULL):
        ans_txt = "Yes" if results["AD8"]["answers"][i]==1 and is_en else ("نعم" if results["AD8"]["answers"][i]==1 and not is_en else ("No" if is_en else "لا"))
        ad8_table.append([qtxt, ans_txt])
    t2 = Table(ad8_table, colWidths=[300,200])
    t2.setStyle(TableStyle([("GRID",(0,0),(-1,-1),0.3,colors.grey),("BACKGROUND",(0,0),(-1,0),colors.lightgrey),("FONTSIZE",(0,0),(-1,-1),8)]))
    flow.append(t2); flow.append(Spacer(1,8))

    # Sessions table + last session images
    if sessions:
        cols = ["filename"] + list(BANDS.keys())
        table_data = [cols]
        for s in sessions:
            row = [s["filename"]] + [f"{s['band_powers'].get(b,0):.6g}" for b in BANDS.keys()]
            table_data.append(row)
        tbl = Table(table_data, colWidths=[120]+[65]*len(BANDS))
        tbl.setStyle(TableStyle([("GRID",(0,0),(-1,-1),0.3,colors.grey),("BACKGROUND",(0,0),(-1,0),colors.lightgrey),("FONTSIZE",(0,0),(-1,-1),8)]))
        flow.append(Paragraph("<b>Sessions band powers</b>" if is_en else "<b>قوى النطاقات لكل جلسة</b>", styles["Heading3"]))
        flow.append(tbl); flow.append(Spacer(1,8))

        last = session_results[-1]
        if last.get("bar_png"):
            flow.append(RLImage(io.BytesIO(last["bar_png"]), width=420, height=160)); flow.append(Spacer(1,6))
        if last.get("sig_png"):
            flow.append(RLImage(io.BytesIO(last["sig_png"]), width=420, height=120)); flow.append(Spacer(1,8))

        if len(sessions) > 1:
            trend_idx = []
            names = []
            for s in sessions:
                idx, _ = compute_early_risk(s["band_powers"], phq_score, ad8_score)
                trend_idx.append(idx); names.append(s["filename"])
            fig, ax = plt.subplots(figsize=(6,2.2))
            ax.plot(names, trend_idx, marker='o'); ax.set_ylim(0,1); ax.set_ylabel("Early Risk Index"); ax.set_xlabel("Session")
            fig.tight_layout()
            tb = io.BytesIO(); fig.savefig(tb, format="png"); plt.close(fig); tb.seek(0)
            flow.append(RLImage(tb, width=420, height=120)); flow.append(Spacer(1,8))

    # add PHQ/AD8 and radar images
    phq_png = plot_phq_bar(phq_answers); ad8_png = plot_ad8_bar(ad8_answers)
    flow.append(RLImage(io.BytesIO(phq_png), width=420, height=100)); flow.append(Spacer(1,6))
    flow.append(RLImage(io.BytesIO(ad8_png), width=420, height=80)); flow.append(Spacer(1,8))

    eeg_score = early_comp["eeg_score"]
    phq_norm = early_comp["phq_norm"]
    ad8_norm = early_comp["ad8_norm"]
    radar_png = plot_radar(eeg_score, phq_norm, ad8_norm)
    flow.append(Paragraph("<b>EEG vs PHQ vs AD8 (Radar)</b>" if is_en else "<b>الـ EEG مقابل PHQ مقابل AD8 (رادار)</b>", styles["Heading3"]))
    flow.append(RLImage(io.BytesIO(radar_png), width=320, height=320)); flow.append(Spacer(1,8))

    # Interpretation
    if is_en:
        flow.append(Paragraph("<b>Interpretation (research)</b>", styles["Heading3"]))
        flow.append(Paragraph(
            "This advanced report includes per-question answers, EEG band-powers and simple heuristics (theta/alpha, beta/alpha). "
            "Early Risk Index is a composite (EEG proxy, PHQ-9, AD8). Elevated theta/alpha has been associated with depressive patterns; "
            "low beta/alpha may indicate cognitive concerns. Use these results as screening — clinical follow-up is required.", styles["Normal"]))
    else:
        flow.append(Paragraph("<b>التفسير (بحثي)</b>", styles["Heading3"]))
        flow.append(Paragraph(
            "يتضمن هذا التقرير المتقدم إجابات كل سؤال، قوى نطاقات EEG وبعض المؤشرات البسيطة (theta/alpha, beta/alpha). "
            "مؤشر الخطر المبكر هو تركيبة من EEG وPHQ-9 وAD8. قد يرتبط ارتفاع theta/alpha بأنماط اكتئابية؛ انخفاض beta/alpha قد يشير "
            "إلى مخاوف إدراكية. هذه نتائج فحصية ويجب المتابعة سريريًا.", styles["Normal"]))

    flow.append(Spacer(1,8)); flow.append(Paragraph(T.get("note",""), styles["Italic"]))
    doc.build(flow)
    buf.seek(0)
    st.download_button(T["download_pdf"], data=buf.getvalue(), file_name="neuroearly_detailed_report.pdf", mime="application/pdf")

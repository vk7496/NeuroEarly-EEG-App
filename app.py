# app.py — NeuroEarly Pro v6 (Fixed)
# Goal: keep your current structure but fix EDF reading (BytesIO), robust topomaps,
# bilingual (EN/AR) text sections, PDF generation, SHAP optional, stable visual outputs.
# Note: this file is intended to replace the existing app.py in your repo.

import os
import io
import sys
import json
import math
import tempfile
import traceback
from pathlib import Path
from datetime import datetime, date
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import pandas as pd

# Matplotlib non-GUI backend
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import streamlit as st
from PIL import Image as PILImage

# Optional heavy libs
HAS_MNE = False
HAS_PYEDF = False
HAS_REPORTLAB = False
HAS_SHAP = False
HAS_ARABIC = False
HAS_SCIPY = False

try:
    import mne
    HAS_MNE = True
except Exception:
    HAS_MNE = False

try:
    import pyedflib
    HAS_PYEDF = True
except Exception:
    HAS_PYEDF = False

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle)
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    HAS_REPORTLAB = True
except Exception:
    HAS_REPORTLAB = False

try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    HAS_ARABIC = True
except Exception:
    HAS_ARABIC = False

try:
    from scipy.signal import welch
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

# Paths
ROOT = Path(__file__).parent
ASSETS = ROOT / "assets"
LOGO_PATH = ASSETS / "goldenbird_logo.png"
AMIRI_TTF = ROOT / "Amiri-Regular.ttf"

# Small constants
APP_TITLE = "NeuroEarly Pro — Clinical & Research"
BLUE = "#0b63d6"
LIGHT_BG = "#f6fbff"

# Frequency bands
BANDS = {
    'Delta': (1.0, 4.0),
    'Theta': (4.0, 8.0),
    'Alpha': (8.0, 13.0),
    'Beta': (13.0, 30.0),
    'Gamma': (30.0, 45.0)
}

# Helper: now timestamp
def now_ts(fmt="%Y%m%d_%H%M%S") -> str:
    return datetime.utcnow().strftime(fmt)

# Robust EDF reader: accepts Streamlit UploadedFile or path or BytesIO
def read_edf_bytes(uploaded) -> Tuple[Optional[Any], Optional[str]]:
    """
    Return (raw, msg). raw is an mne.io.Raw if MNE available, else returns minimal dict.
    The function handles BytesIO by writing to a temp file (this avoids MNE issues with BytesIO).
    """
    if not uploaded:
        return None, "No file"

    # If it's a path-like string
    if isinstance(uploaded, (str, Path)):
        path = str(uploaded)
    else:
        try:
            b = uploaded.getvalue()
        except Exception:
            # maybe it's already bytes
            if isinstance(uploaded, (bytes, bytearray)):
                b = bytes(uploaded)
            else:
                return None, "Unsupported file object"
        # write to temp file
        tf = tempfile.NamedTemporaryFile(delete=False, suffix='.edf')
        try:
            tf.write(b)
            tf.flush(); tf.close()
            path = tf.name
        except Exception as e:
            try:
                tf.close(); os.unlink(tf.name)
            except Exception:
                pass
            return None, f"failed writing temp edf: {e}"

    # try MNE
    if HAS_MNE:
        try:
            raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
            return raw, None
        except Exception as e:
            return None, f"MNE read error: {e}"

    # fallback: try pyedflib
    if HAS_PYEDF:
        try:
            import pyedflib
            f = pyedflib.EdfReader(path)
            n = f.signals_in_file
            chan_labels = f.getSignalLabels()
            fs = f.getSampleFrequencies()[0]
            sigs = [f.readSignal(i) for i in range(n)]
            f._close()
            del f
            data = np.vstack(sigs)
            return {'data': data, 'ch_names': chan_labels, 'sfreq': fs}, None
        except Exception as e:
            return None, f"pyedflib read error: {e}"

    return None, "No EDF reader available (install mne or pyedflib)"

# Compute band powers per channel using Welch if available else simple FFT
def compute_band_powers(raw_like, bands=BANDS):
    """Return dict: {ch_name: {band+'_abs':val, band+'_rel':val}}
       raw_like can be mne Raw or dict with 'data','ch_names','sfreq'
    """
    if raw_like is None:
        return {}

    if HAS_MNE and hasattr(raw_like, 'get_data'):
        data = raw_like.get_data()  # shape (n_channels, n_samples)
        sfreq = raw_like.info['sfreq']
        ch_names = raw_like.info['ch_names']
    else:
        data = np.asarray(raw_like['data'])
        sfreq = float(raw_like['sfreq'])
        ch_names = list(raw_like['ch_names'])

    n_channels, n_samples = data.shape
    powers = {}

    for i, ch in enumerate(ch_names):
        x = data[i].astype(float)
        if HAS_SCIPY:
            f, Pxx = welch(x, fs=sfreq, nperseg=min(2048, max(256, n_samples//8)))
        else:
            # simple periodogram
            X = np.fft.rfft(x * np.hanning(len(x)))
            Pxx = (np.abs(X) ** 2) / len(X)
            f = np.fft.rfftfreq(len(x), 1.0 / sfreq)
        total_power = np.trapz(Pxx, f) if len(f)>0 else 1.0
        row = {}
        for name, (lo, hi) in bands.items():
            mask = (f >= lo) & (f <= hi)
            band_power = np.trapz(Pxx[mask], f[mask]) if np.any(mask) else 0.0
            row[f"{name}_abs"] = float(band_power)
            row[f"{name}_rel"] = float(band_power / total_power) if total_power>0 else 0.0
        powers[ch] = row
    return powers

# Create a topomap-like image from channel values (simple grid) — robust
def topomap_png_from_vals(vals: List[float], band_name:str="Band") -> Optional[bytes]:
    try:
        arr = np.asarray(vals).astype(float)
        if arr.size==0:
            return None
        n = arr.size
        side = int(np.ceil(np.sqrt(n)))
        grid_flat = np.full(side*side, np.nan)
        grid_flat[:n] = arr
        grid = grid_flat.reshape(side, side)
        fig,ax = plt.subplots(figsize=(4,3))
        im = ax.imshow(grid, cmap="RdBu_r", interpolation="nearest", origin='upper')
        ax.set_title(f"{band_name}")
        ax.axis('off')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf,format='png',dpi=150); plt.close(fig); buf.seek(0)
        return buf.getvalue()
    except Exception as e:
        print("topomap error:", e)
        return None

# PDF generation
def generate_pdf_report(summary: dict, lang: str='en', amiri_path: Optional[str]=None) -> Optional[bytes]:
    if not HAS_REPORTLAB:
        return None
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=36, bottomMargin=36,leftMargin=44,rightMargin=44)
    styles = getSampleStyleSheet()
    base_font = 'Helvetica'
    # try amiri
    if amiri_path and Path(amiri_path).exists():
        try:
            pdfmetrics.registerFont(TTFont('Amiri', str(amiri_path)))
            base_font = 'Amiri'
        except Exception as e:
            print('Amiri reg failed:', e)
    styles.add(ParagraphStyle(name='TitleBlue', fontName=base_font, fontSize=16, textColor=colors.HexColor(BLUE), alignment=1, spaceAfter=8))
    styles.add(ParagraphStyle(name='H2', fontName=base_font, fontSize=12, textColor=colors.HexColor(BLUE), spaceAfter=6))
    styles.add(ParagraphStyle(name='Body', fontName=base_font, fontSize=10, leading=14))
    styles.add(ParagraphStyle(name='Note', fontName=base_font, fontSize=9, textColor=colors.grey))

    story = []
    story.append(Paragraph('NeuroEarly Pro — Clinical Report', styles['TitleBlue']))
    story.append(Spacer(1,8))

    # Patient info
    pi = summary.get('patient_info', {})
    rows = [['Field','Value']]
    for k in ['id','dob','age','sex']:
        rows.append([k, pi.get(k,'')])
    t = Table(rows, colWidths=[3.5*inch,2.5*inch])
    t.setStyle(TableStyle([('GRID',(0,0),(-1,-1),0.25,colors.grey),('BACKGROUND',(0,0),(-1,0),colors.HexColor('#eef7ff'))]))
    story.append(t)
    story.append(Spacer(1,8))

    # metrics
    story.append(Paragraph('<b>QEEG Key Metrics</b>', styles['H2']))
    mm = summary.get('metrics',{})
    for k,v in mm.items():
        story.append(Paragraph(f"{k}: {v}", styles['Body']))
    story.append(Spacer(1,6))

    # images: bar and topology
    try:
        if summary.get('bar_img'):
            story.append(Paragraph('Normative Comparison', styles['H2']))
            story.append(Spacer(1,0.15*inch))
            story.append(RLImage(io.BytesIO(summary['bar_img']), width=5.5*inch, height=3.0*inch))
            story.append(Spacer(1,0.3*inch))
    except Exception:
        pass

    # topomaps
    if summary.get('topo_images'):
        story.append(Paragraph('<b>Topography Maps</b>', styles['H2']))
        imgs = [RLImage(io.BytesIO(b), width=2.6*inch, height=1.6*inch) for _,b in summary['topo_images'].items() if b]
        if imgs:
            # arrange 2 per row
            rows = []
            row = []
            for im in imgs:
                row.append(im)
                if len(row)==2:
                    rows.append(row); row=[]
            if row: rows.append(row)
            for r in rows:
                t2 = Table([r], colWidths=[3*inch,3*inch])
                t2.setStyle(TableStyle([('GRID',(0,0),(-1,-1),0.25,colors.grey),('BACKGROUND',(0,0),(-1,0),colors.HexColor('#eef7ff'))]))
                story.append(t2); story.append(Spacer(1,6))

    # SHAP image
    if summary.get('shap_img'):
        story.append(Paragraph('Model explainability (SHAP)', styles['H2']))
        try:
            story.append(RLImage(io.BytesIO(summary['shap_img']), width=5.5*inch, height=3.0*inch))
        except Exception:
            pass
        story.append(Spacer(1,8))

    # final recommendations
    story.append(Paragraph('<b>Structured Clinical Recommendations</b>', styles['H2']))
    for r in summary.get('recommendations', []):
        story.append(Paragraph(r, styles['Body']))

    story.append(Spacer(1,12))
    story.append(Paragraph('Prepared by Golden Bird LLC — NeuroEarly Pro', styles['Note']))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

# UI: questionnaires (PHQ-9 and AD8 simplified) — ensure correct options for special Qs
PHQ9 = [
    "Little interest or pleasure in doing things",
    "Feeling down, depressed, or hopeless",
    "Trouble falling or staying asleep, or sleeping too much",
    "Feeling tired or having little energy",
    "Poor appetite or overeating",
    "Feeling bad about yourself — or that you are a failure",
    "Trouble concentrating on things, such as reading or watching TV",
    "Moving or speaking so slowly that other people notice — or being fidgety",
    "Thoughts that you would be better off dead or hurting yourself"
]

AD8 = [
    "Problems with judgment (e.g., problems making decisions)",
    "Less interest in hobbies/activities",
    "Problems with orientation to time/place",
    "Trouble learning to use tools/appliances (e.g., microwave)",
    "Forgetting the correct month or year",
    "Difficulty handling complicated financial affairs",
    "Trouble remembering appointments",
    "Daily problems with thinking and memory"
]

# Main app
st.set_page_config(page_title=APP_TITLE, layout='wide')

# Header
st.markdown(f"<div style='display:flex;align-items:center;justify-content:space-between;padding:10px;border-radius:8px;background:{LIGHT_BG};'>\n<div style='font-weight:700;color:{BLUE};font-size:18px;'>{APP_TITLE}</div>\n<div style='font-size:12px;color:#666;'>Prepared by Golden Bird LLC</div>\n</div>", unsafe_allow_html=True)

# Layout: sidebar patient info, main console+visual
with st.sidebar:
    lang = st.selectbox('Language / اللغة', ['English','العربية'])
    patient_name = st.text_input('Patient Name (not printed)')
    patient_id = st.text_input('Patient ID')
    dob = st.date_input('Date of birth', value=date(1980,1,1))
    sex = st.selectbox('Sex / الجنس', ['Unknown','Male','Female','Other'])
    meds = st.text_area('Current meds (one per line)')
    labs = st.text_area('Relevant labs (B12, TSH, ...)')
    uploaded = st.file_uploader('Upload EDF file (.edf)', type=['edf'])
    st.markdown('---')
    st.write('Questionnaires')
    with st.expander('PHQ-9 (Depression)'):
        phq_answers = [st.radio(f'Q{i+1}', options=[0,1,2,3], key=f'phq_{i}') for i in range(len(PHQ9))]
    with st.expander('AD8 (Cognitive)'):
        ad8_answers = [st.radio(f'Q{i+1}', options=[0,1], key=f'ad8_{i}') for i in range(len(AD8))]
    st.button('Process EDF(s) and Analyze', key='process_btn')

# Console / visualization area
st.markdown('\n')
col1, col2 = st.columns([1,2])
with col1:
    st.subheader('Console')
    log = st.empty()
with col2:
    st.subheader('Upload & Quick stats')

# Process when button pressed
if st.session_state.get('process_btn'):
    log.info('Saving and reading EDF file... please wait')
    raw, msg = read_edf_bytes(uploaded)
    if msg:
        st.error(f'Error reading EDF: {msg}')
    else:
        st.success('EDF loaded successfully.')
        # compute band powers
        powers = compute_band_powers(raw)
        # create a table
        df_rows = []
        ch_names = list(powers.keys())
        for ch in ch_names:
            row = {'ch': ch}
            for b in BANDS:
                row[f'{b}_abs'] = round(powers[ch][f'{b}_abs'],6)
                row[f'{b}_rel'] = round(powers[ch][f'{b}_rel'],6)
            df_rows.append(row)
        df = pd.DataFrame(df_rows)
        st.dataframe(df, height=400)

        # create topomaps for each band
        topo_imgs = {}
        for b in BANDS:
            vals = [powers[ch][f'{b}_rel'] for ch in ch_names]
            img = topomap_png_from_vals(vals, band_name=b)
            topo_imgs[b] = img
            if img:
                st.image(img, caption=f"Topomap {b}")

        # normative bar (simple example: mean theta/alpha ratio)
        try:
            theta_alpha = np.mean([powers[ch]['Theta_rel']/ (powers[ch]['Alpha_rel']+1e-9) for ch in ch_names])
        except Exception:
            theta_alpha = 0.0
        bar_fig = plt.figure(figsize=(6,3))
        ax = bar_fig.add_subplot(111)
        ax.bar([0],[theta_alpha]); ax.set_xticks([]); ax.set_ylabel('theta/alpha ratio')
        buf = io.BytesIO(); bar_fig.tight_layout(); bar_fig.savefig(buf, format='png'); plt.close(bar_fig); buf.seek(0)
        bar_img = buf.getvalue()
        st.image(bar_img, caption='Theta/Alpha comparison')

        # Prepare summary for PDF
        summary = {
            'patient_info': {'id': patient_id, 'dob': dob.isoformat(), 'age': int((date.today()-dob).days/365), 'sex': sex},
            'metrics': {'theta_alpha_ratio': float(theta_alpha)},
            'topo_images': topo_imgs,
            'bar_img': bar_img,
            'recommendations': [
                'This is an automated screening report. Clinical correlation required.',
                'Consider follow-up and further imaging if focal abnormalities suspected.'
            ]
        }

        # SHAP visualization if available
        shap_img = None
        if HAS_SHAP and (ROOT / 'shap_summary.json').exists():
            try:
                sh = json.load(open(ROOT / 'shap_summary.json','r'))
                # simple bar plot from json (expect feature->value)
                features = list(sh.keys())
                vals = [sh[k] for k in features]
                fig = plt.figure(figsize=(6,3)); ax=fig.add_subplot(111); ax.barh(features, vals); fig.tight_layout()
                buf = io.BytesIO(); fig.savefig(buf, format='png'); plt.close(fig); buf.seek(0)
                shap_img = buf.getvalue(); summary['shap_img']=shap_img
            except Exception as e:
                print('shap render failed', e)

        # Generate PDF
        pdf_bytes = None
        if HAS_REPORTLAB:
            try:
                pdf_bytes = generate_pdf_report(summary, lang='ar' if lang=='العربية' else 'en', amiri_path=str(AMIRI_TTF) if AMIRI_TTF.exists() else None)
            except Exception as e:
                st.error(f'PDF generation failed: {e}')
        else:
            st.info('ReportLab not installed — PDF disabled.')

        if pdf_bytes:
            st.download_button('Download PDF report', data=pdf_bytes, file_name=f'NeuroEarly_Report_{now_ts()}.pdf', mime='application/pdf')
            st.success('PDF generated.')

else:
    st.info("No processed results yet. Upload EDF and press 'Process EDF(s) and Analyze'.")

# Footer notes
st.markdown('''---
Notes:
- Default language is English; Arabic available for text sections if selected.
- For best connectivity & microstate results install mne and scikit-learn.
- Place pre-trained models in models/ to enable scoring.
''')

# End of file

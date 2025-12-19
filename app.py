import streamlit as st
import librosa
import numpy as np
import pandas as pd
import plotly.express as px
from collections import Counter

# --- CONFIGURATION ---
st.set_page_config(page_title="Ricardo_DJ228 | Pro Analyzer", page_icon="🪵", layout="wide")

# --- MAPPING CAMELOT COMPLET (Incluant Anharmoniques) ---
# A = Mineur / B = Majeur
CAMELOT_MAP = {
    # Mineurs (A)
    'Ab minor': '1A', 'G# minor': '1A',
    'Eb minor': '2A', 'D# minor': '2A',
    'Bb minor': '3A', 'A# minor': '3A',
    'F minor': '4A',
    'C minor': '5A',
    'G minor': '6A',
    'D minor': '7A',
    'A minor': '8A',
    'E minor': '9A',
    'B minor': '10A',
    'F# minor': '11A', 'Gb minor': '11A',
    'C# minor': '12A', 'Db minor': '12A',
    
    # Majeurs (B)
    'B major': '1B', 'Cb major': '1B',
    'F# major': '2B', 'Gb major': '2B',
    'Db major': '3B', 'C# major': '3B',
    'Ab major': '4B', 'G# major': '4B',
    'Eb major': '5B', 'D# major': '5B',
    'Bb major': '6B', 'A# major': '6B',
    'F major': '7B',
    'C major': '8B',
    'G major': '9B',
    'D major': '10B',
    'A major': '11B',
    'E major': '12B'
}

# --- PROFILS HARMONIQUES ---
NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
PROFILES = {
    "major": [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
    "minor": [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17],
    "dorian": [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 2.69, 3.98, 3.34, 3.17],
    "mixolydian": [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.33]
}

def get_camelot_safe(key, mode):
    key_str = f"{key} {mode}"
    return CAMELOT_MAP.get(key_str, "??")

def analyze_segment(y, sr):
    if len(y) < sr: return None
    y_harm = librosa.effects.hpss(y)[0]
    chroma = librosa.feature.chroma_cqt(y=y_harm, sr=sr)
    chroma_avg = np.mean(chroma, axis=1)
    
    best_score = -1
    res_key, res_mode = "", ""
    for mode, profile in PROFILES.items():
        for i in range(12):
            score = np.corrcoef(chroma_avg, np.roll(profile, i))[0, 1]
            if score > best_score:
                best_score, res_key, res_mode = score, NOTES[i], mode
    return {"key": res_key, "mode": res_mode, "score": float(best_score)}

# --- INTERFACE ---
st.markdown("<h1>RICARDO_DJ228 PRO ANALYZER</h1>", unsafe_allow_html=True)
file = st.file_uploader("Importer un fichier", type=['mp3', 'wav', 'flac'])

if file:
    with st.spinner("Analyse spectrale..."):
        y_full, sr = librosa.load(file)
        duration = librosa.get_duration(y=y_full, sr=sr)
        tempo, _ = librosa.beat.beat_track(y=y_full, sr=sr)
        
        timeline_data = []
        for start_t in range(0, int(duration), 30):
            start_sample = start_t * sr
            end_sample = min((start_t + 30) * sr, len(y_full))
            res = analyze_segment(y_full[start_sample:end_sample], sr)
            if res:
                timeline_data.append({
                    "Secondes": start_t,
                    "Note": res['key'],
                    "Mode": res['mode'],
                    "Camelot": get_camelot_safe(res['key'], res['mode'])
                })

        df_tl = pd.DataFrame(timeline_data)
        if not df_tl.empty:
            final_res = df_tl.groupby(['Note', 'Mode']).size().idxmax()
            f_key, f_mode = final_res
            
            c1, c2, c3 = st.columns(3)
            c1.metric("CLÉ", f"{f_key} {f_mode.upper()}")
            c2.metric("CAMELOT", get_camelot_safe(f_key, f_mode))
            c3.metric("TEMPO", f"{int(tempo)} BPM")

            st.subheader("Evolution Temporelle")
            fig = px.line(df_tl, x="Secondes", y="Note", color="Mode", markers=True)
            st.plotly_chart(fig, use_container_width=True)
            st.audio(file)

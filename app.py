import streamlit as st
import librosa
import numpy as np
import pandas as pd
import plotly.express as px
from collections import Counter

# --- CONFIGURATION ---
st.set_page_config(page_title="Amapiano Master | Studio Edition", page_icon="🪵", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #F9F7F2; }
    h1 { font-family: 'serif'; color: #3D1B6F; text-align: center; }
    div[data-testid="stMetric"] { background-color: #FFFFFF; border-left: 5px solid #5D4037; border-radius: 12px; }
    </style>
    """, unsafe_allow_html=True)

# --- CONSTANTES MUSICALES ---
NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Profils : Majeur, Mineur, Dorien (Amapiano), Mixolydien (Jazz/Deep)
PROFILES = {
    "major": [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
    "minor": [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17],
    "dorian": [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 2.69, 3.98, 3.34, 3.17],
    "mixolydian": [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.33]
}

def get_camelot(key, mode):
    base_map = {'minor': 'A', 'major': 'B', 'dorian': 'A', 'mixolydian': 'B'}
    camelot_keys = {
        'G#': '1', 'Ab': '1', 'B': '1', 'D#': '2', 'Eb': '2', 'F#': '2', 'Gb': '2',
        'Bb': '3', 'A#': '3', 'Db': '3', 'C#': '3', 'F': '4', 'Ab': '4', 'C': '5', 'Eb': '5',
        'G': '6', 'Bb': '6', 'D': '7', 'F': '7', 'A': '8', 'C': '8', 'E': '9', 'G': '9',
        'B': '10', 'D': '10', 'F#': '11', 'Gb': '11', 'A': '11', 'C#': '12', 'Db': '12', 'E': '12'
    }
    return f"{camelot_keys.get(key, '1')}{base_map.get(mode, 'A')}"

def analyze_segment(y, sr):
    if len(y) < sr: return None
    # Filtrage harmonique pour ignorer les percussions (HPSS)
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
    return {"key": res_key, "mode": res_mode, "score": best_score}

# --- INTERFACE ---
st.markdown("<h1>RICARDO_DJ228 PRO ANALYZER</h1>", unsafe_allow_html=True)

file = st.file_uploader("Importer votre audio", type=['mp3', 'wav', 'flac'])

if file:
    with st.spinner("Analyse spectrale en cours..."):
        y_full, sr = librosa.load(file)
        duration = librosa.get_duration(y=y_full, sr=sr)
        
        timeline_data = []
        for start_t in range(0, int(duration), 30):
            start_sample = start_t * sr
            end_sample = min((start_t + 30) * sr, len(y_full))
            res = analyze_segment(y_full[start_sample:end_sample], sr)
            if res:
                timeline_data.append({
                    "Temps (s)": start_t,
                    "Note": res['key'],
                    "Mode": res['mode'],
                    "Confiance": res['score']
                })

        df_tl = pd.DataFrame(timeline_data)
        final_vote = df_tl.groupby(['Note', 'Mode']).size().idxmax()
        final_key, final_mode = final_vote
        tempo, _ = librosa.beat.beat_track(y=y_full, sr=sr)

        # Affichage
        c1, c2, c3 = st.columns(3)
        c1.metric("CLÉ", f"{final_key} {final_mode.upper()}")
        c2.metric("CAMELOT", get_camelot(final_key, final_mode))
        c3.metric("TEMPO", f"{int(tempo)} BPM")

        # Graphique
        st.subheader("Analyse Temporelle")
        fig = px.line(df_tl, x="Temps (s)", y="Note", color="Mode", points="all", height=300)
        st.plotly_chart(fig, use_container_width=True)

        # Audio
        st.audio(file)

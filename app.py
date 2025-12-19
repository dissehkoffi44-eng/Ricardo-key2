import streamlit as st
import librosa
import numpy as np
import pandas as pd
from collections import Counter

# --- CONFIGURATION ---
st.set_page_config(page_title="RICARDO_DJ228 | Dark Edition", page_icon="🎧", layout="wide")

# CSS Personnalisé : Thème Sombre Anti-Fatigue
st.markdown("""
    <style>
    /* Fond principal : Gris très sombre (pas noir pur) */
    .stApp {
        background-color: #121212;
    }
    
    /* Titre principal : Blanc cassé pour éviter l'éblouissement */
    h1 {
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        color: #E0E0E0;
        text-align: center;
        padding-top: 20px;
    }
    
    /* Cartes de résultats : Gris foncé avec bordure subtile */
    div[data-testid="stMetric"] {
        background-color: #1E1E1E;
        border: 1px solid #333333;
        border-left: 5px solid #BB86FC; /* Accent violet doux */
        border-radius: 12px;
        padding: 20px;
    }
    
    /* Couleur du texte des métriques */
    div[data-testid="stMetricLabel"] {
        color: #B0B0B0 !important;
    }
    
    div[data-testid="stMetricValue"] {
        color: #FFFFFF !important;
    }

    /* Zone d'upload */
    .stFileUploader {
        background-color: #1E1E1E;
        border: 1px dashed #444444;
        border-radius: 15px;
        color: #E0E0E0;
    }

    /* Style des boutons */
    .stButton>button {
        background-color: #BB86FC;
        color: #000000;
        font-weight: bold;
        border-radius: 8px;
    }

    /* Info boxes */
    .stInfo {
        background-color: #2C2C2C;
        color: #D1D1D1;
        border: 1px solid #444444;
    }
    
    /* Couleur du texte global */
    p, span, label {
        color: #D1D1D1 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- LOGIQUE MUSICALE ---
NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
MAJOR_PROFILE = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
MINOR_PROFILE = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

def get_camelot(key, mode):
    camelot_map = {
        'G# minor': '1A', 'Ab minor': '1A', 'B major': '1B', 'Cb major': '1B',
        'D# minor': '2A', 'Eb minor': '2A', 'F# major': '2B', 'Gb major': '2B',
        'Bb minor': '3A', 'A# minor': '3A', 'Db major': '3B', 'C# major': '3B',
        'F minor': '4A', 'Ab major': '4B', 'C minor': '5A', 'Eb major': '5B',
        'G minor': '6A', 'Bb major': '6B', 'D minor': '7A', 'F major': '7B',
        'A minor': '8A', 'C major': '8B', 'E minor': '9A', 'G major': '9B',
        'B minor': '10A', 'D major': '10B', 'F# minor': '11A', 'Gb minor': '11A', 'A major': '11B',
        'C# minor': '12A', 'Db minor': '12A', 'E major': '12B'
    }
    return camelot_map.get(f"{key} {mode}", "1A")

def analyze_segment(y_segment, sr):
    if len(y_segment) < sr * 5: return None
    y_harm = librosa.effects.hpss(y_segment)[0]
    chroma = librosa.feature.chroma_cqt(y=y_harm, sr=sr)
    chroma_avg = np.mean(chroma, axis=1)
    best_score = -1
    res_key, res_mode = "", ""
    for i in range(12):
        for mode, profile in [("major", MAJOR_PROFILE), ("minor", MINOR_PROFILE)]:
            score = np.corrcoef(chroma_avg, np.roll(profile, i))[0, 1]
            if score > best_score:
                best_score, res_key, res_mode = score, NOTES[i], mode
    return (res_key, res_mode)

# --- INTERFACE ---
st.markdown("<h1>RICARDO_DJ228 KEY ANALYZER</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #BB86FC;'>Analyseur de tonalité haute précision - Mode Nuit</p>", unsafe_allow_html=True)

file = st.file_uploader("Déposez votre morceau ici", type=['mp3', 'wav', 'flac'])

if file:
    with st.spinner("Analyse spectrale en cours..."):
        y_full, sr = librosa.load(file)
        duration = librosa.get_duration(y=y_full, sr=sr)
        
        window_size = 30 
        num_segments = int(duration // window_size)
        segment_results = []
        
        # Focus sur le coeur du morceau (15% à 85%)
        start_at = max(0, int(num_segments * 0.15))
        end_at = min(num_segments, int(num_segments * 0.85))

        for s in range(start_at, end_at):
            start_sample = s * window_size * sr
            end_sample = (s + 1) * window_size * sr
            res = analyze_segment(y_full[start_sample:end_sample], sr)
            if res: segment_results.append(res)
        
        if not segment_results:
            res = analyze_segment(y_full, sr)
            if res: segment_results.append(res)

        votes = [f"{k} {m}" for k, m in segment_results]
        final_res = Counter(votes).most_common(1)[0][0]
        final_key, final_mode = final_res.split()
        
        tempo, _ = librosa.beat.beat_track(y=y_full, sr=sr)
        camelot = get_camelot(final_key, final_mode)

        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("TONALITÉ", f"{final_key} {final_mode.upper()}")
        c2.metric("NOTATION CAMELOT", camelot)
        c3.metric("TEMPO", f"{int(round(float(tempo)))} BPM")

        st.divider()
        st.markdown("<h3 style='color: #E0E0E0;'>🔊 Studio de Contrôle</h3>", unsafe_allow_html=True)
        
        v1, v2 = st.columns(2)
        with v1:
            st.markdown("Morceau original")
            st.audio(file)
        with v2:
            st.markdown(f"Fréquence de référence ({final_key})")
            note_freqs = {'C': 261.63, 'C#': 277.18, 'D': 293.66, 'D#': 311.13, 'E': 329.63, 'F': 349.23, 'F#': 369.99, 'G': 392.00, 'G#': 415.30, 'A': 440.00, 'A#': 466.16, 'B': 493.88}
            freq = note_freqs.get(final_key, 440.0)
            t = np.linspace(0, 3, int(22050 * 3), False)
            tone = 0.5 * np.sin(2 * np.pi * freq * t)
            st.audio(tone, sample_rate=22050)

        st.info(f"✨ Analyse terminée avec succès. Tonalité dominante : {final_key} {final_mode}.")
else:
    st.markdown("<br><p style='text-align: center; color: #555555;'>Glissez un fichier audio pour commencer...</p>", unsafe_allow_html=True)

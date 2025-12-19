import streamlit as st
import librosa
import numpy as np
import pandas as pd
from collections import Counter

# --- CONFIGURATION ---
st.set_page_config(page_title="RICARDO_DJ228 | Pro Analyzer", page_icon="🎧", layout="wide")

# Initialisation de l'historique
if 'history' not in st.session_state:
    st.session_state.history = []

# --- CSS OPTIMISÉ (ZONE UPLOAD LISIBLE + MODE NUIT) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700;900&display=swap');

    .stApp { 
        background-color: #0F0F0F; 
        font-family: 'Inter', sans-serif;
    }
    
    /* TITRES */
    h1 { color: #FFFFFF; font-weight: 900; text-align: center; font-size: 3rem !important; }
    .sub-text { color: #BB86FC; text-align: center; font-size: 1.2rem; margin-bottom: 2rem; }

    /* --- OPTIMISATION ZONE UPLOAD (BROWSE FILES) --- */
    .stFileUploader section {
        background-color: #1A1A1A !important;
        border: 2px dashed #BB86FC !important; /* Bordure violette pour attirer l'oeil */
        border-radius: 20px !important;
        padding: 3rem !important;
        transition: all 0.3s ease;
    }
    
    /* Changement de couleur au survol */
    .stFileUploader section:hover {
        border-color: #FFFFFF !important;
        background-color: #252525 !important;
    }

    /* Texte "Drag and drop file here" */
    .stFileUploader section [data-testid="stMarkdownContainer"] p {
        color: #FFFFFF !important;
        font-size: 1.3rem !important; /* Plus grand */
        font-weight: 600 !important;
        text-shadow: 0px 2px 4px rgba(0,0,0,0.5);
    }

    /* Texte "Limit 200MB per file" et formats */
    .stFileUploader section small {
        color: #B0B0B0 !important;
        font-size: 0.9rem !important;
    }

    /* Bouton "Browse files" */
    .stFileUploader button {
        background-color: #FFFFFF !important;
        color: #000000 !important;
        font-weight: 700 !important;
        border-radius: 8px !important;
        border: none !important;
        padding: 0.5rem 1rem !important;
        margin-top: 10px !important;
    }

    /* METRICS & TABLES */
    div[data-testid="stMetric"] {
        background-color: #1A1A1A;
        border: 1px solid #2D2D2D;
        border-radius: 16px;
        padding: 25px !important;
    }
    
    div[data-testid="stMetricLabel"] { color: #A0A0A0 !important; font-size: 1rem !important; }
    div[data-testid="stMetricValue"] { color: #FFFFFF !important; font-size: 2.2rem !important; }

    .stTable { background-color: #1A1A1A; border-radius: 12px; }
    
    /* BOUTONS GENERAUX */
    .stButton>button {
        background: linear-gradient(90deg, #BB86FC, #9965f4);
        color: #000000 !important;
        font-weight: 700;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- LOGIQUE MUSICALE ---
NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
MAJOR_PROFILE = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
MINOR_PROFILE = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

def get_camelot(key, mode):
    camelot_map = {
        'G# minor': '1A', 'Ab minor': '1A', 'B major': '1B', 'D# minor': '2A', 'Eb minor': '2A', 
        'F# major': '2B', 'Bb minor': '3A', 'Db major': '3B', 'F minor': '4A', 'Ab major': '4B', 
        'C minor': '5A', 'Eb major': '5B', 'G minor': '6A', 'Bb major': '6B', 'D minor': '7A', 
        'F major': '7B', 'A minor': '8A', 'C major': '8B', 'E minor': '9A', 'G major': '9B', 
        'B minor': '10A', 'D major': '10B', 'F# minor': '11A', 'A major': '11B', 'C# minor': '12A', 'E major': '12B'
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
st.markdown("<h1>RICARDO_DJ228</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-text'>Analyseur Audio Haute Précision</p>", unsafe_allow_html=True)

# Zone d'upload centrée et large
file = st.file_uploader(" ", type=['mp3', 'wav', 'flac'])

if file:
    with st.spinner("🚀 Analyse en cours..."):
        y_full, sr = librosa.load(file)
        duration = librosa.get_duration(y=y_full, sr=sr)
        
        window_size = 30 
        num_segments = int(duration // window_size)
        segment_results = []
        start_at = max(0, int(num_segments * 0.15))
        end_at = min(num_segments, int(num_segments * 0.85))

        for s in range(start_at, end_at):
            start_sample = s * window_size * sr
            end_sample = (s + 1) * window_size * sr
            res = analyze_segment(y_full[start_sample:end_sample], sr)
            if res: segment_results.append(res)
        
        if segment_results:
            votes = [f"{k} {m}" for k, m in segment_results]
            final_res = Counter(votes).most_common(1)[0][0]
            final_key, final_mode = final_res.split()
            
            tempo, _ = librosa.beat.beat_track(y=y_full, sr=sr)
            bpm = int(round(float(tempo)))
            camelot = get_camelot(final_key, final_mode)

            # Historique
            entry = {"Fichier": file.name, "Key": f"{final_key} {final_mode.upper()}", "Camelot": camelot, "BPM": bpm}
            if not st.session_state.history or st.session_state.history[0]['Fichier'] != file.name:
                st.session_state.history.insert(0, entry)

            # Affichage résultats
            st.markdown("<br>", unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            c1.metric("Tonalité", entry["Key"])
            c2.metric("Notation Camelot", camelot)
            c3.metric("Tempo", f"{bpm} BPM")
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.audio(file)

# --- HISTORIQUE ---
st.markdown("<br><hr style='border: 1px solid #2D2D2D;'><br>", unsafe_allow_html=True)
st.markdown("### 🕒 Dernières Analyses")

if st.session_state.history:
    df_history = pd.DataFrame(st.session_state.history)
    st.table(df_history)
    if st.button("🗑️ Effacer l'historique"):
        st.session_state.history = []
        st.rerun()

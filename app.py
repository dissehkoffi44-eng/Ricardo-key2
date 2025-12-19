import streamlit as st
import librosa
import numpy as np
import pandas as pd
from collections import Counter

# --- CONFIGURATION ---
st.set_page_config(page_title="RICARDO_DJ228 | Dark Edition", page_icon="🎧", layout="wide")

# Initialisation de l'historique dans la session
if 'history' not in st.session_state:
    st.session_state.history = []

# --- CSS PERSONNALISÉ (MODE NUIT SOFT) ---
st.markdown("""
    <style>
    .stApp { background-color: #121212; }
    h1, h2, h3 { font-family: 'Inter', sans-serif; color: #E0E0E0; text-align: center; }
    
    div[data-testid="stMetric"] {
        background-color: #1E1E1E;
        border: 1px solid #333333;
        border-left: 5px solid #BB86FC;
        border-radius: 12px;
        padding: 20px;
    }
    
    /* Style de l'historique */
    .history-card {
        background-color: #1E1E1E;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #333333;
        margin-bottom: 10px;
    }

    .stFileUploader { background-color: #1E1E1E; border: 1px dashed #444444; border-radius: 15px; }
    .stButton>button { background-color: #BB86FC; color: #000000; font-weight: bold; border-radius: 8px; width: 100%; }
    p, span, label { color: #D1D1D1 !important; }
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

# --- INTERFACE PRINCIPALE ---
st.markdown("<h1>RICARDO_DJ228 KEY ANALYZER</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #BB86FC;'>Analyseur de tonalité & BPM (Mode Nuit)</p>", unsafe_allow_html=True)

file = st.file_uploader("Importer un morceau", type=['mp3', 'wav', 'flac'])

if file:
    with st.spinner("Analyse spectrale détaillée..."):
        y_full, sr = librosa.load(file)
        duration = librosa.get_duration(y=y_full, sr=sr)
        
        # Analyse par segments de 30s sur le corps du morceau
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
        
        votes = [f"{k} {m}" for k, m in segment_results]
        final_res = Counter(votes).most_common(1)[0][0]
        final_key, final_mode = final_res.split()
        
        tempo, _ = librosa.beat.beat_track(y=y_full, sr=sr)
        bpm = int(round(float(tempo)))
        camelot = get_camelot(final_key, final_mode)

        # Ajouter à l'historique (éviter les doublons immédiats)
        entry = {"nom": file.name, "tonalite": f"{final_key} {final_mode.upper()}", "camelot": camelot, "bpm": bpm}
        if not st.session_state.history or st.session_state.history[0]['nom'] != file.name:
            st.session_state.history.insert(0, entry)

        # Affichage
        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("TONALITÉ", entry["tonalite"])
        c2.metric("CAMELOT", camelot)
        c3.metric("TEMPO", f"{bpm} BPM")

        st.divider()
        st.audio(file)

# --- SECTION HISTORIQUE ---
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("### 🕒 Historique des Analyses")

if st.session_state.history:
    # Création d'un tableau pour un affichage propre
    df_history = pd.DataFrame(st.session_state.history)
    st.table(df_history)
    
    if st.button("Effacer l'historique"):
        st.session_state.history = []
        st.rerun()
else:
    st.markdown("<p style='text-align: center; color: #555555;'>Aucune analyse pour le moment.</p>", unsafe_allow_html=True)

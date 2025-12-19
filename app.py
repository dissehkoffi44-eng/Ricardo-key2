import streamlit as st
import librosa
import numpy as np
import pandas as pd
import plotly.express as px
from collections import Counter

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Amapiano Master | Studio Edition", page_icon="🪵", layout="wide")

# CSS PERSONNALISÉ
st.markdown("""
    <style>
    .stApp { background-color: #F9F7F2; }
    h1 { font-family: 'Playfair Display', serif; font-weight: 900; color: #3D1B6F; text-align: center; padding-top: 20px; }
    div[data-testid="stMetric"] { 
        background-color: #FFFFFF; border: 1px solid #E0D7C6; 
        border-left: 5px solid #5D4037; border-radius: 12px; padding: 20px; 
    }
    .stInfo { background-color: #EFEBE9; color: #4E342E; border: 1px solid #D7CCC8; }
    </style>
    """, unsafe_allow_html=True)

# --- LOGIQUE MUSICALE ---
NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Profils harmoniques étendus
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
    num = camelot_keys.get(key, "1")
    let = base_map.get(mode, "A")
    return f"{num}{let}"

def analyze_segment(y, sr):
    if len(y) < sr: return None
    # PRÉ-FILTRAGE : Séparation Harmonique (isoler mélodies des drums)
    y_harm = librosa.effects.hpss(y)[0]
    # Chromagramme CQT (plus précis pour les basses Amapiano)
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

# --- INTERFACE UTILISATEUR ---
st.markdown("<h1>RICARDO_DJ228 PRO ANALYZER</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #5D4037;'>Analyse spectrale avancée : Modes, Filtrage & Timeline</p>", unsafe_allow_html=True)

file = st.file_uploader("", type=['mp3', 'wav', 'flac'])

if file:
    with st.spinner("Analyse approfondie du signal (HPSS + CQT)..."):
        # Chargement
        y_full, sr = librosa.load(file)
        duration = librosa.get_duration(y=y_full, sr=sr)
        
        # Détection du Tempo
        tempo, _ = librosa.beat.beat_track(y=y_full, sr=sr)
        
        # Analyse par segments de 30s
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
                    "Confiance": res['score']
                })

        # Données pour l'affichage
        df_tl = pd.DataFrame(timeline_data)
        
        if not df_tl.empty:
            # Vote majoritaire pour la clé globale
            final_vote = df_tl.groupby(['Note', 'Mode']).size().idxmax()
            final_key, final_mode = final_vote
            camelot = get_camelot(final_key, final_mode)

            # --- METRICS ---
            st.markdown("<br>", unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            c1.metric("CLÉ DOMINANTE", f"{final_key} {final_mode.upper()}")
            c2.metric("CAMELOT", camelot)
            c3.metric("TEMPO", f"{int(tempo)} BPM")

            # --- GRAPHIQUE TEMPOREL ---
            st.divider()
            st.subheader("📈 Évolution de la Tonalité")
            
            fig = px.line(
                df_tl, x="Secondes", y="Note", color="Mode",
                markers=True, title="Détection par segment (toutes les 30s)",
                category_orders={"Note": NOTES}, # Ordonner les notes musicalement
                color_discrete_map={"major": "#5D4037", "minor": "#8D6E63", "dorian": "#3D1B6F", "mixolydian": "#D7CCC8"}
            )
            fig.update_layout(plot_bgcolor='white')
            st.plotly_chart(fig, use_container_width=True)

            # --- AUDIO CONTROLE ---
            st.divider()
            v1, v2 = st.columns(2)
            with v1:
                st.markdown("**Vérification Audio**")
                st.audio(file)
            with v2:
                st.markdown(f"**Note Témoin ({final_key})**")
                freqs = {'C': 261.63, 'C#': 277.18, 'D': 293.66, 'D#': 311.13, 'E': 329.63, 'F': 349.23, 'F#': 369.99, 'G': 392.00, 'G#': 415.30, 'A': 440.00, 'A#': 466.16, 'B': 493.88}
                t = np.linspace(0, 2, 44100 * 2, False)
                tone = 0.3 * np.sin(2 * np.pi * freqs.get(final_key, 440) * t)
                st.audio(tone, sample_rate=44100)

            st.info(f"🌿 **Analyse Ricardo_DJ228 :** Le mode **{final_mode}** est prédominant. Pour un mix fluide, cherchez des tracks en **{camelot}** ou déplacez-vous d'un cran sur la roue Camelot.")
        else:
            st.error("Impossible d'analyser le fichier. Vérifiez le format audio.")

else:
    st.markdown("<br><br><p style='text-align: center; color: #D7CCC8;'>En attente d'un signal audio pour commencer l'analyse spectrale...</p>", unsafe_allow_html=True)

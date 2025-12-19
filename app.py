import streamlit as st
import librosa
import numpy as np
import pandas as pd
import plotly.express as px
from collections import Counter

# --- CONFIGURATION ---
st.set_page_config(page_title="Ricardo_DJ228 | Studio Edition", page_icon="🪵", layout="wide")

# CSS Thème Boisé
st.markdown("""
    <style>
    .stApp { background-color: #F9F7F2; }
    h1 { font-family: 'serif'; color: #3D1B6F; text-align: center; }
    div[data-testid="stMetric"] { background-color: #FFFFFF; border-left: 5px solid #5D4037; border-radius: 12px; }
    </style>
    """, unsafe_allow_html=True)

# --- MAPPING CAMELOT PRO & ANHARMONIQUE ---
BASE_CAMELOT = {
    'Ab': '1', 'G#': '1', 'Eb': '2', 'D#': '2', 'Bb': '3', 'A#': '3', 
    'F': '4', 'C': '5', 'G': '6', 'D': '7', 'A': '8', 'E': '9', 
    'B': '10', 'F#': '11', 'Gb': '11', 'C#': '12', 'Db': '12'
}

def get_camelot_pro(key, mode):
    number = BASE_CAMELOT.get(key, "1")
    letter = "A" if mode in ['minor', 'dorian'] else "B"
    return f"{number}{letter}"

# --- LOGIQUE ANALYSE ---
NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
PROFILES = {
    "major": [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
    "minor": [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17],
    "dorian": [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 2.69, 3.98, 3.34, 3.17],
    "mixolydian": [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.33]
}

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
    return {"key": res_key, "mode": res_mode}

# --- INTERFACE ---
st.markdown("<h1>RICARDO_DJ228 PRO ANALYZER</h1>", unsafe_allow_html=True)
file = st.file_uploader("Importer votre audio (Amapiano, House, etc.)", type=['mp3', 'wav', 'flac'])

if file:
    with st.spinner("Analyse spectrale en cours..."):
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
                    "Mode": res['mode']
                })

        df_tl = pd.DataFrame(timeline_data)
        if not df_tl.empty:
            final_res = df_tl.groupby(['Note', 'Mode']).size().idxmax()
            f_key, f_mode = final_res
            
            # --- AFFICHAGE RESULTATS ---
            st.markdown("<br>", unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            c1.metric("CLÉ DÉTECTÉE", f"{f_key} {f_mode.upper()}")
            c2.metric("NOTATION CAMELOT", get_camelot_pro(f_key, f_mode))
            c3.metric("TEMPO ESTIMÉ", f"{int(tempo)} BPM")

            # --- VERIFICATION A L'OREILLE ---
            st.divider()
            st.subheader("🔊 Vérification auditive")
            v1, v2 = st.columns(2)
            
            with v1:
                st.markdown("**Votre morceau :**")
                st.audio(file)
            
            with v2:
                st.markdown(f"**Note de référence ({f_key}) :**")
                # Fréquences des notes (Octave 3/4 pour test)
                freqs = {'C': 261.63, 'C#': 277.18, 'D': 293.66, 'D#': 311.13, 'E': 329.63, 'F': 349.23, 'F#': 369.99, 'G': 392.00, 'G#': 415.30, 'A': 440.00, 'A#': 466.16, 'B': 493.88}
                target_freq = freqs.get(f_key, 440.0)
                
                # Génération d'un son sinus riche (fondamentale + harmonique)
                duration_test = 3.0
                sr_test = 22050
                t = np.linspace(0, duration_test, int(sr_test * duration_test), False)
                # Mix de deux octaves pour une meilleure perception
                tone = 0.4 * np.sin(2 * np.pi * target_freq * t) + 0.2 * np.sin(2 * np.pi * (target_freq * 2) * t)
                
                st.audio(tone, sample_rate=sr_test)
                st.caption("Lancez les deux lecteurs en même temps pour vérifier l'harmonie.")

            # --- TIMELINE ---
            st.divider()
            st.subheader("📈 Timeline des changements de tonalité")
            fig = px.line(df_tl, x="Secondes", y="Note", color="Mode", markers=True,
                          category_orders={"Note": NOTES})
            st.plotly_chart(fig, use_container_width=True)

            st.info(f"💡 **Conseil Ricardo_DJ228 :** Le mode **{f_mode}** a été identifié. Utilisez le code Camelot **{get_camelot_pro(f_key, f_mode)}** pour trier vos bibliothèques.")

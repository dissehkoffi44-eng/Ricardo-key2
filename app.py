import streamlit as st
import librosa
import numpy as np
import pandas as pd
import plotly.express as px
from collections import Counter
import queue
import datetime
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase

# --- CONFIGURATION ---
st.set_page_config(page_title="Ricardo_DJ228 | Precision V4.5 Pro", page_icon="🎧", layout="wide")

# Initialisation de l'historique dans la session
if 'history' not in st.session_state:
    st.session_state.history = []

# Configuration des serveurs STUN
RTC_CONFIGURATION = {
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["stun:stun2.l.google.com:19302"]},
    ]
}

# --- DESIGN CSS ---
st.markdown("""
    <style>
    .stApp { background-color: #F8F9FA; color: #212529; }
    .metric-container { background: white; padding: 15px; border-radius: 12px; border: 1px solid #E0E0E0; text-align: center; transition: transform 0.3s; }
    .metric-container:hover { transform: translateY(-3px); border-color: #6366F1; }
    .label-custom { color: #666; font-size: 0.8em; font-weight: bold; }
    .value-custom { font-size: 1.5em; font-weight: 800; color: #1A1A1A; }
    .reliability-bar-bg { background-color: #E0E0E0; border-radius: 10px; height: 12px; width: 100%; margin: 5px 0; overflow: hidden; }
    .reliability-fill { height: 100%; transition: width 0.8s; display: flex; align-items: center; justify-content: center; color: white; font-size: 0.7em; }
    .live-panel { background: #111; color: #00FF41; padding: 15px; border-radius: 10px; border: 1px solid #333; font-family: monospace; }
    </style>
    """, unsafe_allow_html=True)

# --- MAPPING CAMELOT ---
# F# MINOR = 11A (Sauvegardé dans vos préférences)
BASE_CAMELOT_MINOR = {'Ab':'1A','G#':'1A','Eb':'2A','D#':'2A','Bb':'3A','A#':'3A','F':'4A','C':'5A','G':'6A','D':'7A','A':'8A','E':'9A','B':'10A','F#':'11A', 'Gb':'11A','Db':'12A','C#':'12A'}
BASE_CAMELOT_MAJOR = {'B':'1B','F#':'2B','Gb':'2B','Db':'3B','C#':'3B','Ab':'4B','G#':'4B','Eb':'5B','D#':'5B','Bb':'6B','A#':'6B','F':'7B','C':'8B','G':'9B','D':'10B','A':'11B','E':'12B'}

def get_camelot_pro(key_mode_str):
    try:
        parts = key_mode_str.split(" ")
        key, mode = parts[0], parts[1].lower()
        return BASE_CAMELOT_MINOR.get(key, "??") if mode in ['minor', 'dorian'] else BASE_CAMELOT_MAJOR.get(key, "??")
    except: return "??"

# --- MOTEUR ANALYSE ---
def analyze_audio(file_buffer, file_name):
    y, sr = librosa.load(file_buffer)
    # Détection kick/mélodie
    flatness = np.mean(librosa.feature.spectral_flatness(y=y))
    is_aligned = flatness < 0.045
    y_proc = y if is_aligned else librosa.effects.hpss(y)[0]
    
    # Analyse de la tonalité
    chroma = librosa.feature.chroma_cqt(y=y_proc, sr=sr)
    chroma_avg = np.mean(chroma, axis=1)
    NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    profile_minor = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
    
    best_score, res_key = -1, ""
    for i in range(12):
        score = np.corrcoef(chroma_avg, np.roll(profile_minor, i))[0, 1]
        if score > best_score:
            best_score, res_key = score, f"{NOTES[i]} minor"
    
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    camelot = get_camelot_pro(res_key)
    conf = int(best_score * 100)
    
    result = {
        "Heure": datetime.datetime.now().strftime("%H:%M:%S"),
        "Fichier": file_name,
        "Clef": res_key,
        "Camelot": camelot,
        "BPM": int(float(tempo)),
        "Confiance": f"{conf}%"
    }
    return result

# --- CLASSE LIVE ---
class RealTimeAudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.result_queue = queue.Queue()
        self.notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        self.profile = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

    def recv_audio(self, frame):
        audio_data = frame.to_ndarray().flatten().astype(np.float32) / 32768.0
        sr = frame.sample_rate
        try:
            chroma = librosa.feature.chroma_stft(y=audio_data, sr=sr, n_chroma=12)
            chroma_avg = np.mean(chroma, axis=1)
            best_score, res_key = -1, ""
            for i in range(12):
                score = np.corrcoef(chroma_avg, np.roll(self.profile, i))[0, 1]
                if score > best_score:
                    best_score, res_key = score, f"{self.notes[i]} minor"
            self.result_queue.put({"key": res_key, "conf": int(best_score * 100)})
        except: pass
        return frame

# --- INTERFACE ---
st.markdown("<h1 style='text-align: center;'>🎧 RICARDO_DJ228 | V4.5 PRO</h1>", unsafe_allow_html=True)
tabs = st.tabs(["📁 ANALYSE MULTI-FICHIERS", "📻 SCANNER RADIO", "🕒 HISTORIQUE"])

# ONGLET 1 : MULTI-FICHIERS
with tabs[0]:
    uploaded_files = st.file_uploader("Glissez vos morceaux ici", type=['mp3', 'wav', 'flac'], accept_multiple_files=True)
    if uploaded_files:
        if st.button("Lancer l'analyse groupée"):
            for file in uploaded_files:
                with st.spinner(f"Analyse de {file.name}..."):
                    res = analyze_audio(file, file.name)
                    st.session_state.history.insert(0, res)
                    
                    # Affichage immédiat du résultat
                    c1, c2, c3 = st.columns(3)
                    with c1: st.success(f"**{file.name}**")
                    with c2: st.info(f"Clef: {res['Camelot']} ({res['Clef']})")
                    with c3: st.warning(f"Tempo: {res['BPM']} BPM")
            st.balloons()

# ONGLET 2 : SCANNER LIVE
with tabs[1]:
    col_l, col_r = st.columns([1, 1])
    with col_l:
        ctx = webrtc_streamer(key="live", mode=WebRtcMode.SENDONLY, audio_processor_factory=RealTimeAudioProcessor, rtc_configuration=RTC_CONFIGURATION)
    with col_r:
        if ctx.audio_processor:
            try:
                data = ctx.audio_processor.result_queue.get_nowait()
                st.session_state.live_val = data
            except: pass
            
            val = st.session_state.get("live_val", {"key": "Analyse...", "conf": 0})
            st.markdown(f"<div class='live-panel'>📡 SCANNING...<br>> KEY: {val['key']}<br>> CONF: {val['conf']}%</div>", unsafe_allow_html=True)
            st.metric("CAMELOT LIVE", get_camelot_pro(val['key']))
            if st.button("Enregistrer le scan"):
                st.session_state.history.insert(0, {"Heure": "LIVE", "Fichier": "SCAN RADIO", "Clef": val['key'], "Camelot": get_camelot_pro(val['key']), "BPM": "-", "Confiance": f"{val['conf']}%"})

# ONGLET 3 : HISTORIQUE
with tabs[2]:
    st.markdown("### 🕒 Dernières Analyses")
    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        st.dataframe(df, use_container_width=True)
        if st.button("Effacer l'historique"):
            st.session_state.history = []
            st.rerun()
    else:
        st.write("Aucune analyse pour le moment.")

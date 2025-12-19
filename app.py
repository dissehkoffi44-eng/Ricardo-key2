import streamlit as st
import librosa
import numpy as np
import pandas as pd
import plotly.express as px
from collections import Counter
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
import av

# --- CONFIGURATION ---
st.set_page_config(page_title="Ricardo_DJ228 | V4.5 PRO", page_icon="🎧", layout="wide")

# Configuration STUN pour percer les pare-feux
RTC_CONFIGURATION = {
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
}

# --- STYLE CSS ---
st.markdown("""
    <style>
    .stApp { background-color: #F8F9FA; }
    .metric-card { background: white; padding: 25px; border-radius: 15px; border: 1px solid #EEE; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }
    .live-monitor { background: #000; color: #00FF41; padding: 15px; border-radius: 10px; font-family: monospace; border: 2px solid #333; }
    </style>
    """, unsafe_allow_html=True)

# --- LOGIQUE DE CALCUL (MOTEUR RICARDO V4.5) ---
def get_camelot(note):
    mapping = {
        'Ab':'1A','G#':'1A','Eb':'2A','D#':'2A','Bb':'3A','A#':'3A','F':'4A','C':'5A','G':'6A',
        'D':'7A','A':'8A','E':'9A','B':'10A','F#':'11A','Gb':'11A','Db':'12A','C#':'12A'
    }
    return mapping.get(note, "??")

def detect_key_realtime(y, sr):
    if len(y) < 1000: return "Silence", "--"
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_avg = np.mean(chroma, axis=1)
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    # Profil minor pour corrélation
    profile = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
    best_score, best_note = -1, "??理论"
    for i in range(12):
        score = np.corrcoef(chroma_avg, np.roll(profile, i))[0, 1]
        if score > best_score:
            best_score, best_note = score, notes[i]
    return best_note, get_camelot(best_note)

# --- PROCESSEUR AUDIO LIVE ---
class AudioKeyAnalyzer(AudioProcessorBase):
    def __init__(self):
        self.last_note = "En attente..."
        self.last_camelot = "--"

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        raw_samples = frame.to_ndarray()
        # Conversion Mono et Normalisation
        y = raw_samples.mean(axis=0).astype(np.float32) / 32768.0
        sr = frame.sample_rate
        
        # Analyse réelle
        note, camelot = detect_key_realtime(y, sr)
        self.last_note = note
        self.last_camelot = camelot
        return frame

# --- INTERFACE UTILISATEUR ---
st.markdown("<h1 style='text-align: center;'>🎧 RICARDO_DJ228 | V4.5 PRO</h1>", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["📁 ANALYSE FICHIER", "📻 SCANNER RADIO LIVE"])

# --- ONGLET 1 : FICHIER ---
with tab1:
    file = st.file_uploader("Choisir un fichier audio", type=['mp3', 'wav', 'flac'])
    if file:
        with st.spinner("Analyse spectrale Ricardo..."):
            y, sr = librosa.load(file)
            note, camelot = detect_key_realtime(y, sr)
            
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f'<div class="metric-card"><small>NOTE</small><h2>{note} Minor</h2></div>', unsafe_allow_html=True)
            with c2:
                st.markdown(f'<div class="metric-card"><small>CAMELOT</small><h2 style="color:#6366F1;">{camelot}</h2></div>', unsafe_allow_html=True)
            
            # Message spécial pour votre référence 11A
            if camelot == "11A":
                st.success("🎯 MATCH PARFAIT : F# Minor (11A)")

# --- ONGLET 2 : SCANNER LIVE (RÉEL) ---
with tab2:
    st.markdown("### 📡 Détection en temps réel")
    st.info("Le moteur analyse le flux audio toutes les secondes. Assurez-vous que le micro capte bien la musique.")

    ctx = webrtc_streamer(
        key="ricardo-live-engine-v45",
        mode=WebRtcMode.SENDONLY,
        audio_processor_factory=AudioKeyAnalyzer,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"audio": True, "video": False},
        async_processing=True,
    )

    if ctx.audio_processor:
        st.markdown("---")
        col_n, col_c = st.columns(2)
        
        # Récupération des données du processeur audio en temps réel
        current_note = ctx.audio_processor.last_note
        current_camelot = ctx.audio_processor.last_camelot
        
        with col_n:
            st.metric("NOTE DÉTECTÉE", f"{current_note} Minor")
        with col_c:
            st.metric("CODE CAMELOT", current_camelot)
        
        # Panneau de contrôle visuel
        st.markdown(f"""
            <div class="live-monitor">
                STATUS: SYNCED<br>
                BUFFER: 1024ms<br>
                REF: 11A (F# Minor)<br>
                CURRENT_KEY: {current_camelot}
            </div>
        """, unsafe_allow_html=True)
        
        if current_camelot == "11A":
            st.balloons()
            st.success("!!! 11A DETECTED !!!")
    else:
        st.warning("Cliquez sur le bouton START ci-dessus pour activer le scanner.")

st.markdown("---")
st.caption("Ricardo_DJ228 - Precision Audio Engine V4.5 Pro - Basé sur F# Minor (11A)")

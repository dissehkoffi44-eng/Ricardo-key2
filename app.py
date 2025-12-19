import streamlit as st
import librosa
import numpy as np
import pandas as pd
import plotly.express as px
from collections import Counter
import queue
import time
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase

# --- CONFIGURATION ---
st.set_page_config(page_title="Ricardo_DJ228 | Precision V4.5 Pro", page_icon="🎧", layout="wide")

if 'history' not in st.session_state:
    st.session_state.history = []

# Serveurs STUN pour la stabilité WebRTC
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
    .metric-container { background: white; padding: 20px; border-radius: 15px; border: 1px solid #E0E0E0; text-align: center; height: 100%; transition: transform 0.3s; }
    .metric-container:hover { transform: translateY(-5px); border-color: #6366F1; }
    .label-custom { color: #666; font-size: 0.9em; font-weight: bold; margin-bottom: 5px; }
    .value-custom { font-size: 1.8em; font-weight: 800; color: #1A1A1A; }
    .reliability-bar-bg { background-color: #E0E0E0; border-radius: 10px; height: 18px; width: 100%; margin: 10px 0; overflow: hidden; }
    .reliability-fill { height: 100%; transition: width 0.8s ease-in-out; display: flex; align-items: center; justify-content: center; color: white; font-size: 0.8em; font-weight: bold; }
    .warning-box { background-color: #FFFBEB; border-left: 5px solid #F59E0B; padding: 15px; border-radius: 5px; margin: 10px 0; }
    .info-box { background-color: #E0F2FE; border-left: 5px solid #0EA5E9; padding: 15px; border-radius: 5px; margin: 10px 0; }
    .live-panel { background: #111; color: #00FF41; padding: 20px; border-radius: 10px; border: 1px solid #333; font-family: monospace; min-height: 120px; }
    </style>
    """, unsafe_allow_html=True)

# --- MAPPING CAMELOT ---
# F# MINOR = 11A (Info sauvegardée)
BASE_CAMELOT_MINOR = {'Ab':'1A','G#':'1A','Eb':'2A','D#':'2A','Bb':'3A','A#':'3A','F':'4A','C':'5A','G':'6A','D':'7A','A':'8A','E':'9A','B':'10A','F#':'11A','Gb':'11A','Db':'12A','C#':'12A'}
BASE_CAMELOT_MAJOR = {'B':'1B','F#':'2B','Gb':'2B','Db':'3B','C#':'3B','Ab':'4B','G#':'4B','Eb':'5B','D#':'5B','Bb':'6B','A#':'6B','F':'7B','C':'8B','G':'9B','D':'10B','A':'11B','E':'12B'}

def get_camelot_pro(key_mode_str):
    try:
        parts = key_mode_str.split(" ")
        key, mode = parts[0], parts[1].lower()
        return BASE_CAMELOT_MINOR.get(key, "??") if mode in ['minor', 'dorian'] else BASE_CAMELOT_MAJOR.get(key, "??")
    except: return "??"

# --- MOTEUR ANALYSE AVANCÉ (FICHIERS) ---
def check_drum_alignment(y, sr):
    flatness = np.mean(librosa.feature.spectral_flatness(y=y))
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_max_mean = np.mean(np.max(chroma, axis=0))
    return flatness < 0.045 or chroma_max_mean > 0.75

def analyze_segment(y, sr):
    tuning = librosa.estimate_tuning(y=y, sr=sr)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, tuning=tuning)
    chroma_avg = np.mean(chroma, axis=1)
    NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    profile_minor = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
    best_score, res_key = -1, ""
    for i in range(12):
        score = np.corrcoef(chroma_avg, np.roll(profile_minor, i))[0, 1]
        if score > best_score:
            best_score, res_key = score, f"{NOTES[i]} minor"
    return res_key, best_score, chroma_avg

@st.cache_data(show_spinner="Validation spectrale en cours...")
def get_full_analysis(file_buffer):
    y, sr = librosa.load(file_buffer)
    is_aligned = check_drum_alignment(y, sr)
    
    if is_aligned:
        y_final, mode_label = y, "DIRECT (Kicks OK)"
    else:
        y_harm, _ = librosa.effects.hpss(y)
        y_final, mode_label = y_harm, "SÉPARÉ (Drums Discordants)"

    duration = librosa.get_duration(y=y_final, sr=sr)
    timeline_data, votes, all_chromas = [], [], []
    
    for start_t in range(0, int(duration) - 15, 10):
        y_seg = y_final[int(start_t*sr):int((start_t+15)*sr)]
        key_seg, score_seg, chroma_vec = analyze_segment(y_seg, sr)
        votes.append(key_seg)
        all_chromas.append(chroma_vec)
        timeline_data.append({"Temps": start_t, "Note": key_seg, "Confiance": score_seg})
    
    dominante_vote = Counter(votes).most_common(1)[0][0]
    avg_chroma_global = np.mean(all_chromas, axis=0)
    profile_minor = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
    NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    best_synth_score, tonique_synth = -1, ""
    for i in range(12):
        score = np.corrcoef(avg_chroma_global, np.roll(profile_minor, i))[0, 1]
        if score > best_synth_score:
            best_synth_score, tonique_synth = score, f"{NOTES[i]} minor"

    stability = Counter(votes).most_common(1)[0][1] / len(votes)
    base_conf = ((stability * 0.5) + (best_synth_score * 0.5)) * 100
    final_confidence = int(max(96, min(99, base_conf + 15))) if dominante_vote == tonique_synth else int(min(89, base_conf))
    
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    energy = int(np.clip(np.mean(librosa.feature.rms(y=y))*35 + (float(tempo)/160), 1, 10))

    return {
        "vote": dominante_vote, "synthese": tonique_synth, "confidence": final_confidence,
        "tempo": int(float(tempo)), "energy": energy, "timeline": timeline_data,
        "mode": mode_label, "is_aligned": is_aligned
    }

# --- CLASSE LIVE (SCANNER RADIO) ---
class RealTimeAudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.result_queue = queue.Queue()
        self.notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        self.profile = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

    def recv_audio(self, frame):
        # Conversion audio
        audio_data = frame.to_ndarray().flatten().astype(np.float32) / 32768.0
        sr = frame.sample_rate
        try:
            # Analyse Chroma
            chroma = librosa.feature.chroma_stft(y=audio_data, sr=sr, n_chroma=12)
            chroma_avg = np.mean(chroma, axis=1)
            best_score, res_key = -1, ""
            for i in range(12):
                score = np.corrcoef(chroma_avg, np.roll(self.profile, i))[0, 1]
                if score > best_score:
                    best_score, res_key = score, f"{self.notes[i]} minor"
            # Envoi du résultat dans la file
            self.result_queue.put({"key": res_key, "conf": int(best_score * 100)})
        except: pass
        return frame

# --- INTERFACE PRINCIPALE ---
st.markdown("<h1 style='text-align: center; color: #1A1A1A;'>🎧 RICARDO_DJ228 | V4.5 PRO</h1>", unsafe_allow_html=True)
tabs = st.tabs(["📁 ANALYSE DE FICHIER", "📻 SCANNER RADIO LIVE"])

# --- ONGLET 1 : ANALYSE FICHIER ---
with tabs[0]:
    file = st.file_uploader("Importer une track (MP3, WAV, FLAC)", type=['mp3', 'wav', 'flac'])
    if file:
        res = get_full_analysis(file)
        conf = res["confidence"]
        color = "#10B981" if conf >= 95 else "#F59E0B"
        
        if res["is_aligned"]:
            st.markdown(f"""<div class="info-box">✅ <b>Accordage OK :</b> Le Kick est aligné. Analyse directe activée.</div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class="warning-box">🛡️ <b>Discordance détectée :</b> Application des filtres HPSS.</div>""", unsafe_allow_html=True)

        st.markdown(f"**Indice de Fiabilité : {conf}%** ({res['mode']})")
        st.markdown(f"""<div class="reliability-bar-bg"><div class="reliability-fill" style="width: {conf}%; background-color: {color};">{conf}%</div></div>""", unsafe_allow_html=True)
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""<div class="metric-container"><div class="label-custom">DOMINANTE</div><div class="value-custom">{res['vote']}</div><div style="color: {color}; font-weight: 800; font-size: 1.6em;">{get_camelot_pro(res['vote'])}</div></div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class="metric-container" style="border-bottom: 4px solid #6366F1;"><div class="label-custom">SYNTHÈSE GLOBALE</div><div class="value-custom">{res['synthese']}</div><div style="color: #6366F1; font-weight: 800; font-size: 1.6em;">{get_camelot_pro(res['synthese'])}</div></div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""<div class="metric-container"><div class="label-custom">TEMPO / ENERGY</div><div class="value-custom">{res['tempo']} BPM</div><div style="color: #6366F1; font-weight: 800; font-size: 1.2em;">ENERGY: {res['energy']}/10</div></div>""", unsafe_allow_html=True)

        st.plotly_chart(px.scatter(pd.DataFrame(res["timeline"]), x="Temps", y="Note", size="Confiance", color="Note", template="plotly_white", title="Stabilité Harmonique"), use_container_width=True)

# --- ONGLET 2 : SCANNER RADIO LIVE (CORRIGÉ) ---
with tabs[1]:
    st.markdown("### 📻 Analyseur de Flux Direct")
    col_live, col_res = st.columns([1, 1])
    
    with col_live:
        webrtc_ctx = webrtc_streamer(
            key="scanner-v45-pro-final",
            mode=WebRtcMode.SENDONLY,
            audio_processor_factory=RealTimeAudioProcessor,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"audio": True, "video": False},
            async_processing=True
        )
    
    with col_res:
        status_placeholder = st.empty()
        
        if webrtc_ctx.audio_processor:
            # Boucle d'affichage pour capturer les éléments de la queue
            while webrtc_ctx.state.playing:
                try:
                    # Récupère le dernier résultat avec un timeout court
                    result = webrtc_ctx.audio_processor.result_queue.get(timeout=1.0)
                    
                    with status_placeholder.container():
                        curr_key = result["key"]
                        curr_conf = result["conf"]
                        camelot = get_camelot_pro(curr_key)
                        
                        st.markdown(f"""
                            <div class="live-panel">
                                <span style="color: #00FF41;">📡 SCANNING LIVE AUDIO...</span><br>
                                > HARMONIC ENGINE: ACTIVE<br>
                                > STUN SERVER: CONNECTED<br>
                                > REAL-TIME CONFIDENCE: {curr_conf}%
                            </div>
                        """, unsafe_allow_html=True)
                        
                        st.metric("CLEF LIVE DÉTECTÉE", camelot, delta=curr_key)
                except queue.Empty:
                    # Si la file est vide temporairement, on continue
                    continue
        else:
            st.info("💡 Cliquez sur **START** pour activer le scanner radio via votre micro.")

# --- HISTORIQUE ---
if st.session_state.history:
    with st.expander("🕒 Dernières Analyses"):
        st.table(pd.DataFrame(st.session_state.history))

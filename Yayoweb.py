import streamlit as st
import streamlit.components.v1 as components
import cv2
import mediapipe as mp
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# Configuración de la web
st.set_page_config(page_title="Yayobot Pro", page_icon="👵")
st.title("👵 Yayobot: Detector de Caídas")

# --- FUNCIÓN DE VOZ (JavaScript) ---
def hablar(texto):
    js_code = f"""
    <script>
    var msg = new SpeechSynthesisUtterance('{texto}');
    msg.lang = 'es-ES';
    msg.rate = 0.9;
    window.speechSynthesis.speak(msg);
    </script>
    """
    components.html(js_code, height=0)

# --- INICIALIZACIÓN DE IA ---
@st.cache_resource
def load_models():
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils
    return mp_pose, pose, mp_drawing

mp_pose, pose, mp_drawing = load_models()

# --- LÓGICA DE PROCESAMIENTO ---
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)

    if results.pose_landmarks:
        # Dibujamos el esqueleto
        mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        # Punto 0 es la nariz. Si baja del 80% de la pantalla (0.8) es caída.
        y_nariz = results.pose_landmarks.landmark[0].y
        if y_nariz > 0.8:
            cv2.putText(img, "!!! CAIDA DETECTADA !!!", (50, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)

    return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- CONFIGURACIÓN DE CÁMARA (MULTISERVIDOR STUN) ---
# Esto ayuda a saltar bloqueos de routers y firewalls
RTC_CONFIG = RTCConfiguration(
    {"iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["stun:stun2.l.google.com:19302"]}
    ]}
)

st.subheader("Estado: Sistema de Vigilancia")

# Botón para activar el sonido (Obligatorio por seguridad del navegador)
if st.button("🔊 ACTIVAR VOZ Y EMPEZAR"):
    hablar("Sistema de seguridad activado. Estoy vigilando.")

webrtc_streamer(
    key="yayo-final-v10",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIG,
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

st.write("---")
st.info("Paso 1: Dale a 'Activar Voz'. Paso 2: Dale a 'START' en la cámara.")

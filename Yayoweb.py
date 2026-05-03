import streamlit as st
import streamlit.components.v1 as components
import cv2
import mediapipe as mp
import av
from streamlit_webrtc import webrtc_streamer

# Configuración inicial
st.set_page_config(page_title="Yayobot Pro", page_icon="👵")
st.title("👵 Yayobot: Protección Total")

# --- MOTOR DE VOZ ---
def lanzar_voz(texto):
    # JavaScript para que el navegador hable
    js = f"""
    <script>
    var msg = new SpeechSynthesisUtterance('{texto}');
    msg.lang = 'es-ES';
    msg.rate = 0.9;
    window.speechSynthesis.speak(msg);
    </script>
    """
    components.html(js, height=0)

# --- CONFIGURACIÓN DE IA ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Interfaz
st.subheader("Estado: Vigilando...")

def callback(frame):
    img = frame.to_ndarray(format="bgr24")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)
    
    caida = False
    if results.pose_landmarks:
        # Dibujar esqueleto
        mp.solutions.drawing_utils.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        # Lógica de caída (Nariz muy abajo)
        if results.pose_landmarks.landmark[0].y > 0.82:
            caida = True
            cv2.putText(img, "ALERTA DE CAIDA", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

    return av.VideoFrame.from_ndarray(img, format="bgr24")

# Cámara
webrtc_streamer(
    key="yayo-final-voz",
    video_frame_callback=callback,
    media_stream_constraints={"video": True, "audio": False},
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Botón de prueba para la voz (El navegador necesita una interacción previa para permitir sonido)
if st.button("Activar sonido de alerta"):
    lanzar_voz("Sistema de ayuda activado. Si te caes, te preguntaré si estás bien.")

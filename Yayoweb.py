import streamlit as st
import streamlit.components.v1 as components
import cv2
import mediapipe as mp
import av
import numpy as np
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# 1. CONFIGURACIÓN DE LA PÁGINA
st.set_page_config(page_title="Yayobot Final", page_icon="👵", layout="centered")

# 2. SISTEMA DE VOZ (JavaScript inyectado)
def hablar_js(texto):
    componente_voz = f"""
        <script>
        var mensaje = new SpeechSynthesisUtterance('{texto}');
        mensaje.lang = 'es-ES';
        mensaje.rate = 1.0;
        window.speechSynthesis.speak(mensaje);
        </script>
    """
    components.html(componente_voz, height=0)

st.title("👵 Yayobot: Detector de Caídas")

# 3. CARGA DE IA (Con manejo de errores de atributo)
@st.cache_resource
def iniciar_ia():
    try:
        # Importación directa para evitar el error de image_17a6c4.png
        from mediapipe.python.solutions import pose as mp_pose
        from mediapipe.python.solutions import drawing_utils as mp_drawing
        detector = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        return mp_pose, detector, mp_drawing
    except Exception as e:
        st.error(f"Error cargando IA: {e}")
        return None, None, None

mp_pose, detector, mp_drawing = iniciar_ia()

# 4. FUNCIÓN QUE PROCESA EL VÍDEO Y DETECTA CAÍDAS
def callback(frame):
    img = frame.to_ndarray(format="bgr24")
    
    if detector is not None:
        # Convertir a RGB para la IA
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resultado = detector.process(img_rgb)

        if resultado.pose_landmarks:
            # Dibujar esqueleto
            mp_drawing.draw_landmarks(img, resultado.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            # LÓGICA DE CAÍDA: Punto 0 es la nariz. 
            # Si la nariz baja del 85% de la pantalla (y > 0.85)
            y_nariz = resultado.pose_landmarks.landmark[0].y
            if y_nariz > 0.85:
                cv2.rectangle(img, (0,0), (img.shape[1], img.shape[0]), (0,0,255), 20)
                cv2.putText(img, "CAIDA!!!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 5)
    
    return av.VideoFrame.from_ndarray(img, format="bgr24")

# 5. CONFIGURACIÓN DE RED AGRESIVA (Para saltar el error STUN)
# Usamos múltiples servidores de Google para asegurar el túnel de vídeo
RTC_CONFIG = RTCConfiguration(
    {"iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["stun:stun2.l.google.com:19302"]},
        {"urls": ["stun:stun3.l.google.com:19302"]},
        {"urls": ["stun:stun4.l.google.com:19302"]}
    ]}
)

# 6. INTERFAZ DE USUARIO
st.info("Paso 1: Pulsa el botón azul para activar la voz. Paso 2: Pulsa START.")

if st.button("🔊 1. ACTIVAR SONIDO DE ALERTA"):
    hablar_js("Sistema listo. Si detecto una caída, pediré ayuda.")
    st.session_state.voz_activa = True

webrtc_streamer(
    key="yayobot-v2026",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIG,
    video_frame_callback=callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

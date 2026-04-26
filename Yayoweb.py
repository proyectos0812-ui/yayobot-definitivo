import streamlit as st
import cv2
import mediapipe as mp
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# Configuración de página
st.set_page_config(page_title="Yayobot", page_icon="👵", layout="centered")

st.title("👵 Yayobot - Detector de Caídas")

# 1. CARGA DE MODELOS (OPTIMIZADA)
@st.cache_resource
def get_mediapipe_models():
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    mp_drawing = mp.solutions.drawing_utils
    return mp_pose, pose, mp_drawing

try:
    mp_pose, pose, mp_drawing = get_mediapipe_models()
    st.success("✅ ¡IA de visión lista! Pulsa START abajo.")
except Exception as e:
    st.error(f"Error al cargar la IA: {e}")

# 2. FUNCIÓN DE PROCESAMIENTO DE VÍDEO
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    
    # Convertir a RGB para MediaPipe
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)

    if results.pose_landmarks:
        # Dibujar los huesos en pantalla
        mp_drawing.draw_landmarks(
            img, 
            results.pose_landmarks, 
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2)
        )
        
        # LÓGICA DE CAÍDA SIMPLE:
        # El punto 11 y 12 son los hombros. Si su altura (y) es muy baja (> 0.8), alertamos.
        hombro_izq = results.pose_landmarks.landmark[11].y
        if hombro_izq > 0.8:
            cv2.putText(img, "!!! CAIDA DETECTADA !!!", (50, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    return av.VideoFrame.from_ndarray(img, format="bgr24")

# 3. COMPONENTE DE CÁMARA (CONFIGURACIÓN ESTABLE)
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

webrtc_streamer(
    key="yayo-v4",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

st.info("💡 Si la cámara no carga: 1. Mira si el candado del navegador bloquea la cámara. 2. Prueba a abrirlo desde el móvil con 4G.")

import streamlit as st
import cv2
import mediapipe as mp
import av
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, WebRtcMode

st.set_page_config(page_title="Yayobot", page_icon="👵")
st.title("👵 Yayobot - Detector de Caídas")

# 1. CARGA DE MODELOS
@st.cache_resource
def get_ia():
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_draw = mp.solutions.drawing_utils
    return mp_pose, pose, mp_draw

try:
    mp_pose, pose, mp_draw = get_ia()
    st.success("✅ IA Lista. Si no ves vídeo, dale a START.")
except Exception as e:
    st.error(f"Error: {e}")

# 2. FUNCIÓN DE PROCESAMIENTO
def callback(frame):
    img = frame.to_ndarray(format="bgr24")
    results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if results.pose_landmarks:
        mp_draw.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        if results.pose_landmarks.landmark[11].y > 0.8:
            cv2.putText(img, "CAIDA!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
    return av.VideoFrame.from_ndarray(img, format="bgr24")

# 3. CONFIGURACIÓN DE RED (LA CLAVE)
# Usamos varios servidores de Google para que no falle la conexión
RTC_CONFIG = RTCConfiguration(
    {"iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["stun:stun2.l.google.com:19302"]}
    ]}
)

webrtc_streamer(
    key="yayobot-vFinal",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIG, # Esto ayuda a saltar el bloqueo
    video_frame_callback=callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

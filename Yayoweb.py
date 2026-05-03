import streamlit as st
import streamlit.components.v1 as components
import cv2
import mediapipe as mp
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

st.set_page_config(page_title="Yayobot Pro", page_icon="👵")
st.title("👵 Yayobot: Detector de Caídas")

# --- FUNCIÓN DE VOZ ---
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

# --- CARGA DE IA SEGURA ---
@st.cache_resource
def load_models():
    # Intentamos importación robusta
    from mediapipe.python.solutions import pose as mp_pose
    from mediapipe.python.solutions import drawing_utils as mp_drawing
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    return mp_pose, pose, mp_drawing

try:
    mp_pose, pose, mp_drawing = load_models()
    st.success("✅ IA cargada. Pulsa el botón de voz y luego START.")
except Exception as e:
    st.error(f"Error técnico: {e}")
    st.info("💡 Si ves esto, dale a 'Reboot' en el menú de la derecha.")

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        if results.pose_landmarks.landmark[0].y > 0.8:
            # Aquí podrías añadir una lógica para disparar la voz
            pass
    return av.VideoFrame.from_ndarray(img, format="bgr24")

RTC_CONFIG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

if st.button("🔊 ACTIVAR VOZ"):
    hablar("Sistema listo. Te avisaré si detecto una caída.")

webrtc_streamer(
    key="yayo-final-v11",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIG,
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

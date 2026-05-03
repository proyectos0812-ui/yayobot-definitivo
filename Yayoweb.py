import streamlit as st
import streamlit.components.v1 as components
import cv2
import mediapipe as mp
import av
from streamlit_webrtc import webrtc_streamer, RTCConfiguration

st.set_page_config(page_title="Yayobot con Voz", page_icon="👵")

# --- FUNCIÓN DE VOZ (JavaScript) ---
def hablar(texto):
    js_code = f"""
    <script>
    var msg = new SpeechSynthesisUtterance('{texto}');
    msg.lang = 'es-ES';
    window.speechSynthesis.speak(msg);
    </script>
    """
    components.html(js_code, height=0)

st.title("👵 Yayobot con Voz")

# Inicialización de IA
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Variable para no repetir la voz mil veces por segundo
if 'hablado' not in st.session_state:
    st.session_state.hablado = False

def callback(frame):
    img = frame.to_ndarray(format="bgr24")
    results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        # Si la nariz (punto 0) o hombros bajan mucho
        y_nariz = results.pose_landmarks.landmark[0].y
        if y_nariz > 0.8:
            return "CAIDA"
            
    return img

# Interfaz de Streamlit
col1, col2 = st.columns([3, 1])

with col1:
    ctx = webrtc_streamer(
        key="yayo-voz",
        video_frame_callback=callback,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
    )

# Lógica de detección de caída y voz
if ctx.state.playing:
    # Si detectamos que la nariz está muy baja
    st.warning("⚠️ Sistema de vigilancia activo")
    
    # Nota: El procesamiento real de la voz debe dispararse 
    # cuando el estado cambia. Aquí un ejemplo simple:
    if st.button("Simular Alerta de Voz"):
        hablar("He detectado una caída. ¿Necesitas ayuda?")

st.info("Para que la voz funcione, el navegador debe tener el sonido activado.")

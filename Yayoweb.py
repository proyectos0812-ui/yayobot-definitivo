import streamlit as st
import os

# Esto evita conflictos de versiones de librerías
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

st.set_page_config(page_title="Yayobot", page_icon="👵")
st.title("👵 Yayobot")

try:
    import cv2
    import mediapipe as mp
    import av
    from streamlit_webrtc import webrtc_streamer

    # Configuramos la IA de forma ultra-simple
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5)
    mp_draw = mp.solutions.drawing_utils
    
    st.success("✅ ¡IA Cargada! Haz clic en START abajo.")

    def callback(frame):
        img = frame.to_ndarray(format="bgr24")
        # Esqueleto azul
        results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if results.pose_landmarks:
            mp_draw.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

    webrtc_streamer(
        key="yayo-final-final",
        video_frame_callback=callback,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False}
    )

except Exception as e:
    st.error(f"Error: {e}")
    st.info("💡 Si sale error de NumPy, espera 1 minuto y dale a 'Reboot app'.")

import streamlit as st
import os

# Forzar implementación de Python para evitar errores de sistema
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

st.set_page_config(page_title="Yayobot", page_icon="👵")
st.title("👵 Yayobot")

try:
    import cv2
    import mediapipe as mp
    import av
    from streamlit_webrtc import webrtc_streamer

    # Inicializar IA
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5)
    mp_draw = mp.solutions.drawing_utils
    
    st.success("✅ ¡IA lista! Pulsa START abajo.")

    def callback(frame):
        img = frame.to_ndarray(format="bgr24")
        # Procesar huesos
        results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if results.pose_landmarks:
            mp_draw.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

    webrtc_streamer(
        key="yayo-ok",
        video_frame_callback=callback,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False}
    )

except Exception as e:
    st.error(f"Error detectado: {e}")
    st.write("Si ves error de libGL, borra el archivo packages.txt de GitHub.")

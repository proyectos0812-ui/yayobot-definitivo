import streamlit as st
import cv2
import mediapipe as mp
import av
from streamlit_webrtc import webrtc_streamer

st.set_page_config(page_title="Yayobot", page_icon="👵")
st.title("👵 Yayobot")

# Cargamos MediaPipe de forma segura
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

st.success("✅ ¡Versión compatible de IA cargada!")

def callback(frame):
    img = frame.to_ndarray(format="bgr24")
    # Esqueleto
    results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_streamer(
    key="yayo-final",
    video_frame_callback=callback,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False}
)

import streamlit as st
import cv2
import mediapipe as mp
import av
from streamlit_webrtc import webrtc_streamer
import os

os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

st.set_page_config(page_title="Yayobot", page_icon="👵")
st.title("👵 Yayobot")

@st.cache_resource
def load_ia():
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils
    return mp_pose, pose, mp_drawing

try:
    mp_pose, pose, mp_drawing = load_ia()
    st.success("✅ ¡IA Lista!")
except Exception as e:
    st.error(f"Fallo al cargar IA: {e}")

def callback(frame):
    img = frame.to_ndarray(format="bgr24")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        if results.pose_landmarks.landmark[11].y > 0.8:
            cv2.putText(img, "CAIDA DETECTADA", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
    return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_streamer(
    key="yayo-cam",
    video_frame_callback=callback,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False}
)

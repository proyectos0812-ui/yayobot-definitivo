import streamlit as st
import os

# Soluciona errores de librerías antiguas
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

st.set_page_config(page_title="Yayobot", page_icon="👵")
st.title("👵 Yayobot")

# --- CARGA SEGURA DE LIBRERÍAS ---
try:
    import cv2
    import mediapipe as mp
    import av
    from streamlit_webrtc import webrtc_streamer
    
    # Inicializar MediaPipe
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    st.success("✅ ¡Todo cargado correctamente! Dale a Start.")

    def video_frame_callback(frame):
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(img_rgb)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            # Detección de caída (hombro por debajo del 80% de la pantalla)
            if results.pose_landmarks.landmark[11].y > 0.8:
                cv2.putText(img, "ALERTA: CAIDA", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

    webrtc_streamer(
        key="yayo-final-v1",
        video_frame_callback=video_frame_callback,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )

except ImportError as e:
    st.error(f"Falta una librería: {e}")
    st.info("Asegúrate de que en requirements.txt pusiste 'opencv-python-headless'.")
except Exception as e:
    st.error(f"Error inesperado: {e}")

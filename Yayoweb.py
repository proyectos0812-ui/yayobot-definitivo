import streamlit as st
import os

# Evitar conflictos de librerías
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

st.set_page_config(page_title="Yayobot", page_icon="👵")
st.title("👵 Yayobot - Detector de Caídas")

try:
    # Importación directa y específica
    import mediapipe as mp
    from mediapipe.python.solutions import pose as mp_pose
    from mediapipe.python.solutions import drawing_utils as mp_drawing
    import cv2
    import av
    from streamlit_webrtc import webrtc_streamer

    # Inicializar el modelo
    pose_model = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    st.success("✅ ¡IA de visión cargada con éxito!")

    def callback(frame):
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose_model.process(img_rgb)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            # Alerta de caída si el hombro baja mucho
            if results.pose_landmarks.landmark[11].y > 0.8:
                cv2.putText(img, "CAIDA DETECTADA", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

    webrtc_streamer(
        key="yayo-final-v5",
        video_frame_callback=callback,
        media_stream_constraints={"video": True, "audio": False}
    )

except Exception as e:
    st.error(f"Error de inicialización: {e}")
    st.info("💡 Consejo: Si el error persiste, borra la app en Streamlit Cloud y créala de nuevo para limpiar la caché.")

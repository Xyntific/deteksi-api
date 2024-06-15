import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode, RTCConfiguration
from ultralytics import YOLO
import cv2
import numpy as np
import tempfile
import os
import requests
import logging

# Setup logging
logging.basicConfig(level=logging.DEBUG)

# Load YOLOv8 model
model = YOLO("best(2).pt")

st.set_page_config(page_title="Sistem Deteksi Api🔥", layout="wide")
st.markdown("<h1 style='text-align: center;'>Sistem Deteksi Api🔥 YOLOv8</h1>", unsafe_allow_html=True)

# Sidebar for navigation
st.sidebar.title("MENU")
page = st.sidebar.radio("", ["Dashboard", "Live Webcam Detection", "Video File Detection"])

# Sidebar for user input
st.sidebar.title("Settings")
confidence_threshold = st.sidebar.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.5)

# Telegram bot details
bot_token = "7440075729:AAHgebp2usoIQYWMjdnMYGDA29DrcT4COA8"
chat_id = "972821613"

# WebRTC configuration with TURN servers
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": "turn:openrelay.metered.ca:80", "username": "openrelayproject", "credential": "openrelayproject"}
    ]
})

# Define a class to process video frames for webcam
class YOLOVideoProcessor(VideoProcessorBase):
    def __init__(self, confidence_threshold):
        self.model = model
        self.confidence_threshold = confidence_threshold

    def send_telegram_notification(self, image, message):
        url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
        _, buffer = cv2.imencode('.jpg', image)
        files = {
            'photo': buffer.tobytes()
        }
        data = {
            "chat_id": chat_id,
            "caption": message
        }
        requests.post(url, data=data, files=files)

    def recv(self, frame):
        try:
            # Convert the frame to an OpenCV image
            image = frame.to_ndarray(format="bgr24")

            # YOLOv8 object detection
            results = self.model(image)
            detection_message = ""

            # Draw bounding boxes and labels on image
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = box.conf[0].item()
                    cls = box.cls[0].item()
                    label = self.model.names[int(cls)]

                    # Check if confidence is above threshold
                    if conf >= self.confidence_threshold:
                        detection_message += f"API TERDETEKSI🔥!!! dengan confidence {conf:.2f}\n"
                        # Draw bounding box
                        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.putText(image, f"Api {conf:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        # Send notification for each detected object
                        self.send_telegram_notification(image, detection_message.strip())

            return frame.from_ndarray(image, format="bgr24")
        except Exception as e:
            logging.error(f"Error processing frame: {e}")
            return frame

# Function to process video file
def process_video_file(video_path, confidence_threshold):
    cap = cv2.VideoCapture(video_path)
    stframe = st.empty()
    notified = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # YOLOv8 object detection
        results = model(frame)
        detection_message = ""

        # Draw bounding boxes and labels on image
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                cls = box.cls[0].item()
                label = model.names[int(cls)]

                # Check if confidence is above threshold
                if conf >= confidence_threshold:
                    detection_message += f"API TERDETEKSI🔥!!! dengan confidence {conf:.2f}\n"
                    # Draw bounding box
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, f"Api {conf:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    # Send notification only once per detection
                    if not notified:
                        send_telegram_notification(frame, detection_message.strip())
                        notified = True

        # Display the resulting frame
        stframe.image(frame, channels="BGR")

    cap.release()

# Function to send notification to Telegram
def send_telegram_notification(image, message):
    url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
    _, buffer = cv2.imencode('.jpg', image)
    files = {
        'photo': buffer.tobytes()
    }
    data = {
        "chat_id": chat_id,
        "caption": message
    }
    requests.post(url, data=data, files=files)

# Main dashboard
if page == "Dashboard":
    st.header("Selamat Datang di Dashboard Deteksi Objek YOLOv8")
    st.write("Gunakan menu di sebelah kiri untuk mengakses deteksi webcam langsung atau deteksi file video.")
    st.image("dashboard_image.jpg")  # Tambahkan gambar dashboard yang relevan
    st.markdown("""
        ### Fitur:
        - **Deteksi Secara Realtime:** Deteksi objek secara instan dalam realtime menggunakan webcam Anda.
        - **Deteksi File Video:** Unggah dan analisis file video untuk deteksi objek.
        - **Notifikasi Telegram:** Terima notifikasi realtime di Telegram untuk objek yang terdeteksi.

        ### Instruksi:
        1. Navigasikan ke halaman "Deteksi Webcam Langsung" untuk mulai mendeteksi objek melalui webcam Anda.
        2. Pergi ke halaman "Deteksi File Video" untuk mengunggah dan memproses file video untuk deteksi objek.
        3. Sesuaikan ambang kepercayaan di sidebar pengaturan untuk mengatur sensitivitas deteksi.

        ---
        **Dikembangkan oleh [Xyntific-]**
    """)
# Live webcam detection
elif page == "Live Webcam Detection":
    st.header("Live Webcam Detection")
    webrtc_ctx = webrtc_streamer(
        key="example",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=lambda: YOLOVideoProcessor(confidence_threshold),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    if webrtc_ctx.state.playing:
        st.write("Webcam is running")
    else:
        st.write("Turn on your webcam to start object detection")

# Video file detection
elif page == "Video File Detection":
    st.header("Video File Detection")
    video_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi", "mkv"])

    if video_file is not None:
        with tempfile.NamedTemporaryFile(delete=False) as tfile:
            tfile.write(video_file.read())
            video_path = tfile.name
        st.video(video_path)
        process_video_file(video_path, confidence_threshold)
        os.unlink(video_path)

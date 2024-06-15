import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode, RTCConfiguration

st.set_page_config(page_title="Sistem Deteksi Api🔥", layout="wide")
st.markdown("<h1 style='text-align: center;'>Sistem Deteksi Api🔥 YOLOv8</h1>", unsafe_allow_html=True)

# Sidebar for navigation
st.sidebar.title("MENU")
page = st.sidebar.radio("", ["Dashboard", "Live Webcam Detection", "Video File Detection"])

# Sidebar for user input
st.sidebar.title("Settings")
confidence_threshold = st.sidebar.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.5)

# WebRTC configuration with TURN servers
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": "turn:openrelay.metered.ca:80", "username": "openrelayproject", "credential": "openrelayproject"}
    ]
})

# Define a class to process video frames for webcam
class SimpleVideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        return frame

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
        video_processor_factory=SimpleVideoProcessor,
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
        # You can re-add video processing code here if needed
        os.unlink(video_path)

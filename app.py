import streamlit as st
import tempfile
import cv2
import torch
import numpy as np
from PIL import Image

# ---------------- AI MODEL ---------------- #

class SimpleAIDetector(torch.nn.Module):
    def __init__(self):
        super(SimpleAIDetector, self).__init__()
        self.flatten = torch.nn.Flatten()
        self.fc = torch.nn.Linear(224 * 224 * 3, 1)

    def forward(self, x):
        x = self.flatten(x)
        x = torch.sigmoid(self.fc(x))
        return x

model = SimpleAIDetector()
model.eval()

def preprocess(frame):
    frame = cv2.resize(frame, (224, 224))
    frame = frame / 255.0
    frame = np.transpose(frame, (2, 0, 1))
    frame = torch.tensor(frame, dtype=torch.float32)
    frame = frame.unsqueeze(0)
    return frame

# ---------------- VIDEO DETECTION ---------------- #

def check_video(video_path):
    cap = cv2.VideoCapture(video_path)
    
    predictions = []
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % 10 == 0:
            input_tensor = preprocess(frame)
            
            with torch.no_grad():
                output = model(input_tensor)
                predictions.append(output.item())
        
        frame_count += 1

    cap.release()

    if len(predictions) == 0:
        return False, 0

    avg_score = np.mean(predictions)

    return avg_score > 0.5, avg_score

# ---------------- IMAGE DETECTION ---------------- #

def check_image(image):
    image = np.array(image)
    input_tensor = preprocess(image)
    
    with torch.no_grad():
        output = model(input_tensor)
    
    score = output.item()
    return score > 0.5, score

# ---------------- STREAMLIT UI ---------------- #

st.title("AI Media Detector (Video + Image)")

uploaded_file = st.file_uploader(
    "Upload a video or image",
    type=["mp4", "avi", "mov", "jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    file_type = uploaded_file.type

    # 🟢 IMAGE
    if "image" in file_type:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image")

        st.write("Analyzing image...")

        result, score = check_image(image)

    # 🔵 VIDEO
    else:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())

        st.video(tfile.name)

        st.write("Analyzing video...")

        result, score = check_video(tfile.name)

    # 📊 RESULT
    st.write(f"Confidence Score: {score:.2f}")

    if result:
        st.error("⚠️ AI GENERATED DETECTED")
    else:
        st.success("✅ NOT AI GENERATED")

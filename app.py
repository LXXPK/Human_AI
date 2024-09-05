import streamlit as st
import cv2
import numpy as np
from keras.models import load_model
import google.generativeai as genai
import os
import speech_recognition as sr
import pyttsx3
import av
from streamlit_webrtc import (
    RTCConfiguration,
    VideoProcessorBase,
    WebRtcMode,
    webrtc_streamer,
)

# --- Configure Google Gemini API ---
genai.configure(api_key="")  # Set your API key from environment variable

generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model_gemini = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
)

# --- Load Emotion Detection Model ---
model = load_model('model_file_30epochs.h5')  # Replace with your model path
faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  # Replace with your haarcascade path
labels_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

# --- Initialize Speech Recognizer and Text-to-Speech Engine ---
recognizer = sr.Recognizer()
tts_engine = pyttsx3.init()

def speak(text):
    tts_engine.say(text)
    tts_engine.runAndWait()

def get_audio():
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)
        try:
            print("Recognizing...")
            text = recognizer.recognize_google(audio)
            print(f"User said: {text}")
            return text
        except sr.UnknownValueError:
            print("Sorry, I did not understand that.")
            # speak("Sorry, I did not understand that.")
            return None

# --- Emotion Detection Function ---
def detect_emotion(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray, 1.3, 3)
    for x, y, w, h in faces:
        sub_face_img = gray[y:y+h, x:x+w]
        resized = cv2.resize(sub_face_img, (48, 48))
        normalize = resized / 255.0
        reshaped = np.reshape(normalize, (1, 48, 48, 1))
        result = model.predict(reshaped)
        label = np.argmax(result, axis=1)[0]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 2)
        cv2.rectangle(frame, (x, y-40), (x+w, y), (50, 50, 255), -1)
        cv2.putText(frame, labels_dict[label], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        return labels_dict[label], frame
    return "Neutral", frame

# --- Gemini Response Function ---
def get_gemini_response(history, question, emotion=None):
    system_prompt = (
       "You are an AI designed to interact with users in a human-like manner, adjusting your responses based on the user's emotional state. "
    "Your goal is to ensure the user understands the topic and feels positive. React to their emotions appropriately: motivate or console them if they are upset, encourage them to ask more questions if they are happy. "
    "Keep responses concise, natural, and empathetic. If the user is sad or expresses any negative emotion, apologize and offer support to make them feel better. "
    "If the user seems unsatisfied after your explanation, console them and provide a response that addresses their concerns in a helpful and positive manner. "
    "Be interactive and avoid repeating the same definitions or explanations. Instead, adapt your responses to keep the conversation engaging and supportive."
)
    
    if emotion:
        system_prompt += f" The user's current emotion is {emotion}. Please respond in a way that is helpful, empathetic, and appropriate to the situation. "
        if emotion in ['Sad', 'Angry', 'Disgust', 'Fear']:
            system_prompt += "If the user's emotion is negative, apologize and provide a supportive response to address their concerns and help them feel better. "
        system_prompt += "Encourage the user to ask more questions or share their thoughts to continue the conversation positively."
    
    history.append({"role": "user", "parts": question})
    
    chat_session = model_gemini.start_chat(
        history=history
    )
    response = chat_session.send_message(question)
    return response.text

# --- Streamlit App Setup ---
st.set_page_config(layout="wide")
st.title("Emotion Interaction")
st.image("bg.jpeg", use_column_width=True)  

# --- Initialize session state for chat history ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "show_history" not in st.session_state:
    st.session_state.show_history = False

# --- Streamlit App Features ---
feature = st.sidebar.selectbox("Select a feature:", ["Emotion Chat", "Real-Time Emotion Detection"])

if feature == "Emotion Chat":
    st.header("Emotion Chat")

    if st.button("Start Chat"):
        st.session_state.chat_history.append({"role": "model", "parts": "Hello! How can I help you today?"})
        st.write("**Bot:** Hello! How can I help you today?")

    if st.button("End Chat"):
        st.session_state.chat_history.append({"role": "model", "parts": "Goodbye! Have a great day."})
        st.write("**Bot:** Goodbye! Have a great day.")

    user_input = st.text_input("Enter your message:")
    if st.button("Send"):
        st.session_state.chat_history.append({"role": "user", "parts": user_input})
        st.write(f"**You:** {user_input}")

        # --- Emotion Detection Loop ---
        emotion = "Neutral" 
        response = get_gemini_response(st.session_state.chat_history, user_input, emotion)
        st.session_state.chat_history.append({"role": "model", "parts": response})
        st.write(f"**Bot:** {response}")

       
        video = cv2.VideoCapture(0)  
        ret, frame = video.read()
        if ret: 
            emotion, processed_frame = detect_emotion(frame)

            
            st.image(processed_frame, caption=f"User Emotion: {emotion}", width=300)

            
            cv2.destroyAllWindows()
            video.release()

            st.session_state.chat_history.append({"role": "user", "parts": f"User Emotion: {emotion}"})
            st.write(f"User Emotion: {emotion}")

            
            response = get_gemini_response(st.session_state.chat_history, user_input, emotion)
            st.session_state.chat_history.append({"role": "model", "parts": response})
            st.write(f"**Bot:** {response}")

    if st.button("Show Conversation History"):
        st.session_state.show_history = not st.session_state.show_history

    if st.session_state.show_history:
        st.header("Conversation History")
        for message in st.session_state.chat_history:
            role = "You" if message["role"] == "user" else "Bot"
            st.write(f"**{role}:** {message['parts']}")

elif feature == "Real-Time Emotion Detection":
    st.header("Real-Time Emotion Detection")
    
    class EmotionProcessor(VideoProcessorBase):
        def __init__(self):
            super().__init__()
            self.detected_emotion = "Neutral"

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            emotion, processed_frame = detect_emotion(img)

            # Update the detected emotion (only if it changes)
            if emotion != self.detected_emotion:
                self.detected_emotion = emotion
                st.session_state.detected_emotion = emotion

            return av.VideoFrame.from_ndarray(processed_frame, format="bgr24")

   
    webrtc_streamer(
        key="example",
        rtc_configuration=RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        ),
        video_processor_factory=EmotionProcessor,
        mode=WebRtcMode.SENDRECV,
        media_stream_constraints={"video": True, "audio": False},
    )

    # Display the detected emotion in a single line
    if "detected_emotion" in st.session_state:
        st.write(f"User Emotion: {st.session_state.detected_emotion}")

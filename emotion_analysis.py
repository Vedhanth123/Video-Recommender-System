"""
Emotion Analysis module for the Emotion-Based Video Recommendation system.
Handles face recognition and emotion detection.
"""

import os
import cv2
import face_recognition
import time
from fer import FER
from collections import Counter
import streamlit as st

from config import FACE_DETECTION_DURATION
from database_utils import get_geolocation, log_emotion_data

class EmotionAnalysis:
    """
    Class for detecting faces, recognizing known faces, and analyzing emotions.
    """
    
    def __init__(self):
        """Initialize the emotion analyzer with empty face encodings and detector."""
        self.known_face_encodings = []
        self.known_face_names = []
        self.known_faces_dir = 'known_faces'
        self.emotion_detector = FER()
        self.load_known_faces()
        
    def load_known_faces(self):
        """
        Load known faces from the directory.
        
        Returns:
            int: Number of known faces loaded.
        """
        self.known_face_encodings = []
        self.known_face_names = []
        
        for filename in os.listdir(self.known_faces_dir):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(self.known_faces_dir, filename)
                image = face_recognition.load_image_file(image_path)
                face_encodings = face_recognition.face_encodings(image)
                if len(face_encodings) > 0:
                    self.known_face_encodings.append(face_encodings[0])
                    name = os.path.splitext(filename)[0]
                    self.known_face_names.append(name)
                else:
                    st.warning(f"No face found in {filename}")
                    
        return len(self.known_face_names)

    def add_new_face(self, frame, face_location, name):
        """
        Add a new face with the given name.
        
        Args:
            frame: Image frame containing the face
            face_location: Tuple of (top, right, bottom, left) coordinates
            name: Name to associate with the face
            
        Returns:
            bool: True if face was added successfully, False otherwise
        """
        if not name or name.strip() == "":
            return False
            
        name = name.strip()
        top, right, bottom, left = face_location
        face_image = frame[top:bottom, left:right]
        
        save_path = os.path.join(self.known_faces_dir, f"{name}.jpg")
        cv2.imwrite(save_path, face_image)
        
        new_image = face_recognition.load_image_file(save_path)
        new_encoding = face_recognition.face_encodings(new_image)
        if len(new_encoding) > 0:
            self.known_face_encodings.append(new_encoding[0])
            self.known_face_names.append(name)
            return True
        else:
            os.remove(save_path)  # Remove the file if face detection failed
            return False
                
    def run_emotion_detection(self, stframe, status_text, progress_bar, duration=FACE_DETECTION_DURATION):
        """
        Run face detection and emotion analysis for specified duration.
        
        Args:
            stframe: Streamlit frame element to display the video feed
            status_text: Streamlit text element to display status messages
            progress_bar: Streamlit progress bar element
            duration: Duration in seconds to run detection
            
        Returns:
            tuple: (dominant_emotions, pending_faces)
                - dominant_emotions: Dictionary of emotions by detected person
                - pending_faces: Dictionary of unidentified faces for later naming
        """
        # Initialize webcam
        video_capture = cv2.VideoCapture(0)
        if not video_capture.isOpened():
            status_text.error("Error: Could not open webcam")
            return {}
            
        start_time = time.time()
        emotion_history = {}  # Dictionary to store emotion history for each detected face
        
        status_text.info(f"Starting emotion detection for {duration} seconds...")
        
        # For new faces that need names
        pending_faces = {}
        
        while True:
            current_time = time.time()
            elapsed_time = current_time - start_time
            remaining_time = max(0, duration - elapsed_time)
            
            # Update progress bar
            progress = min(1.0, elapsed_time / duration)
            progress_bar.progress(progress)
            
            # Break if time exceeded
            if elapsed_time >= duration:
                break

            ret, frame = video_capture.read()
            if not ret:
                status_text.error("Error: Could not read frame from webcam")
                continue
                
            # Resize frame for faster processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            # Process each detected face
            for face_index, face_encoding in enumerate(face_encodings):
                # Scale back up face locations
                top, right, bottom, left = [coord * 4 for coord in face_locations[face_index]]
                
                # See if face matches any known faces
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                name = "Unknown"
                detected_emotion = None
                
                if True in matches:
                    first_match_index = matches.index(True)
                    name = self.known_face_names[first_match_index]

                    # Detect emotion for the recognized face
                    face_image = frame[top:bottom, left:right]
                    emotion_result = self.emotion_detector.detect_emotions(face_image)
                    if emotion_result:
                        emotions = emotion_result[0]['emotions']
                        detected_emotion = max(emotions, key=emotions.get)

                        if name not in emotion_history:
                            emotion_history[name] = []
                        emotion_history[name].append(detected_emotion)
                else:
                    # Store face location for naming later
                    face_img = frame[top:bottom, left:right].copy()
                    face_key = f"face_{face_index}_{int(time.time())}"
                    pending_faces[face_key] = {
                        "image": face_img,
                        "location": (top, right, bottom, left)
                    }
                    name = "Unknown"

                # Draw rectangle and label
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                # Display current emotion above the bounding box
                if detected_emotion:
                    cv2.putText(frame, f"Emotion: {detected_emotion}", (left + 6, top - 10), 
                               cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0), 1)

                # Draw the name below the bounding box
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                cv2.putText(frame, name, (left + 6, bottom - 10), 
                           cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

            # Show progress
            cv2.putText(frame, f"Time remaining: {int(remaining_time)}s", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                       
            # Display the frame in Streamlit
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stframe.image(frame_rgb, channels="RGB")

        video_capture.release()
        
        # Process the emotion history to get dominant emotions
        dominant_emotions = {}
        for name, emotions in emotion_history.items():
            if emotions:
                emotion_counts = Counter(emotions)
                dominant_emotion = emotion_counts.most_common(1)[0][0]
                dominant_emotions[name] = {
                    'emotion': dominant_emotion
                }
                
                # Log to database
                city, country = get_geolocation()
                log_emotion_data(name, dominant_emotion, "", int(time.time()), city, country)
        
        return dominant_emotions, pending_faces

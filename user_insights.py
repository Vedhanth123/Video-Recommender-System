"""
User insights module for the Emotion-Based Video Recommendation system.
Provides analytics and visualization of user emotions and video interactions.
"""

import sqlite3
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
import time

from config import DATABASE_PATH

class UserInsights:
    """Class for generating user insights from interaction data."""
    
    def __init__(self, user_name):
        """
        Initialize the user insights generator.
        
        Args:
            user_name (str): The name of the user to generate insights for
        """
        self.user_name = user_name
        self.emotion_data = None
        self.video_data = None
        self.click_data = None
        self.load_user_data()
    
    def load_user_data(self):
        """Load user interaction data from the database."""
        try:
            conn = sqlite3.connect(DATABASE_PATH)
            
            # Load emotion logs
            query = f"""
            SELECT * FROM emotion_logs 
            WHERE name = '{self.user_name}'
            ORDER BY timestamp ASC
            """
            self.emotion_data = pd.read_sql_query(query, conn)
            if not self.emotion_data.empty:
                # Convert timestamps to datetime
                self.emotion_data['datetime'] = pd.to_datetime(self.emotion_data['timestamp'], unit='s')
            
            # Load video clicks
            try:
                query = f"""
                SELECT * FROM video_clicks 
                WHERE name = '{self.user_name}'
                ORDER BY timestamp ASC
                """
                self.click_data = pd.read_sql_query(query, conn)
                if not self.click_data.empty:
                    # Convert timestamps to datetime
                    self.click_data['datetime'] = pd.to_datetime(self.click_data['timestamp'], unit='s')
            except:
                # Table might not exist yet
                self.click_data = pd.DataFrame()
            
            # Load video interactions
            try:
                query = f"""
                SELECT * FROM video_interactions 
                WHERE name = '{self.user_name}'
                ORDER BY timestamp ASC
                """
                self.video_data = pd.read_sql_query(query, conn)
                if not self.video_data.empty:
                    # Convert timestamps to datetime
                    self.video_data['datetime'] = pd.to_datetime(self.video_data['timestamp'], unit='s')
            except:
                # Table might not exist yet
                self.video_data = pd.DataFrame()
                
            conn.close()
        except Exception as e:
            st.error(f"Error loading user data: {e}")
    
    def has_data(self):
        """Check if there is data available for insights."""
        has_emotion = self.emotion_data is not None and not self.emotion_data.empty
        has_videos = (self.click_data is not None and not self.click_data.empty) or \
                     (self.video_data is not None and not self.video_data.empty)
        return has_emotion or has_videos
    
    def show_emotion_trends(self):
        """Show trends in the user's emotional states over time."""
        if self.emotion_data is None or self.emotion_data.empty:
            st.info("No emotion data available for analysis.")
            return
        
        st.subheader("Emotion Trends Over Time")
        
        # Count emotions
        emotion_counts = self.emotion_data['emotion'].value_counts()
        
        # Create a pie chart of emotion distribution
        fig1 = px.pie(
            values=emotion_counts.values,
            names=emotion_counts.index,
            title=f"Emotion Distribution for {self.user_name}",
            color_discrete_sequence=px.colors.qualitative.Plotly
        )
        
        st.plotly_chart(fig1, use_container_width=True)
        
        # Create a timeline of emotions
        if len(self.emotion_data) > 1:
            # Get most recent week of data
            one_week_ago = datetime.now() - timedelta(days=7)
            recent_data = self.emotion_data[self.emotion_data['datetime'] > pd.Timestamp(one_week_ago)]
            
            if not recent_data.empty:
                st.subheader("Recent Emotional Journey")
                
                # Create emotion timeline
                fig2 = px.line(
                    recent_data, 
                    x='datetime', 
                    y='emotion',
                    title=f"Emotional Journey for {self.user_name}",
                    markers=True
                )
                
                st.plotly_chart(fig2, use_container_width=True)
        
        # Show emotion statistics
        st.subheader("Emotion Statistics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            most_common = emotion_counts.idxmax()
            st.metric("Most Common Emotion", most_common)
            
        with col2:
            if not self.emotion_data.empty and 'datetime' in self.emotion_data:
                latest_record = self.emotion_data.sort_values('datetime', ascending=False).iloc[0]
                latest_emotion = latest_record['emotion']
                st.metric("Latest Recorded Emotion", latest_emotion)
                
        with col3:
            emotion_variety = len(emotion_counts)
            st.metric("Emotional Variety", f"{emotion_variety} emotions")
    
    def show_video_insights(self):
        """Show insights on video watching patterns."""
        if (self.click_data is None or self.click_data.empty) and (self.video_data is None or self.video_data.empty):
            st.info("No video interaction data available for analysis.")
            return
        
        st.subheader("Video Watching Insights")
        
        if self.click_data is not None and not self.click_data.empty:
            # Show video category preferences from clicks
            category_counts = self.click_data['video_category'].value_counts()
            
            fig1 = px.bar(
                x=category_counts.index,
                y=category_counts.values,
                title=f"Video Category Preferences for {self.user_name}",
                labels={'x': 'Category', 'y': 'Count'},
                color=category_counts.values,
                color_continuous_scale='Viridis'
            )
            
            st.plotly_chart(fig1, use_container_width=True)
            
            # Display recent videos viewed
            st.subheader("Recent Videos Viewed")
            recent_videos = self.click_data.sort_values('datetime', ascending=False).head(5)
            
            for _, video in recent_videos.iterrows():
                with st.container():
                    col1, col2 = st.columns([1, 3])
                    
                    with col1:
                        # Display emotion icon
                        emotion = video['emotion']
                        emotion_icon = "😊"  # default
                        if emotion == "happy":
                            emotion_icon = "😊"
                        elif emotion == "sad":
                            emotion_icon = "😢"
                        elif emotion == "angry":
                            emotion_icon = "😠"
                        elif emotion == "fear":
                            emotion_icon = "😨"
                        elif emotion == "surprise":
                            emotion_icon = "😲"
                        elif emotion == "disgust":
                            emotion_icon = "🤢"
                        elif emotion == "neutral":
                            emotion_icon = "😐"
                        
                        st.markdown(f"<h1 style='text-align: center;'>{emotion_icon}</h1>", unsafe_allow_html=True)
                        st.markdown(f"<p style='text-align: center;'>{emotion.capitalize()}</p>", unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"**{video['video_title']}**")
                        st.markdown(f"Category: {video['video_category']}")
                        st.markdown(f"Watched on: {video['datetime'].strftime('%Y-%m-%d %H:%M')}")
                        video_id = video['video_id']
                        st.markdown(f"[Watch again](https://www.youtube.com/watch?v={video_id})")
                    
                    st.markdown("---")
        
        # Show emotion-video correlations if both data sources are available
        if self.click_data is not None and not self.click_data.empty and self.emotion_data is not None and not self.emotion_data.empty:
            st.subheader("Emotion-Video Correlation")
            
            # Count videos watched per emotion
            emotion_video_counts = self.click_data['emotion'].value_counts()
            
            fig2 = px.bar(
                x=emotion_video_counts.index,
                y=emotion_video_counts.values,
                title=f"Videos Watched by Emotion for {self.user_name}",
                labels={'x': 'Emotion', 'y': 'Number of Videos'},
                color=emotion_video_counts.index,
                color_discrete_sequence=px.colors.qualitative.Plotly
            )
            
            st.plotly_chart(fig2, use_container_width=True)
    
    def show_recommendations(self):
        """Show personalized recommendations based on user data."""
        if not self.has_data():
            st.info("Not enough interaction data to generate personalized recommendations.")
            return
        
        st.subheader("Personalized Recommendations")
        
        # Get user's most frequent emotion
        if self.emotion_data is not None and not self.emotion_data.empty:
            most_common_emotion = self.emotion_data['emotion'].value_counts().idxmax()
            
            # Suggest content based on this emotion
            st.markdown(f"Based on your emotion patterns, we recommend content that suits your **{most_common_emotion}** moments.")
            
            # Suggest times of day for watching
            if len(self.emotion_data) > 5 and 'datetime' in self.emotion_data:
                self.emotion_data['hour'] = self.emotion_data['datetime'].dt.hour
                hour_counts = self.emotion_data['hour'].value_counts()
                most_active_hour = hour_counts.idxmax()
                
                # Convert to 12-hour format
                am_pm = "AM" if most_active_hour < 12 else "PM"
                hour_12 = most_active_hour % 12
                hour_12 = 12 if hour_12 == 0 else hour_12
                
                st.markdown(f"You tend to use the system most often around **{hour_12} {am_pm}**.")
        
        # Show category preferences if click data is available
        if self.click_data is not None and not self.click_data.empty:
            favorite_category = self.click_data['video_category'].value_counts().idxmax()
            st.markdown(f"Your favorite video category appears to be **{favorite_category}**.")
            
            # Show personalized tip
            st.markdown("### Personalized Tip")
            st.markdown(f"""
            Based on your interaction patterns:
            
            We suggest exploring more **{favorite_category}** content when you're feeling **{most_common_emotion}**,
            as this combination seems to resonate well with your preferences.
            """)

def authenticate_user_by_face():
    """
    Authenticate a user by recognizing their face.
    
    Returns:
        str or None: The name of the authenticated user, or None if authentication fails
    """
    import cv2
    import face_recognition
    import os
    from emotion_analysis import EmotionAnalysis
    
    # Initialize Streamlit elements
    st.subheader("Face Authentication")
    st.write("Please look at the camera to authenticate.")
    
    video_placeholder = st.empty()
    status_text = st.empty()
    
    # Initialize webcam
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        status_text.error("Error: Could not open webcam")
        return None
    
    # Load known faces
    analyzer = EmotionAnalysis()
    num_faces = analyzer.load_known_faces()
    
    if num_faces == 0:
        status_text.error("No known faces found in the system.")
        video_capture.release()
        return None
    
    status_text.info(f"Looking for a match among {num_faces} known users...")
    
    # Check for max_duration seconds
    max_duration = 10  # seconds
    start_time = time.time()
    authenticated_user = None
    
    while time.time() - start_time < max_duration:
        # Capture frame
        ret, frame = video_capture.read()
        
        if not ret:
            status_text.error("Error: Could not read frame from webcam")
            continue
        
        # Find faces in frame
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        
        # Check each face against known faces
        for face_index, face_encoding in enumerate(face_encodings):
            matches = face_recognition.compare_faces(analyzer.known_face_encodings, face_encoding)
            
            if True in matches:
                first_match_index = matches.index(True)
                authenticated_user = analyzer.known_face_names[first_match_index]
                
                # Draw green rectangle around authenticated user
                top, right, bottom, left = [coord * 4 for coord in face_locations[face_index]]
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                
                # Draw the name below the bounding box
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                cv2.putText(frame, authenticated_user, (left + 6, bottom - 10), 
                           cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
        
        # Display the frame in Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame_rgb, channels="RGB")
        
        # If authenticated, break the loop
        if authenticated_user:
            status_text.success(f"Authentication successful! Welcome, {authenticated_user}!")
            time.sleep(1)  # Show success message for a second
            break
    
    # Release webcam
    video_capture.release()
    
    # Return result
    if not authenticated_user:
        status_text.error("Authentication failed. Could not recognize user.")
    
    return authenticated_user

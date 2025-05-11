"""
Main application module for the Emotion-Based Video Recommendation system.
Creates the Streamlit UI and handles user interactions.
"""
# Import warning suppression (must be first)
import suppress_warnings

import os
import cv2
import time
import threading
import sqlite3
import streamlit as st
from datetime import datetime

# Import configuration
from config import (
    YOUTUBE_API_KEY,
    GEMINI_API_KEY,
    DATABASE_PATH,
    FACE_DETECTION_DURATION
)

# Import modules
from emotion_analysis import EmotionAnalysis
from youtube_recommender import YouTubeRecommender
from gemini_explainer import GeminiExplainer
from recommendation_engine import RecommendationEngine
from emotion_constants import EMOTION_DESCRIPTIONS
from database_utils import ensure_emotion_logs_schema, log_video_click, get_geolocation, ensure_all_database_tables
from ml_recommender import MLRecommendationEngine
from user_insights import UserInsights, authenticate_user_by_face

def main():
    """Main application function for the Streamlit app."""
    
    # Check if the app should exit
    if 'should_exit' in st.session_state and st.session_state.should_exit:
        st.info("Application is shutting down...")
        # Use threading to delay the exit slightly so the message shows up
        import threading
        threading.Timer(1.0, lambda: os._exit(0)).start()
        return
    
    st.set_page_config(
        page_title="Emotion-Based Video Recommendation",
        page_icon="📺",
        layout="wide",
    )
    
    # Ensure all database tables have the correct schema
    ensure_all_database_tables()
    
    # Initialize session state for storing data between reruns
    if 'emotion_analyzer' not in st.session_state:
        st.session_state.emotion_analyzer = EmotionAnalysis()
    
    if 'youtube_recommender' not in st.session_state:
        st.session_state.youtube_recommender = YouTubeRecommender(YOUTUBE_API_KEY)
    
    if 'gemini_explainer' not in st.session_state:
        st.session_state.gemini_explainer = GeminiExplainer(GEMINI_API_KEY)
    
    if 'recommendation_engine' not in st.session_state:
        st.session_state.recommendation_engine = RecommendationEngine(
            st.session_state.youtube_recommender,
            st.session_state.gemini_explainer
        )
    
    if 'ml_recommender' not in st.session_state:
        st.session_state.ml_recommender = MLRecommendationEngine()
    
    if 'detected_emotions' not in st.session_state:
        st.session_state.detected_emotions = {}
    
    if 'current_user' not in st.session_state:
        st.session_state.current_user = None
    
    if 'pending_faces' not in st.session_state:
        st.session_state.pending_faces = {}
        
    # App title and description
    st.title("Emotion-Based Video Recommendation")
    st.markdown("This application detects your facial emotions and recommends YouTube videos based on your mood.")
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Face Detection", "Video Recommendations", "User Insights", "Admin Panel"])
    
    # Face Detection Tab
    with tab1:
        st.header("Face Detection")
        
        col1, col2 = st.columns([3, 1])
        
        with col2:
            st.subheader("Controls")
            duration = st.slider("Detection Duration (seconds)", 
                               min_value=5, max_value=60, value=FACE_DETECTION_DURATION)
            
            start_detection = st.button("Start Face Detection", use_container_width=True)
            refresh_faces = st.button("Refresh Known Faces", use_container_width=True)
            
            if refresh_faces:
                num_faces = st.session_state.emotion_analyzer.load_known_faces()
                st.success(f"Loaded {num_faces} known faces")
            
            # Display known faces
            st.subheader("Known Faces")
            if st.session_state.emotion_analyzer.known_face_names:
                for name in st.session_state.emotion_analyzer.known_face_names:
                    st.text(f"• {name}")
            else:
                st.info("No known faces yet")
        
        with col1:
            # Video feed and results
            video_placeholder = st.empty()
            status_text = st.empty()
            progress_bar = st.progress(0)
            results_area = st.expander("Detection Results", expanded=True)
            
            # Process any pending faces that need names
            if st.session_state.pending_faces:
                st.subheader("Unidentified Faces")
                st.write("Please name these detected faces:")
                
                cols = st.columns(3)
                faces_to_remove = []
                
                for i, (face_key, face_data) in enumerate(st.session_state.pending_faces.items()):
                    col_idx = i % 3
                    with cols[col_idx]:
                        st.image(face_data["image"], caption=f"Face #{i+1}", width=150)
                        face_name = st.text_input(f"Name for Face #{i+1}", key=f"face_name_{face_key}")
                        
                        if st.button(f"Save Face #{i+1}", key=f"save_face_{face_key}"):
                            if face_name and face_name.strip():
                                success = st.session_state.emotion_analyzer.add_new_face(
                                    cv2.cvtColor(face_data["image"], cv2.COLOR_RGB2BGR), 
                                    (0, face_data["image"].shape[1], face_data["image"].shape[0], 0),
                                    face_name
                                )
                                if success:
                                    st.success(f"Face saved as {face_name}")
                                    faces_to_remove.append(face_key)
                                else:
                                    st.error("Failed to save face")
                            else:
                                st.warning("Please enter a name")
                
                # Remove processed faces
                for key in faces_to_remove:
                    del st.session_state.pending_faces[key]
                
                if faces_to_remove:
                    st.experimental_rerun()
            
            if start_detection:
                with results_area:
                    st.info("Detection started. Please look at the camera.")
                    
                # Reset state
                st.session_state.detected_emotions = {}
                
                # Run detection
                detected_emotions, pending_faces = st.session_state.emotion_analyzer.run_emotion_detection(
                    video_placeholder, status_text, progress_bar, duration
                )
                
                # Store results
                st.session_state.detected_emotions = detected_emotions
                st.session_state.pending_faces = pending_faces
                
                # Show results
                with results_area:
                    if detected_emotions:
                        st.success("Detection Complete!")
                        st.subheader("Detection Results:")
                        
                        for name, data in detected_emotions.items():
                            st.markdown(f"**Name:** {name}")
                            st.markdown(f"**Dominant Emotion:** {data['emotion']}")
                            st.markdown("---")
                            
                        st.info("You can now go to the 'Video Recommendations' tab to get personalized recommendations.")
                    else:
                        st.warning("No faces with emotions were detected. Please try again.")
    
    # Video Recommendations Tab
    with tab2:
        st.header("Video Recommendations")
        
        # User selection
        users = st.session_state.emotion_analyzer.known_face_names
        if users:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                selected_user = st.selectbox(
                    "Select User",
                    options=users,
                    index=0 if st.session_state.current_user not in users else users.index(st.session_state.current_user)
                )
            
            with col2:
                get_recommendations = st.button("Get Recommendations", use_container_width=True)
            
            if get_recommendations:
                st.session_state.current_user = selected_user
                  # Check if we have emotion data for this user
                user_emotion = None
                if selected_user in st.session_state.detected_emotions:
                    user_emotion = st.session_state.detected_emotions[selected_user]
                else:
                    # Query from database (most recent emotion)
                    try:
                        ensure_emotion_logs_schema()
                        
                        conn = sqlite3.connect(DATABASE_PATH)
                        cursor = conn.cursor()
                        cursor.execute("""
                            SELECT emotion FROM emotion_logs 
                            WHERE name = ? ORDER BY timestamp DESC LIMIT 1
                        """, (selected_user,))
                        result = cursor.fetchone()
                        conn.close()
                        
                        if result:
                            user_emotion = {'emotion': result[0]}
                    except Exception as e:
                        st.error(f"Database error: {e}")
                        
                if not user_emotion:
                    st.warning(f"No emotion data available for {selected_user}. Please run face detection first.")
                else:
                    with st.spinner(f"Loading recommendations for {selected_user}..."):
                        recommendations = st.session_state.recommendation_engine.get_recommendations_for_user(
                            selected_user, user_emotion
                        )
                        
                        if recommendations:
                            # Apply ML ranking if we have enough historical data
                            ml_rankings = st.session_state.ml_recommender.rank_recommendations(
                                selected_user, 
                                recommendations, 
                                current_emotion=user_emotion['emotion'].lower()
                            )
                            
                            # If ML ranking was successful, use it. Otherwise, use the default recommendations
                            if ml_rankings:
                                recommendations = ml_rankings
                                st.success(f"Found {len(recommendations)} recommendations based on {user_emotion['emotion']} emotion and your previous interactions")
                            else:
                                st.success(f"Found {len(recommendations)} recommendations based on {user_emotion['emotion']} emotion")
                            
                            # Display user info
                            st.markdown(f"### Recommendations for: {selected_user}")
                            st.markdown(f"**Detected Emotion:** {user_emotion['emotion']}")
                            st.markdown(f"**Recommendation Focus:** {EMOTION_DESCRIPTIONS.get(user_emotion['emotion'].lower(), '')}")
                            st.markdown("---")
                            
                            # Display recommendations in a grid
                            cols_per_row = 2
                            for i in range(0, len(recommendations), cols_per_row):
                                cols = st.columns(cols_per_row)
                                for j in range(cols_per_row):
                                    if i + j < len(recommendations):
                                        video = recommendations[i + j]
                                        with cols[j]:
                                            st.markdown(f"#### {video['title']}")
                                            st.image(video['thumbnail'])
                                            
                                            # Display the Gemini-generated explanation in a highlighted box
                                            st.markdown(f"""
                                            <div style="background-color: #f0f7fb; border-left: 5px solid #2196F3; padding: 10px; margin-bottom: 10px;">
                                                <strong>Why this might help:</strong> {video['explanation']}
                                            </div>
                                            """, unsafe_allow_html=True)
                                            
                                            st.markdown(f"**Channel:** {video['channel']}")
                                            st.markdown(f"**Views:** {video['views']:,} • **Duration:** {video['duration']}")
                                            st.markdown(f"**Description:** {video['description'][:100]}...")
                                              # YouTube embed or link with click tracking
                                            video_id = video['id']
                                            video_url = f"https://www.youtube.com/watch?v={video_id}"
                                            
                                            # Use a regular button for logging + a direct link for opening
                                            if st.button(f"Watch on YouTube", key=f"watch_button_{video_id}_{i}_{j}"):
                                                # Log the video click
                                                city, country = get_geolocation()
                                                timestamp = int(time.time())
                                                
                                                # Extract a category from the video's description or title
                                                video_category = "Unknown"
                                                if "music" in video['title'].lower() or "song" in video['title'].lower():
                                                    video_category = "Music"
                                                elif "game" in video['title'].lower() or "gaming" in video['title'].lower():
                                                    video_category = "Gaming"
                                                elif "news" in video['title'].lower():
                                                    video_category = "News"
                                                elif "tutorial" in video['title'].lower() or "how to" in video['title'].lower():
                                                    video_category = "Education"
                                                
                                                # Log the click in the database
                                                log_video_click(
                                                    selected_user,
                                                    video_id,
                                                    video['title'],
                                                    video_category,
                                                    user_emotion['emotion'].lower(),
                                                    timestamp,
                                                    city,
                                                    country
                                                )
                                                
                                                # Update the ML model with the new click data
                                                st.session_state.ml_recommender.build_user_profile(selected_user)
                                                
                                                # Provide a direct link that will work
                                                st.markdown(f"""
                                                <div style="text-align: center; margin-top: 10px;">
                                                    <a href="{video_url}" target="_blank">
                                                        <button style="
                                                            background-color: #FF0000;
                                                            color: white;
                                                            border: none;
                                                            border-radius: 4px;
                                                            padding: 10px 16px;
                                                            font-size: 16px;
                                                            cursor: pointer;">
                                                            Click here to open YouTube
                                                        </button>
                                                    </a>
                                                </div>
                                                """, unsafe_allow_html=True)
                                            
                                            st.markdown("---")
                        else:
                            st.error("Failed to get recommendations. Please try again.")
        else:
            st.warning("No known users found. Please add users via Face Detection first.")
    
    # User Insights Tab
    with tab3:
        st.header("User Insights Dashboard")
        st.markdown("View personalized analytics about your emotions and video interactions.")
        
        # Face authentication for accessing insights
        if 'insights_authenticated_user' not in st.session_state or not st.session_state.insights_authenticated_user:
            st.warning("Please authenticate with your face to view your insights.")
            
            authenticate_col1, authenticate_col2 = st.columns([3, 1])
            
            with authenticate_col1:
                start_auth = st.button("Start Face Authentication", use_container_width=True)
                
            if start_auth:
                authenticated_user = authenticate_user_by_face()
                if authenticated_user:
                    st.session_state.insights_authenticated_user = authenticated_user
                    st.experimental_rerun()  # Rerun to refresh the page with authentication
        else:
            # Already authenticated, show insights
            authenticated_user = st.session_state.insights_authenticated_user
            
            st.success(f"Authenticated as: {authenticated_user}")
            
            # Add a logout button
            if st.button("Logout from Insights", key="insights_logout"):
                st.session_state.insights_authenticated_user = None
                st.experimental_rerun()
                
            # Initialize and display user insights
            insights = UserInsights(authenticated_user)
            
            if not insights.has_data():
                st.warning(f"No interaction data found for {authenticated_user}. Please use the system more to generate insights.")
            else:
                # Create insights sections
                insight_section = st.radio(
                    "Choose insight category:",
                    options=["Emotion Analysis", "Video Watching Patterns", "Personalized Recommendations"],
                    horizontal=True
                )
                
                if insight_section == "Emotion Analysis":
                    insights.show_emotion_trends()
                    
                elif insight_section == "Video Watching Patterns":
                    insights.show_video_insights()
                    
                elif insight_section == "Personalized Recommendations":
                    insights.show_recommendations()
    
    # Admin Panel Tab
    with tab4:
        st.header("Admin Control Panel")
        st.markdown("Use this panel to manage the application.")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.subheader("Application Control")
            st.markdown("""
            This panel provides administrative controls for the application. Use the buttons 
            to the right to perform administrative tasks.
            
            **Warning:** The Exit Application button will shut down the Streamlit server.
            """)
            
            # Add more admin features here in the future
            
        with col2:
            st.subheader("Actions")
            if st.button("Exit Application", use_container_width=True, type="primary"):
                st.session_state.should_exit = True
                st.rerun()

if __name__ == "__main__":
    main()

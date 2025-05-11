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
import pandas as pd
import plotly.express as px
from datetime import datetime

# Import configuration
from config import (
    YOUTUBE_API_KEY,
    DATABASE_PATH,
    FACE_DETECTION_DURATION
)

# Import modules
from emotion_analysis import EmotionAnalysis
from youtube_recommender import YouTubeRecommender
from local_explainer import LocalExplainer
from recommendation_engine import RecommendationEngine
from emotion_constants import EMOTION_DESCRIPTIONS
from database_utils import ensure_emotion_logs_schema, log_video_click, get_geolocation, ensure_all_database_tables
from ml_recommender import MLRecommendationEngine
from nn_recommender import NNRecommender
from user_insights import UserInsights, authenticate_user_by_face
from integration import initialize_enhancement_system

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
    if 'explainer' not in st.session_state:
        st.session_state.explainer = LocalExplainer()
    
    if 'recommendation_engine' not in st.session_state:
        st.session_state.recommendation_engine = RecommendationEngine(
            st.session_state.youtube_recommender,
            st.session_state.explainer
        )
    
    if 'ml_recommender' not in st.session_state:
        st.session_state.ml_recommender = MLRecommendationEngine()
    
    if 'neural_recommender' not in st.session_state:
        st.session_state.neural_recommender = NNRecommender()
    
    # Initialize the enhancement system
    initialize_enhancement_system()
    
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
        
        # Show if neural network is available
        nn_ready = False
        if 'neural_recommender' in st.session_state and st.session_state.neural_recommender.is_ready():
            nn_ready = True
            st.sidebar.success("🧠 Neural Network Recommendation System: Active")
        else:
            st.sidebar.info("🔄 Using traditional recommendation system")
            if 'neural_recommender' in st.session_state:
                st.sidebar.markdown("To activate neural recommendations, use the Admin Panel to train the model")
        
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
                                            
                                            # Add badge for neural network recommendation
                                            if 'nn_category' in video and 'nn_confidence' in video:
                                                confidence = int(video['nn_confidence'] * 100)
                                                st.markdown(f"""
                                                <div style="background-color: #6a0dad; color: white; padding: 5px 10px; 
                                                            border-radius: 15px; display: inline-block; font-size: 0.8em; margin-bottom: 10px;">
                                                    🧠 AI Recommendation ({confidence}% match)
                                                </div>
                                                """, unsafe_allow_html=True)
                                            
                                            # Show enhancement badges if available
                                            if 'enhancement_tags' in video and video['enhancement_tags']:
                                                for tag in video['enhancement_tags']:
                                                    if tag == 'time-optimized':
                                                        st.markdown(f"""
                                                        <div style="background-color: #2E8B57; color: white; padding: 5px 10px; 
                                                                    border-radius: 15px; display: inline-block; font-size: 0.8em; margin-bottom: 10px; margin-right: 5px;">
                                                            ⏰ Time-Optimized
                                                        </div>
                                                        """, unsafe_allow_html=True)
                                                    elif tag == 'collaborative':
                                                        st.markdown(f"""
                                                        <div style="background-color: #1E90FF; color: white; padding: 5px 10px; 
                                                                    border-radius: 15px; display: inline-block; font-size: 0.8em; margin-bottom: 10px; margin-right: 5px;">
                                                            👥 Collaborative
                                                        </div>
                                                        """, unsafe_allow_html=True)
                                            
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
                                            
                                            # YouTube embed or link
                                            video_id = video['id']
                                            st.markdown(f'''
                                            <div style="text-align: center;">
                                                <a href="https://www.youtube.com/watch?v={video_id}" target="_blank">
                                                    <button style="
                                                        background-color: #FF0000;
                                                        color: white;
                                                        border: none;
                                                        border-radius: 4px;
                                                        padding: 10px 16px;
                                                        font-size: 16px;
                                                        cursor: pointer;
                                                        display: inline-flex;
                                                        align-items: center;
                                                        margin: 10px 0;">
                                                        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="white" style="margin-right: 8px;">
                                                            <path d="M10,16.5V7.5L16,12M12,2A10,10 0 0,0 2,12A10,10 0 0,0 12,22A10,10 0 0,0 22,12A10,10 0 0,0 12,2Z" />
                                                        </svg>
                                                        Watch on YouTube
                                                    </button>
                                                </a>
                                            </div>
                                            ''', unsafe_allow_html=True)
                                            
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
            
            st.subheader("Neural Network Training")
            st.markdown("""
            Train the neural network model with the current interaction data. This will improve 
            recommendations based on emotional patterns and viewing history.
            
            **Note:** You'll need at least 10 video interactions before training can begin.
            """)            # Neural Network Training Instructions
            st.info("""
            **Neural Network Training Instructions**
            
            The neural network model must be trained separately before using the app. Follow these steps:
            
            1. Ensure you have at least 10 video interactions in the database
            2. Close this application
            3. Run the Training Script: `Train_Model.bat` 
            4. Restart this application after training completes
            
            This separation allows for more efficient training without consuming application resources.
            """)
            
            # Show neural network status
            if 'neural_recommender' in st.session_state and st.session_state.neural_recommender.is_ready():
                st.success("✅ Neural Network model is loaded and ready")
                
                # Show model information
                st.markdown("**Model Information**")
                st.markdown("- Model file: `models/emotion_video_nn.h5`")
                st.markdown("- Encoders file: `models/emotion_video_encoders.pkl`")
                st.markdown("- Status: Ready for predictions")
                
                # Display training history chart if available
                history_plot_path = os.path.join("models", "training_history.png")
                if os.path.exists(history_plot_path):
                    st.subheader("Training History")
                    st.image(history_plot_path, caption="Neural Network Training Progress", use_column_width=True)
                    st.markdown("*This chart shows the model's accuracy and loss during training*")
            else:
                st.warning("⚠️ Neural Network model is not available")
                st.markdown("""
                No trained model found. Please run the training script as described above.
                The system will fall back to traditional recommendations until a model is trained.
                """)
            
        with col2:
            st.subheader("Actions")
            
            # Open training script button
            if os.path.exists("Train_Model.bat"):
                if st.button("Run Neural Network Training Script", use_container_width=True):
                    # Launch the training script in a separate window
                    os.startfile("Train_Model.bat")
                    st.info("Training script opened in a separate window")
                
                st.markdown("""
                **Note:** The training script will:
                - Process your interaction data
                - Train a neural network model
                - Save the model for future predictions
                - Create a visualization of training progress
                """)
            
            # Refresh button
            if st.button("Refresh Model Status", use_container_width=True):
                st.experimental_rerun()
            
            st.markdown("---")
            
            st.markdown("---")
            
            if st.button("Exit Application", use_container_width=True, type="primary"):
                st.session_state.should_exit = True
                st.rerun()

if __name__ == "__main__":
    main()

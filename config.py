"""
Configuration file for the Emotion-Based Video Recommendation system.
Contains API keys, paths, and other configuration settings.
"""

import os
import streamlit as st

# API Keys
YOUTUBE_API_KEY = "AIzaSyAVer6TDzEBXthg6l4LtxJSMW6ybHGqB8c"  # Replace with your actual API key
# GEMINI_API_KEY is no longer used - using local model instead

# Database settings
DATABASE_PATH = "user_data.db" 

# Application settings
MAX_RESULTS = 10
FACE_DETECTION_DURATION = 8  # Default seconds to run face detection
OFFLINE_MODE = True  # Set to True to use local models instead of online APIs
"""
Integration module to connect the RecommendationEnhancer with the existing recommendation system.
This module handles the integration between existing recommendation components and the new enhancement features.
"""

import streamlit as st
from datetime import datetime

from recommendation_enhancement import RecommendationEnhancer
from config import DATABASE_PATH

def integrate_enhancements(recommendations, user_name, current_emotion=None):
    """
    Integrate enhanced recommendations with the existing recommendation system.
    
    Args:
        recommendations (list): Original video recommendations
        user_name (str): Current user's name
        current_emotion (str, optional): Current detected emotion
        
    Returns:
        list: Enhanced and reordered video recommendations
    """
    # Initialize enhancer if not already in session state
    if 'recommendation_enhancer' not in st.session_state:
        st.session_state.recommendation_enhancer = RecommendationEnhancer(DATABASE_PATH)
    
    enhancer = st.session_state.recommendation_enhancer
    
    # Get current hour for time-of-day awareness
    current_hour = datetime.now().hour
    
    # Apply the full enhancement pipeline
    enhanced_recommendations = enhancer.rerank_recommendations(
        user_name=user_name,
        videos=recommendations,
        current_emotion=current_emotion,
        current_hour=current_hour
    )
    
    # Add enhancement markers to the videos for UI display
    for video in enhanced_recommendations:
        # Add enhancement indicators
        if video.get('time_score', 0) > 0.7:
            video['enhancement_tags'] = video.get('enhancement_tags', []) + ['time-optimized']
            
        if video.get('collab_score', 0) > 0:
            video['enhancement_tags'] = video.get('enhancement_tags', []) + ['collaborative']
            
        if 'final_score' in video:
            video['enhancement_score'] = video['final_score']
    
    return enhanced_recommendations

def update_recommendation_engine():
    """
    Update the RecommendationEngine class to use the enhancer.
    This function patches the existing recommendation engine to incorporate enhancements.
    """
    from recommendation_engine import RecommendationEngine
    
    # Save the original get_recommendations_for_user method
    if not hasattr(RecommendationEngine, '_original_get_recommendations'):
        RecommendationEngine._original_get_recommendations = RecommendationEngine.get_recommendations_for_user
    
    # Define a new method that enhances the original recommendations
    def enhanced_get_recommendations(self, user_name, emotion_data):
        # Get original recommendations
        original_recommendations = RecommendationEngine._original_get_recommendations(self, user_name, emotion_data)
        
        # Apply enhancements
        if original_recommendations:
            current_emotion = emotion_data.get('emotion', '').lower() if emotion_data else None
            return integrate_enhancements(original_recommendations, user_name, current_emotion)
        
        return original_recommendations
    
    # Replace the method in the RecommendationEngine class
    RecommendationEngine.get_recommendations_for_user = enhanced_get_recommendations
    
    return True

def initialize_enhancement_system():
    """
    Initialize the enhancement system and integrate it with the existing system.
    Call this function during application startup.
    
    Returns:
        bool: True if integration was successful
    """
    # Create enhancer instance in session state if it doesn't exist
    if 'recommendation_enhancer' not in st.session_state:
        st.session_state.recommendation_enhancer = RecommendationEnhancer(DATABASE_PATH)
    
    # Update the recommendation engine to use enhancements
    update_result = update_recommendation_engine()
    
    return update_result

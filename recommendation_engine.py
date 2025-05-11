"""
Recommendation Engine module for the Emotion-Based Video Recommendation system.
Combines emotion data with video content preferences to provide personalized recommendations.
"""

import random
from datetime import datetime, timedelta
import streamlit as st

from database_utils import get_geolocation
from emotion_constants import EMOTION_VIDEO_MAPPING
from config import YOUTUBE_API_KEY

class RecommendationEngine:
    """
    Class for generating personalized recommendations based on user emotions.
    Combines YouTube recommendations with Gemini explanations.
    """
    
    def __init__(self, youtube_recommender, gemini_explainer):
        """
        Initialize the recommendation engine with recommender components.
        
        Args:
            youtube_recommender: An instance of YouTubeRecommender
            gemini_explainer: An instance of GeminiExplainer
        """
        self.youtube = youtube_recommender
        self.gemini = gemini_explainer
        self.cached_recommendations = {}
        
    def get_recommendations_for_user(self, user_name, emotion_data):
        """
        Generate personalized video recommendations based on emotion data.
        
        Args:
            user_name (str): Name of the user
            emotion_data (dict): Emotion data for the user
            
        Returns:
            list: List of personalized video recommendations with explanations
        """
        # Check cache first (for returning users)
        if user_name in self.cached_recommendations:
            cache_time, recommendations = self.cached_recommendations[user_name]
            # Cache valid for 1 hour
            if datetime.now() - cache_time < timedelta(hours=1):
                return recommendations
        
        if not emotion_data:
            st.warning(f"No emotion data available for {user_name}")
            return []
            
        # Extract data
        dominant_emotion = emotion_data['emotion'].lower()
        
        # Get emotion-based video preferences
        emotion_settings = EMOTION_VIDEO_MAPPING.get(dominant_emotion, EMOTION_VIDEO_MAPPING["neutral"])
        
        # Build search queries
        all_recommendations = []
        
        # Get geolocation for regional content
        city, country = get_geolocation()
        
        # Try each keyword
        for keyword in emotion_settings['keywords']:
            full_query = keyword
            
            # Sometimes add location-specific content (50% chance)
            if country and country != "Unknown Country" and random.random() > 0.5:
                full_query += f" {country}"
                
            # Search in each relevant category
            for category in emotion_settings['categories']:
                videos = self.youtube.search_videos(
                    query=full_query,
                    category_id=category,
                    max_results=3
                )
                all_recommendations.extend(videos)
        
        # Remove duplicate videos (by ID)
        unique_recommendations = []
        seen_ids = set()
        
        for video in all_recommendations:
            if video['id'] not in seen_ids:
                seen_ids.add(video['id'])
                unique_recommendations.append(video)
        
        # Sort by views and limit to 20 recommendations
        unique_recommendations.sort(key=lambda x: x['views'], reverse=True)
        final_recommendations = unique_recommendations[:20]
        
        # Add explanations to each recommendation
        for video in final_recommendations:
            video['explanation'] = self.gemini.explain_recommendation(
                video, 
                user_name, 
                dominant_emotion
            )
        
        # Cache the recommendations
        self.cached_recommendations[user_name] = (datetime.now(), final_recommendations)
        
        return final_recommendations
